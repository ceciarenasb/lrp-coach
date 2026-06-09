"""
Garmin Connect sync — authentication, FIT download, activity parsing.

Auth flow:
  First login: email + password (may trigger Garmin MFA via email/TOTP).
  Subsequent calls: garth reuses cached OAuth tokens from data/garmin_token/.
  Credentials stored: email only in data/garmin_creds.json (password never persisted).
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR        = Path(__file__).parent.parent / "data"
GARMIN_TOKEN_DIR = _DATA_DIR / "garmin_token"
GARMIN_CREDS_FILE = _DATA_DIR / "garmin_creds.json"


# ── Credential helpers ─────────────────────────────────────────────────────

def save_email(email: str) -> None:
    _DATA_DIR.mkdir(exist_ok=True)
    GARMIN_CREDS_FILE.write_text(json.dumps({"email": email}))


def load_email() -> Optional[str]:
    try:
        return json.loads(GARMIN_CREDS_FILE.read_text()).get("email")
    except Exception:
        return None


def is_authenticated() -> bool:
    """True when garth token files exist — no need to re-enter credentials."""
    return GARMIN_TOKEN_DIR.exists() and any(GARMIN_TOKEN_DIR.iterdir())


def clear_auth() -> None:
    """Remove cached tokens (forces re-login on next sync)."""
    import shutil
    if GARMIN_TOKEN_DIR.exists():
        shutil.rmtree(GARMIN_TOKEN_DIR)
    if GARMIN_CREDS_FILE.exists():
        GARMIN_CREDS_FILE.unlink()


# ── Client factory ─────────────────────────────────────────────────────────

_pending_mfa_client = None  # held between connect() and submit_mfa()


def _make_client(email: str = "", password: str = "") -> "Garmin":
    from garminconnect import Garmin
    return Garmin(email=email or (load_email() or ""), password=password)


def connect(email: str, password: str) -> tuple[bool, str]:
    """
    Authenticate with Garmin Connect.
    Returns (success, message).
    On success garth tokens are dumped to GARMIN_TOKEN_DIR.
    Never pass tokenstore on fresh login — garth raises FileNotFoundError
    if the directory exists but token files are absent.
    """
    global _pending_mfa_client
    GARMIN_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        client = _make_client(email, password)
        client.login()
        client.garth.dump(str(GARMIN_TOKEN_DIR))
        save_email(email)
        _pending_mfa_client = None
        return True, f"Connected as {email}"
    except Exception as exc:
        msg = str(exc)
        if "NEEDS_MFA" in msg or "MFA" in msg.upper() or "factor" in msg.lower():
            _pending_mfa_client = client  # garth keeps OAuth state in this object
            return False, "MFA_REQUIRED"
        _pending_mfa_client = None
        return False, f"Login failed: {msg}"


def submit_mfa(email: str, password: str, mfa_code: str) -> tuple[bool, str]:
    """Complete login when Garmin requested an MFA code."""
    global _pending_mfa_client
    GARMIN_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    # Reuse the client from connect() so garth's in-progress OAuth state is intact.
    client = _pending_mfa_client or _make_client(email, password)
    try:
        client.login(mfa_code=mfa_code)
        client.garth.dump(str(GARMIN_TOKEN_DIR))
        save_email(email)
        _pending_mfa_client = None
        return True, f"Connected as {email}"
    except Exception as exc:
        return False, f"MFA failed: {exc}"


def get_client() -> Optional["Garmin"]:
    """Return an authenticated client using cached tokens, or None."""
    if not is_authenticated():
        return None
    try:
        client = _make_client()
        client.garth.load(str(GARMIN_TOKEN_DIR))
        return client
    except Exception as exc:
        logger.warning("Token refresh failed: %s", exc)
        return None


# ── Activity sync ──────────────────────────────────────────────────────────

def sync_activities(days: int = 30) -> tuple[list[dict], str]:
    """
    Download running activities from the last N days, parse via fit.summarize,
    and return (records, status_message).
    """
    from coach.fit import summarize as fit_summarize

    client = get_client()
    if client is None:
        return [], "Not connected — enter your Garmin credentials first."

    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=days)

    try:
        activities = client.get_activities_by_date(
            start_dt.isoformat(), end_dt.isoformat(), "running"
        )
    except Exception as exc:
        return [], f"Failed to fetch activity list: {exc}"

    if not activities:
        return [], f"No running activities found in the last {days} days."

    records, errors = [], []

    for act in activities:
        activity_id = act.get("activityId")
        act_name    = act.get("activityName", str(activity_id))
        try:
            zip_data = client.download_activity(
                activity_id,
                dl_fmt=client.ActivityDownloadFormat.ORIGINAL,
            )
            record = _parse_fit_from_zip(zip_data, fit_summarize)
            if record:
                records.append(record)
        except Exception as exc:
            errors.append(act_name)
            logger.warning("Could not sync '%s' (%s): %s", act_name, activity_id, exc)

    msg = f"Synced {len(records)} run{'s' if len(records) != 1 else ''}"
    if errors:
        msg += f"  ({len(errors)} skipped)"
    return records, msg


def _parse_fit_from_zip(zip_bytes: bytes, fit_summarize) -> Optional[dict]:
    """Extract the first .fit file from a ZIP and return a parsed summary."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            fit_names = [n for n in zf.namelist() if n.lower().endswith(".fit")]
            if not fit_names:
                return None
            fit_bytes = zf.read(fit_names[0])
    except zipfile.BadZipFile:
        # Some Garmin endpoints return raw FIT data (not zipped)
        fit_bytes = zip_bytes

    with tempfile.NamedTemporaryFile(suffix=".fit", delete=False) as tf:
        tf.write(fit_bytes)
        tmp_path = tf.name
    try:
        record = fit_summarize(tmp_path)
        return record if record and "error" not in record else None
    finally:
        os.unlink(tmp_path)


def connection_status() -> str:
    """Human-readable connection status for the UI status badge."""
    if not is_authenticated():
        return "not_connected"
    email = load_email()
    return f"connected:{email}" if email else "connected"
