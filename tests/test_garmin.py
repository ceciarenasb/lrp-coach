"""Tests for Garmin Connect auth, credential helpers, and sync logic."""

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

import coach.garmin as garmin


@pytest.fixture(autouse=True)
def patch_paths(tmp_path, monkeypatch):
    """Redirect all file I/O to a temp directory and reset module state."""
    monkeypatch.setattr(garmin, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(garmin, "GARMIN_TOKEN_DIR", tmp_path / "garmin_token")
    monkeypatch.setattr(garmin, "GARMIN_CREDS_FILE", tmp_path / "garmin_creds.json")
    monkeypatch.setattr(garmin, "_pending_mfa_client", None)


# ── Credential helpers ─────────────────────────────────────────────────────

def test_save_and_load_email():
    garmin.save_email("runner@example.com")
    assert garmin.load_email() == "runner@example.com"


def test_load_email_returns_none_when_missing():
    assert garmin.load_email() is None


def test_load_email_returns_none_on_corrupt_file(tmp_path, monkeypatch):
    creds = tmp_path / "garmin_creds.json"
    creds.write_text("not json{{{")
    monkeypatch.setattr(garmin, "GARMIN_CREDS_FILE", creds)
    assert garmin.load_email() is None


# ── is_authenticated ───────────────────────────────────────────────────────

def test_not_authenticated_when_no_token_dir():
    assert not garmin.is_authenticated()


def test_not_authenticated_when_token_dir_is_empty():
    garmin.GARMIN_TOKEN_DIR.mkdir()
    assert not garmin.is_authenticated()


def test_authenticated_when_token_file_exists():
    garmin.GARMIN_TOKEN_DIR.mkdir()
    (garmin.GARMIN_TOKEN_DIR / "oauth2_token.json").write_text("{}")
    assert garmin.is_authenticated()


# ── clear_auth ─────────────────────────────────────────────────────────────

def test_clear_auth_removes_token_dir_and_creds():
    garmin.GARMIN_TOKEN_DIR.mkdir()
    (garmin.GARMIN_TOKEN_DIR / "token.json").write_text("{}")
    garmin.save_email("runner@example.com")
    garmin.clear_auth()
    assert not garmin.GARMIN_TOKEN_DIR.exists()
    assert not garmin.GARMIN_CREDS_FILE.exists()


def test_clear_auth_is_safe_when_nothing_exists():
    garmin.clear_auth()  # must not raise


# ── connection_status ──────────────────────────────────────────────────────

def test_connection_status_not_connected():
    assert garmin.connection_status() == "not_connected"


def test_connection_status_connected_with_email():
    garmin.GARMIN_TOKEN_DIR.mkdir()
    (garmin.GARMIN_TOKEN_DIR / "token.json").write_text("{}")
    garmin.save_email("runner@lrp.run")
    assert garmin.connection_status() == "connected:runner@lrp.run"


def test_connection_status_connected_without_email():
    garmin.GARMIN_TOKEN_DIR.mkdir()
    (garmin.GARMIN_TOKEN_DIR / "token.json").write_text("{}")
    assert garmin.connection_status() == "connected"


# ── connect ────────────────────────────────────────────────────────────────

def test_connect_success():
    with patch("garminconnect.Garmin") as MockGarmin:
        MockGarmin.return_value = MagicMock()
        ok, msg = garmin.connect("runner@lrp.run", "secret")
    assert ok is True
    assert "runner@lrp.run" in msg
    assert garmin.load_email() == "runner@lrp.run"
    assert garmin._pending_mfa_client is None


def test_connect_mfa_required_saves_client_for_submit():
    with patch("garminconnect.Garmin") as MockGarmin:
        client = MagicMock()
        client.login.side_effect = Exception("NEEDS_MFA to continue")
        MockGarmin.return_value = client
        ok, msg = garmin.connect("runner@lrp.run", "secret")
    assert ok is False
    assert msg == "MFA_REQUIRED"
    assert garmin._pending_mfa_client is client  # held for submit_mfa


def test_connect_mfa_case_insensitive_detection():
    with patch("garminconnect.Garmin") as MockGarmin:
        client = MagicMock()
        client.login.side_effect = Exception("multi-factor authentication required")
        MockGarmin.return_value = client
        ok, msg = garmin.connect("runner@lrp.run", "secret")
    assert ok is False
    assert msg == "MFA_REQUIRED"


def test_connect_generic_error_clears_pending_client():
    with patch("garminconnect.Garmin") as MockGarmin:
        client = MagicMock()
        client.login.side_effect = Exception("Invalid credentials")
        MockGarmin.return_value = client
        ok, msg = garmin.connect("runner@lrp.run", "wrong")
    assert ok is False
    assert "Invalid credentials" in msg
    assert garmin._pending_mfa_client is None


# ── submit_mfa ─────────────────────────────────────────────────────────────

def test_submit_mfa_success():
    with patch("garminconnect.Garmin") as MockGarmin:
        MockGarmin.return_value = MagicMock()
        ok, msg = garmin.submit_mfa("runner@lrp.run", "secret", "123456")
    assert ok is True
    assert "runner@lrp.run" in msg
    assert garmin.load_email() == "runner@lrp.run"
    assert garmin._pending_mfa_client is None


def test_submit_mfa_reuses_pending_client():
    """submit_mfa must use the saved client so garth OAuth state is intact."""
    pending = MagicMock()
    garmin._pending_mfa_client = pending
    ok, msg = garmin.submit_mfa("runner@lrp.run", "secret", "123456")
    assert ok is True
    pending.login.assert_called_once_with(mfa_code="123456")


def test_submit_mfa_failure():
    with patch("garminconnect.Garmin") as MockGarmin:
        client = MagicMock()
        client.login.side_effect = Exception("Invalid code")
        MockGarmin.return_value = client
        ok, msg = garmin.submit_mfa("runner@lrp.run", "secret", "999999")
    assert ok is False
    assert "Invalid code" in msg


# ── get_client ─────────────────────────────────────────────────────────────

def test_get_client_returns_none_when_not_authenticated():
    assert garmin.get_client() is None


def test_get_client_loads_tokens_via_garth():
    garmin.GARMIN_TOKEN_DIR.mkdir()
    (garmin.GARMIN_TOKEN_DIR / "oauth2_token.json").write_text("{}")
    with patch("garminconnect.Garmin") as MockGarmin:
        client = MagicMock()
        MockGarmin.return_value = client
        result = garmin.get_client()
    assert result is client
    client.garth.load.assert_called_once_with(str(garmin.GARMIN_TOKEN_DIR))


def test_get_client_returns_none_on_token_load_failure():
    garmin.GARMIN_TOKEN_DIR.mkdir()
    (garmin.GARMIN_TOKEN_DIR / "oauth2_token.json").write_text("{}")
    with patch("garminconnect.Garmin") as MockGarmin:
        client = MagicMock()
        client.garth.load.side_effect = Exception("token expired")
        MockGarmin.return_value = client
        result = garmin.get_client()
    assert result is None


# ── sync_activities ────────────────────────────────────────────────────────

def test_sync_activities_returns_empty_when_not_connected():
    records, msg = garmin.sync_activities(days=7)
    assert records == []
    assert "Not connected" in msg


def test_sync_activities_returns_empty_when_no_runs():
    with patch.object(garmin, "get_client") as mock_get:
        client = MagicMock()
        client.get_activities_by_date.return_value = []
        mock_get.return_value = client
        records, msg = garmin.sync_activities(days=7)
    assert records == []
    assert "No activities found" in msg


def test_sync_activities_returns_parsed_records():
    fake_summary = {"date": "2026-06-01", "distance_km": 10.5, "avg_pace_s": 325}
    with patch.object(garmin, "get_client") as mock_get, \
         patch.object(garmin, "_parse_fit_from_zip", return_value=fake_summary):
        client = MagicMock()
        client.get_activities_by_date.return_value = [
            {"activityId": 42, "activityName": "Morning Run"}
        ]
        client.download_activity.return_value = _make_zip()
        mock_get.return_value = client
        records, msg = garmin.sync_activities(days=7)
    assert len(records) == 1
    assert records[0]["date"] == "2026-06-01"
    assert "Synced 1 activity" in msg


def test_sync_activities_skips_unparseable_activities():
    with patch.object(garmin, "get_client") as mock_get, \
         patch.object(garmin, "_parse_fit_from_zip", return_value=None):
        client = MagicMock()
        client.get_activities_by_date.return_value = [
            {"activityId": 1, "activityName": "Bad Run"},
            {"activityId": 2, "activityName": "Also Bad"},
        ]
        client.download_activity.return_value = _make_zip()
        mock_get.return_value = client
        records, msg = garmin.sync_activities(days=7)
    assert records == []
    assert "Synced 0 activities" in msg


def test_sync_activities_reports_skipped_on_download_error():
    with patch.object(garmin, "get_client") as mock_get:
        client = MagicMock()
        client.get_activities_by_date.return_value = [
            {"activityId": 99, "activityName": "Fail Run"}
        ]
        client.download_activity.side_effect = Exception("network error")
        mock_get.return_value = client
        records, msg = garmin.sync_activities(days=7)
    assert records == []
    assert "skipped" in msg


# ── _parse_fit_from_zip ────────────────────────────────────────────────────

def test_parse_fit_from_zip_extracts_first_fit_file():
    fake_result = {"date": "2026-06-05", "distance_km": 8.0}
    fake_fit_fn = MagicMock(return_value=fake_result)
    result = garmin._parse_fit_from_zip(_make_zip(fit_name="run.fit"), fake_fit_fn)
    assert result == fake_result
    fake_fit_fn.assert_called_once()


def test_parse_fit_from_zip_returns_none_when_no_fit_in_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no fit here")
    result = garmin._parse_fit_from_zip(buf.getvalue(), MagicMock())
    assert result is None


def test_parse_fit_from_zip_returns_none_on_error_in_summary():
    fake_fit_fn = MagicMock(return_value={"error": "bad file"})
    result = garmin._parse_fit_from_zip(_make_zip(), fake_fit_fn)
    assert result is None


def test_parse_fit_from_zip_handles_raw_fit_bytes():
    fake_result = {"date": "2026-06-05", "distance_km": 5.0}
    fake_fit_fn = MagicMock(return_value=fake_result)
    result = garmin._parse_fit_from_zip(b"notazip_rawbytes", fake_fit_fn)
    assert result == fake_result
    fake_fit_fn.assert_called_once()


# ── helpers ────────────────────────────────────────────────────────────────

def _make_zip(fit_name: str = "activity.fit") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(fit_name, b"\x0e\x10\x00\x00fake_fit_bytes")
    return buf.getvalue()
