"""
Background scheduler — daily Garmin sync at 07:00 local time.
Uses APScheduler 3.x BackgroundScheduler (runs inside the Gradio process).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_scheduler = None  # apscheduler.schedulers.background.BackgroundScheduler
_JOB_ID = "garmin_daily"


def start(sync_fn: Callable) -> None:
    """Start the scheduler and register the daily sync job."""
    global _scheduler
    from apscheduler.schedulers.background import BackgroundScheduler

    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        sync_fn,
        trigger="cron",
        hour=7,
        minute=0,
        id=_JOB_ID,
        replace_existing=True,
        misfire_grace_time=3600,  # allow up to 1h late if machine was asleep
    )
    _scheduler.start()
    logger.info("Garmin daily sync scheduler started (fires at 07:00)")


def stop() -> None:
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        _scheduler = None


def set_enabled(enabled: bool, sync_fn: Optional[Callable] = None) -> str:
    """Enable or disable the daily job at runtime."""
    global _scheduler
    if enabled:
        if sync_fn is None:
            return "Cannot enable: no sync function provided."
        if not (_scheduler and _scheduler.running):
            start(sync_fn)
        return f"Auto-sync enabled — next run {next_run_str()}"
    else:
        if _scheduler and _scheduler.running:
            try:
                _scheduler.remove_job(_JOB_ID)
            except Exception:
                pass
        return "Auto-sync disabled."


def is_running() -> bool:
    return bool(_scheduler and _scheduler.running)


def has_job() -> bool:
    if not is_running():
        return False
    return _scheduler.get_job(_JOB_ID) is not None


def next_run_str() -> str:
    if not has_job():
        return "—"
    job = _scheduler.get_job(_JOB_ID)
    nxt: Optional[datetime] = job.next_run_time if job else None
    return nxt.strftime("%d %b %Y %H:%M") if nxt else "—"
