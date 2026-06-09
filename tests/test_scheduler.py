"""Tests for the APScheduler-based daily sync scheduler."""

import pytest
import coach.scheduler as sched


@pytest.fixture(autouse=True)
def cleanup():
    """Stop the scheduler after every test to prevent thread leaks."""
    yield
    sched.stop()


# ── Initial state ──────────────────────────────────────────────────────────

def test_not_running_initially():
    assert not sched.is_running()


def test_has_no_job_initially():
    assert not sched.has_job()


def test_next_run_str_returns_dash_when_not_running():
    assert sched.next_run_str() == "—"


# ── start / stop ───────────────────────────────────────────────────────────

def test_start_makes_scheduler_running():
    sched.start(lambda: None)
    assert sched.is_running()


def test_start_registers_daily_job():
    sched.start(lambda: None)
    assert sched.has_job()


def test_start_is_idempotent():
    sched.start(lambda: None)
    sched.start(lambda: None)
    assert sched.is_running()


def test_stop_halts_scheduler():
    sched.start(lambda: None)
    sched.stop()
    assert not sched.is_running()


def test_stop_is_safe_when_not_running():
    sched.stop()  # must not raise


# ── next_run_str ───────────────────────────────────────────────────────────

def test_next_run_str_returns_formatted_time_when_running():
    sched.start(lambda: None)
    result = sched.next_run_str()
    assert result != "—"
    # Should look like "DD Mon YYYY HH:MM"
    assert len(result) > 8


def test_next_run_str_returns_dash_after_stop():
    sched.start(lambda: None)
    sched.stop()
    assert sched.next_run_str() == "—"


# ── set_enabled ────────────────────────────────────────────────────────────

def test_set_enabled_true_starts_scheduler():
    msg = sched.set_enabled(True, lambda: None)
    assert sched.is_running()
    assert sched.has_job()
    assert "enabled" in msg.lower()
    assert "next run" in msg.lower()


def test_set_enabled_true_without_fn_returns_error():
    msg = sched.set_enabled(True, None)
    assert "Cannot enable" in msg
    assert not sched.is_running()


def test_set_enabled_false_removes_job():
    sched.start(lambda: None)
    assert sched.has_job()
    sched.set_enabled(False)
    assert not sched.has_job()


def test_set_enabled_false_returns_disabled_message():
    sched.start(lambda: None)
    msg = sched.set_enabled(False)
    assert "disabled" in msg.lower()


def test_set_enabled_true_when_already_running_does_not_restart():
    sched.start(lambda: None)
    scheduler_before = sched._scheduler
    sched.set_enabled(True, lambda: None)
    # Same scheduler instance — not restarted
    assert sched._scheduler is scheduler_before


def test_set_enabled_false_when_not_running_is_safe():
    sched.set_enabled(False)  # must not raise
