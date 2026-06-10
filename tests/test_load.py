"""Tests for coach/load.py — Banister TRIMP and fallback paths."""

import pytest
from coach.load import compute, weekly_load
from coach.zones import build_zones

_HR_MAX  = 177
_HR_REST = 50
_ZONES   = build_zones(45.0, 0.0)


def _run(**kw):
    base = dict(duration_s=3600, distance_km=10.0, avg_hr=None,
                avg_pace_s=None, training_load=None)
    base.update(kw)
    return base


# ── Path selection ─────────────────────────────────────────────────────────

def test_hr_path_used_when_hr_present():
    r = _run(avg_hr=148, avg_pace_s=360)
    load = compute(r, _HR_MAX, _HR_REST)
    assert load is not None and load > 0

def test_pace_path_used_when_no_hr():
    r = _run(avg_hr=None, avg_pace_s=360)
    load = compute(r, _HR_MAX, _HR_REST, _ZONES)
    assert load is not None and load > 0

def test_distance_fallback_when_no_hr_no_zones():
    r = _run(avg_hr=None, avg_pace_s=None)
    load = compute(r, _HR_MAX, _HR_REST, zones=None)
    assert load is not None and load > 0

def test_returns_none_on_error_run():
    assert compute({"error": "parse failed"}, _HR_MAX, _HR_REST) is None

def test_returns_none_on_zero_duration():
    assert compute(_run(duration_s=0), _HR_MAX, _HR_REST) is None


# ── TRIMP monotonicity ─────────────────────────────────────────────────────

def test_higher_hr_means_higher_load():
    easy   = compute(_run(avg_hr=130), _HR_MAX, _HR_REST)
    hard   = compute(_run(avg_hr=168), _HR_MAX, _HR_REST)
    assert hard > easy

def test_longer_duration_means_higher_load():
    short = compute(_run(avg_hr=148, duration_s=1800), _HR_MAX, _HR_REST)
    long_ = compute(_run(avg_hr=148, duration_s=3600), _HR_MAX, _HR_REST)
    assert long_ > short

def test_faster_pace_fallback_means_higher_load():
    easy  = compute(_run(avg_pace_s=420), _HR_MAX, _HR_REST, _ZONES)
    fast  = compute(_run(avg_pace_s=300), _HR_MAX, _HR_REST, _ZONES)
    assert fast > easy


# ── HR sanity guards ───────────────────────────────────────────────────────

def test_hr_below_resting_uses_fallback():
    """HR barely above resting should not produce a HR-path result."""
    r = _run(avg_hr=_HR_REST + 5, avg_pace_s=360)
    load_no_hr   = compute(_run(avg_hr=None, avg_pace_s=360), _HR_MAX, _HR_REST, _ZONES)
    load_low_hr  = compute(r, _HR_MAX, _HR_REST, _ZONES)
    # Both should fall through to pace path (or distance), results should be close
    assert abs((load_no_hr or 0) - (load_low_hr or 0)) < 5.0


# ── weekly_load aggregation ────────────────────────────────────────────────

def test_weekly_load_sums_within_window():
    history = [
        {"date": "2026-06-07", "training_load": 80.0},
        {"date": "2026-06-08", "training_load": 60.0},
        {"date": "2026-05-01", "training_load": 999.0},  # outside window
    ]
    total = weekly_load(history, "2026-06-08", days=7)
    assert total == pytest.approx(140.0)

def test_weekly_load_ignores_null():
    history = [
        {"date": "2026-06-07", "training_load": None},
        {"date": "2026-06-08", "training_load": 60.0},
    ]
    assert weekly_load(history, "2026-06-08") == pytest.approx(60.0)
