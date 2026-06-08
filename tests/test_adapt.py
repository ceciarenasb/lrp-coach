"""Tests for weekly scoring and adaptation decisions."""

import pytest
from coach.adapt import WeekMetrics, score_week


def _metrics(**overrides):
    defaults = dict(
        actual_km=60, target_km=60,
        quality_avg_pace_s=None, quality_target_pace_s=None,
        easy_avg_hr=None, prev_easy_hr=None,
        avg_hr_drift=None,
        avg_feeling=3.0,
    )
    defaults.update(overrides)
    return WeekMetrics(**defaults)


# ── Score bounds ───────────────────────────────────────────────────────────

def test_score_always_0_to_100():
    # Worst possible week
    result = score_week(_metrics(
        actual_km=20, target_km=60,
        quality_avg_pace_s=360, quality_target_pace_s=300,
        easy_avg_hr=165, prev_easy_hr=150,
        avg_hr_drift=12.0,
        avg_feeling=1.0,
    ))
    assert 0 <= result.score <= 100

def test_score_always_0_to_100_perfect():
    # Best possible week
    result = score_week(_metrics(
        actual_km=60, target_km=60,
        quality_avg_pace_s=295, quality_target_pace_s=300,
        easy_avg_hr=140, prev_easy_hr=145,
        avg_hr_drift=2.0,
        avg_feeling=5.0,
    ))
    assert 0 <= result.score <= 100


# ── Decision thresholds ────────────────────────────────────────────────────

def test_great_week_progresses():
    result = score_week(_metrics(
        actual_km=60, target_km=60,
        quality_avg_pace_s=295, quality_target_pace_s=300,
        easy_avg_hr=140, prev_easy_hr=145,
        avg_hr_drift=2.0,
        avg_feeling=4.5,
    ))
    assert result.decision == "PROGRESS"
    assert result.volume_adj == pytest.approx(1.06)
    assert not result.drop_quality
    assert not result.add_recovery

def test_terrible_week_recovers():
    result = score_week(_metrics(
        actual_km=20, target_km=60,
        easy_avg_hr=170, prev_easy_hr=150,
        avg_hr_drift=10.0,
        avg_feeling=1.0,
    ))
    assert result.decision == "RECOVER"
    assert result.volume_adj == pytest.approx(0.85)
    assert result.drop_quality
    assert result.add_recovery

def test_average_week_maintains():
    result = score_week(_metrics(avg_feeling=3.0, actual_km=58, target_km=60))
    assert result.decision == "MAINTAIN"
    assert result.volume_adj == pytest.approx(1.00)


# ── Individual signal contributions ───────────────────────────────────────

def test_high_hr_drift_penalises():
    base  = score_week(_metrics(avg_hr_drift=2.0)).score
    drift = score_week(_metrics(avg_hr_drift=10.0)).score
    assert drift < base

def test_low_volume_penalises():
    full = score_week(_metrics(actual_km=60, target_km=60)).score
    low  = score_week(_metrics(actual_km=30, target_km=60)).score
    assert low < full

def test_good_quality_pace_rewards():
    without = score_week(_metrics()).score
    with_q  = score_week(_metrics(
        quality_avg_pace_s=298, quality_target_pace_s=300
    )).score
    assert with_q > without

def test_slow_quality_pace_penalises():
    fast = score_week(_metrics(quality_avg_pace_s=298, quality_target_pace_s=300)).score
    slow = score_week(_metrics(quality_avg_pace_s=340, quality_target_pace_s=300)).score
    assert slow < fast

def test_improving_hr_rewards():
    flat     = score_week(_metrics(easy_avg_hr=150, prev_easy_hr=150)).score
    dropping = score_week(_metrics(easy_avg_hr=145, prev_easy_hr=150)).score
    assert dropping > flat

def test_feeling_5_better_than_1():
    good = score_week(_metrics(avg_feeling=5.0)).score
    bad  = score_week(_metrics(avg_feeling=1.0)).score
    assert good > bad


# ── Result fields always present ───────────────────────────────────────────

def test_result_has_all_fields():
    r = score_week(_metrics())
    assert isinstance(r.score, int)
    assert r.decision in {"PROGRESS", "MAINTAIN", "CONSOLIDATE", "RECOVER"}
    assert isinstance(r.flags, list)
    assert isinstance(r.volume_adj, float)
    assert isinstance(r.drop_quality, bool)
    assert isinstance(r.add_recovery, bool)
    assert isinstance(r.reasoning, str)
