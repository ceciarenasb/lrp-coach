"""Tests for coach/readiness.py — readiness multiplier."""

import pytest
from coach.readiness import score, label


# ── Range ──────────────────────────────────────────────────────────────────

def test_output_always_in_range():
    # Worst case
    assert 0.60 <= score(1.0, 170.0, 150.0, 12.0) <= 1.10
    # Best case
    assert 0.60 <= score(5.0, 140.0, 150.0, 1.0)  <= 1.10


def test_neutral_inputs_return_base():
    """Feeling=3, no HR data → result equals base 0.85."""
    assert score(3.0, None, None, None) == pytest.approx(0.85)


# ── Feeling ────────────────────────────────────────────────────────────────

def test_feeling_5_higher_than_feeling_1():
    good = score(5.0, None, None, None)
    bad  = score(1.0, None, None, None)
    assert good > bad

def test_feeling_contributes_correct_delta():
    base  = score(3.0, None, None, None)
    good  = score(5.0, None, None, None)
    assert good == pytest.approx(base + 0.10, abs=0.001)


# ── HR trend ───────────────────────────────────────────────────────────────

def test_rising_hr_reduces_readiness():
    flat    = score(3.0, 150.0, 150.0, None)
    rising  = score(3.0, 162.0, 150.0, None)   # > 1.06 ratio
    assert rising < flat

def test_dropping_hr_increases_readiness():
    flat     = score(3.0, 150.0, 150.0, None)
    dropping = score(3.0, 144.0, 150.0, None)  # < 0.97 ratio
    assert dropping > flat

def test_no_prev_hr_no_trend_contribution():
    with_prev    = score(3.0, 150.0, 150.0, None)
    without_prev = score(3.0, 150.0, None,  None)
    assert with_prev == without_prev   # ratio can't be computed → same as base


# ── HR drift ───────────────────────────────────────────────────────────────

def test_high_drift_reduces_readiness():
    low  = score(3.0, None, None, 2.0)
    high = score(3.0, None, None, 9.0)
    assert high < low

def test_extreme_stress_stack_hits_floor():
    s = score(1.0, 170.0, 150.0, 12.0)
    assert s == pytest.approx(0.60, abs=0.01)


# ── Labels ─────────────────────────────────────────────────────────────────

def test_label_excellent():
    assert label(1.06) == "Excellent"

def test_label_recovery_needed():
    assert label(0.72) == "Recovery needed"
