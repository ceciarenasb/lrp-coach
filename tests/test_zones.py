"""Tests for VDOT, Critical Velocity, and zone builder."""

from coach.zones import (
    build_zones, cv_from_two_efforts, cv_from_vdot,
    fmt_pace, vdot_from_race,
)


# ── VDOT ──────────────────────────────────────────────────────────────────

def test_vdot_half_marathon_1h55():
    # 1:55:00 half → VDOT ~38 per this implementation of Daniels' formula
    vdot = vdot_from_race(21_097, 6900)
    assert 37 < vdot < 40

def test_vdot_marathon_sub3():
    # 2:58:00 marathon → VDOT ~54
    vdot = vdot_from_race(42_195, 10680)
    assert 53 < vdot < 56

def test_vdot_increases_with_faster_time():
    slow = vdot_from_race(10_000, 3600)   # 60:00 10K
    fast = vdot_from_race(10_000, 2700)   # 45:00 10K
    assert fast > slow

def test_vdot_positive():
    vdot = vdot_from_race(5_000, 1500)
    assert vdot > 0


# ── Critical Velocity ──────────────────────────────────────────────────────

def test_cv_two_efforts_basic():
    # 1500m in 375s and 3000m in 720s → CV = (3000-1500)/(720-375) = 4.35 m/s
    cv = cv_from_two_efforts(1500, 375, 3000, 720)
    assert abs(cv - (1500 / 345)) < 0.01

def test_cv_two_efforts_ordering_invariant():
    # Swapping efforts should give same CV
    cv1 = cv_from_two_efforts(1500, 375, 3000, 720)
    cv2 = cv_from_two_efforts(3000, 720, 1500, 375)
    assert abs(cv1 - cv2) < 0.01

def test_cv_from_vdot_plausible():
    # CV should sit between threshold and interval speeds
    vdot = 52.0
    cv = cv_from_vdot(vdot)
    # ~3.5–4.5 m/s is a reasonable range for a VDOT 52 runner
    assert 3.0 < cv < 5.5

def test_cv_from_vdot_same_distance_returns_zero():
    cv = cv_from_two_efforts(1500, 375, 1500, 500)
    assert cv == 0.0


# ── Zone builder ───────────────────────────────────────────────────────────

def test_build_zones_ordering():
    # rep > interval > cv_interval > threshold > marathon > easy_hi > easy_lo
    # In pace (sec/km): lower = faster, so faster zones have lower values
    z = build_zones(52.0, cv_from_vdot(52.0))
    assert z.rep < z.interval < z.cv_interval < z.threshold < z.marathon < z.easy_hi < z.easy_lo

def test_build_zones_vdot_stored():
    z = build_zones(52.0, cv_from_vdot(52.0))
    assert z.vdot == 52.0

def test_build_zones_zero_cv_falls_back():
    # cv_mps=0 should fall back to estimated CV
    z_estimated = build_zones(52.0, 0.0)
    z_explicit  = build_zones(52.0, cv_from_vdot(52.0))
    assert z_estimated.cv_interval == z_explicit.cv_interval


# ── fmt_pace ───────────────────────────────────────────────────────────────

def test_fmt_pace_basic():
    assert fmt_pace(300) == "5:00 /km"

def test_fmt_pace_with_seconds():
    assert fmt_pace(330) == "5:30 /km"

def test_fmt_pace_none():
    assert fmt_pace(None) == "n/a"

def test_fmt_pace_zero():
    assert fmt_pace(0) == "n/a"
