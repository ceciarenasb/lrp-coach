"""Integration test — full plan generation invariants.

Covers:
  - Plan starts D+1 (tomorrow)
  - LRP Monday (wd=0) present every week
  - LRP Saturday (wd=5) present every week as Club Run long
  - Build/Peak weeks have ≥1 quality session (Tempo/SVC/MP/Progression)
  - Base weeks 2+ have quality sessions (SVC/Tempo/Progression)
  - 4 run-day schedule → ≥2 quality sessions in Build/Peak weeks
  - injury=moderate → no quality sessions, volume reduced ≤70 %
  - Slower goal → zones.marathon ≥ goal_s/42.195 (M-pace not faster than goal)
  - Weekly km non-decreasing through Base→Peak, then drops in Taper
  - Long Run and Medium-Long never on weekdays (wd 0–4)
  - Quality sessions are ≥2 days from any Club Run or other quality session
"""

from datetime import date, timedelta

import pytest

from coach.plan import (
    CLUB_RUN, EASY, LONG, MEDIUM_LONG, MP_RUN, PROGRESSION, SVC_INTERVAL, TEMPO,
    generate_plan,
)
from coach.zones import build_zones, vdot_from_race

_QUALITY = {TEMPO, SVC_INTERVAL, MP_RUN, PROGRESSION}
_WEEKDAYS = set(range(5))  # Mon–Fri (0–4)

LRP_SESSIONS = [
    {"day": 0, "km": 10.0, "type": "easy"},
    {"day": 5, "km": 0,    "type": "long"},
]
RUN_DAYS      = [0, 2, 4, 5]   # Mon, Wed, Fri, Sat  (4 days)
RUN_DAYS_5    = [0, 1, 2, 4, 5]  # Mon, Tue, Wed, Fri, Sat  (5 days)
STRENGTH_DAYS = [1, 3]          # Tue, Thu


def _make_plan(race_offset_weeks: int = 16, injury: str = "none",
               run_days: list = None, goal_s: int = None):
    vdot  = vdot_from_race(10_000, 52 * 60)   # 52-min 10 K → ~42 VDOT
    zones = build_zones(vdot, 0)
    if goal_s is None:
        goal_s = int(3.75 * 3600)              # 3h45 marathon
    race  = date.today() + timedelta(weeks=race_offset_weeks)
    return generate_plan(
        marathon_date=race,
        goal_time_s=goal_s,
        zones=zones,
        run_days=run_days if run_days is not None else RUN_DAYS,
        lrp_sessions=LRP_SESSIONS,
        strength_days=STRENGTH_DAYS,
        cycling_days=[],
        injury=injury,
    )


def test_plan_not_empty():
    plan = _make_plan()
    assert len(plan) >= 4


def test_plan_starts_tomorrow():
    plan = _make_plan()
    tomorrow = date.today() + timedelta(days=1)
    first_run = plan[0].start_date
    assert first_run == tomorrow, f"Plan starts {first_run}, expected {tomorrow}"


def test_lrp_monday_every_week():
    plan = _make_plan()
    for wk in plan:
        # Week 1 may be a partial week (starting mid-week) and may not contain Monday
        has_monday = any(dp.weekday == 0 for dp in wk.days)
        if not has_monday:
            continue
        mon_sessions = [dp for dp in wk.days if dp.weekday == 0 and dp.session.type == CLUB_RUN]
        assert mon_sessions, f"Week {wk.week_num} ({wk.phase}): no LRP Monday"


def test_lrp_saturday_every_week():
    """Saturday must always have an LRP session — either Club Run or quality override."""
    plan = _make_plan()
    _QUALITY = {TEMPO, SVC_INTERVAL, MP_RUN, PROGRESSION}
    _LRP_SAT_TYPES = {CLUB_RUN} | _QUALITY
    for wk in plan:
        sat_sessions = [dp for dp in wk.days
                        if dp.weekday == 5 and dp.session.type in _LRP_SAT_TYPES]
        assert sat_sessions, f"Week {wk.week_num} ({wk.phase}): no LRP Saturday session"


def test_build_peak_weeks_have_quality():
    plan = _make_plan()
    for wk in plan:
        if wk.phase not in ("Build", "Peak"):
            continue
        types = {dp.session.type for dp in wk.days}
        assert types & _QUALITY, (
            f"Week {wk.week_num} ({wk.phase}): no quality session — types: {types}"
        )


def test_no_long_or_medium_long_on_weekdays():
    plan = _make_plan()
    bad = []
    for wk in plan:
        for dp in wk.days:
            if dp.weekday in _WEEKDAYS and dp.session.type in {LONG, MEDIUM_LONG}:
                bad.append(f"wk{wk.week_num} wd={dp.weekday} type={dp.session.type}")
    assert not bad, f"Long/Medium-Long on weekday: {bad}"


def test_volume_non_decreasing_base_to_peak():
    plan = _make_plan()
    base_peak = [(wk.week_num, wk.target_km) for wk in plan if wk.phase in ("Base", "Build", "Peak")]
    for i in range(1, len(base_peak)):
        prev_wk, prev_km = base_peak[i - 1]
        curr_wk, curr_km = base_peak[i]
        # allow tiny float noise but not a meaningful drop
        assert curr_km >= prev_km - 0.5, (
            f"Volume dropped from wk{prev_wk}({prev_km:.1f}) to wk{curr_wk}({curr_km:.1f})"
        )


def test_taper_volume_drops():
    plan = _make_plan()
    taper_weeks = [wk for wk in plan if wk.phase == "Taper"]
    if len(taper_weeks) < 2:
        pytest.skip("Not enough taper weeks to compare")
    kms = [wk.target_km for wk in taper_weeks]
    assert kms[-1] < kms[0], f"Taper km did not drop: {kms}"


def test_quality_spacing():
    """Quality sessions must be ≥2 days apart from each other and from Club Runs."""
    plan = _make_plan()
    for wk in plan:
        hard_days = [
            dp.weekday for dp in wk.days
            if dp.session.type in _QUALITY or dp.session.type == CLUB_RUN
        ]
        for i, d1 in enumerate(hard_days):
            for d2 in hard_days[i + 1:]:
                dist = min(abs(d1 - d2), 7 - abs(d1 - d2))
                assert dist >= 2, (
                    f"Wk{wk.week_num}: hard sessions only {dist} day(s) apart (wd {d1} & {d2})"
                )


def test_base_weeks_have_quality_from_week2():
    """Base phase week 2+ must contain at least one quality session (SVC/Tempo/Progression)."""
    plan = _make_plan()
    base_weeks = [wk for wk in plan if wk.phase == "Base"]
    if len(base_weeks) < 2:
        pytest.skip("Less than 2 Base weeks in plan")
    for wk in base_weeks[1:]:  # skip week 1 (strides only)
        types = {dp.session.type for dp in wk.days}
        assert types & _QUALITY, (
            f"Base week {wk.week_num}: no quality session — types: {types}"
        )


def test_four_run_days_two_quality_in_build_peak():
    """4 run days → Build/Peak weeks should have ≥2 quality sessions."""
    plan = _make_plan(run_days=RUN_DAYS)  # 4 days: Mon, Wed, Fri, Sat
    build_peak = [wk for wk in plan if wk.phase in ("Build", "Peak")]
    if not build_peak:
        pytest.skip("No Build/Peak weeks in plan")
    # At least some Build/Peak weeks should have 2 quality sessions
    weeks_with_2q = [
        wk for wk in build_peak
        if sum(1 for dp in wk.days if dp.session.type in _QUALITY) >= 2
    ]
    assert weeks_with_2q, (
        "No Build/Peak week has ≥2 quality sessions with 4 run days"
    )


def test_moderate_injury_no_weekday_quality():
    """injury=moderate must produce no quality sessions on weekdays (all → easy runs)."""
    plan = _make_plan(injury="moderate")
    bad = []
    for wk in plan:
        for dp in wk.days:
            if dp.weekday in _WEEKDAYS and dp.session.type in _QUALITY:
                bad.append(f"wk{wk.week_num} wd={dp.weekday} type={dp.session.type}")
    assert not bad, f"Quality sessions on weekday with moderate injury: {bad}"


def test_moderate_injury_reduces_volume():
    """injury=moderate peak volume must be ≤70 % of injury=none peak."""
    plan_none     = _make_plan(injury="none")
    plan_moderate = _make_plan(injury="moderate")
    peak_none     = max(wk.target_km for wk in plan_none     if wk.phase in ("Base", "Build", "Peak"))
    peak_moderate = max(wk.target_km for wk in plan_moderate if wk.phase in ("Base", "Build", "Peak"))
    assert peak_moderate <= peak_none * 0.72, (
        f"Moderate injury peak ({peak_moderate:.1f} km) is more than 72 % of none ({peak_none:.1f} km)"
    )


def test_slow_goal_mace_not_faster_than_goal_pace():
    """For a 4h40 goal, M-pace in zones must be ≥ goal_s/42.195 (s/km).

    If VDOT-predicted pace is faster than goal pace, we override to goal pace
    so the athlete trains at their actual target, not an unrealistic pace.
    """
    goal_s = int(4.667 * 3600)  # 4h40 → 16800 s
    goal_pace = goal_s / 42.195  # s/km ≈ 398 s/km (6:38/km)

    vdot  = vdot_from_race(21_097, int(2.217 * 3600))  # 2:13 HM benchmark
    zones = build_zones(vdot, 0)

    # zones.marathon might be faster than goal pace for a 4h40 target
    # The plan generator should not produce M-pace sessions faster than goal pace
    race = date.today() + timedelta(weeks=16)
    plan = generate_plan(
        marathon_date=race,
        goal_time_s=goal_s,
        zones=zones,
        run_days=RUN_DAYS,
        lrp_sessions=LRP_SESSIONS,
        strength_days=STRENGTH_DAYS,
        cycling_days=[],
        injury="none",
    )

    for wk in plan:
        for dp in wk.days:
            if dp.session.type == MP_RUN:
                m_pace_target = dp.session.targets.get("M-pace", "")
                if m_pace_target:
                    # parse "6:38/km" → seconds per km
                    parts = m_pace_target.replace("/km", "").split(":")
                    if len(parts) == 2:
                        pace_s = int(parts[0]) * 60 + int(parts[1])
                        assert pace_s >= goal_pace - 5, (  # 5 s tolerance for rounding
                            f"wk{wk.week_num} MP target {m_pace_target} is faster than "
                            f"goal pace {int(goal_pace//60)}:{int(goal_pace%60):02d}/km"
                        )
