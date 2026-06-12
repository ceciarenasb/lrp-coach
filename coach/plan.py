"""
Marathon training plan generator — Daniels + SVC hybrid periodization.

Phases : Base → Build → Peak → Taper
Hard/easy principle: quality/long days always separated by ≥1 easy/rest day.
Cross-training combines with run days (noted in description) rather than blocking them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .zones import Zones, fmt_pace

REST         = "Rest"
RECOVERY     = "Recovery"
EASY         = "Easy"
MEDIUM_LONG  = "Medium-Long"
LONG         = "Long Run"
TEMPO        = "Tempo"
SVC_INTERVAL = "SVC Intervals"
MP_RUN       = "Marathon Pace"
PROGRESSION  = "Progression Run"
STRENGTH     = "Strength"
CYCLING      = "Cycling / Zwift"
CLUB_RUN     = "Club Run (LRP)"

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

_HARD = {TEMPO, SVC_INTERVAL, MP_RUN, LONG, MEDIUM_LONG, PROGRESSION}


@dataclass
class Session:
    type: str
    description: str
    distance_km: float = 0.0
    duration_min: int = 0
    targets: dict = field(default_factory=dict)


@dataclass
class DayPlan:
    date: date
    weekday: int
    session: Session


@dataclass
class WeekPlan:
    week_num: int
    start_date: date
    phase: str
    focus: str
    target_km: float
    days: list


# ---------------------------------------------------------------------------
# Volume targets
# ---------------------------------------------------------------------------

def _peak_km(goal_s: int) -> float:
    h = goal_s / 3600
    if h <= 3.0:  return 90.0
    if h <= 3.5:  return 78.0
    if h <= 4.0:  return 65.0
    if h <= 4.5:  return 62.0
    if h <= 5.0:  return 57.0
    return 50.0


def _long_run_km(wk: int, total: int) -> float:
    taper_start = total - 1  # 2-week taper: one taper week + race week
    if wk >= taper_start:
        return (26.0, 16.0)[min(wk - taper_start, 1)]
    prog = (wk - 1) / max(1, taper_start - 1)
    return round(18.0 + prog * 14.0, 1)


def _medium_long_km(long_km: float) -> float:
    return round(max(14.0, min(22.0, long_km * 0.65)), 1)


# ---------------------------------------------------------------------------
# Session builders
# ---------------------------------------------------------------------------

def _recovery(z: Zones, note: str = "") -> Session:
    extra = f" + {note}" if note else ""
    return Session(
        RECOVERY,
        f"Recovery 6 km — very easy, HR in Z1, legs loose{extra}",
        distance_km=6.0,
        targets={"pace": f"easier than {fmt_pace(z.easy_lo)}"},
    )


def _easy(km: float, z: Zones, note: str = "") -> Session:
    extra = f" + {note}" if note else ""
    return Session(
        EASY,
        f"Easy {km:.0f} km — conversational pace, nasal breathing{extra}",
        distance_km=round(km, 1),
        targets={"pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}"},
    )


def _easy_strides(km: float, z: Zones, note: str = "") -> Session:
    extra = f" + {note}" if note else ""
    return Session(
        EASY,
        f"Easy {km:.0f} km + 6×100 m strides — float fast, full recovery between each{extra}",
        distance_km=round(km, 1),
        targets={"easy_pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}", "strides": "near 5 K effort"},
    )


def _long(km: float, phase: str, z: Zones) -> Session:
    if phase == "Peak" and km >= 26:
        return Session(
            LONG,
            f"Long run {km:.0f} km — easy first 16 km, km 17–{int(km)-4} at M-pace, last 4 km easy",
            distance_km=km,
            targets={
                "easy_section": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}",
                "M-pace_section": fmt_pace(z.marathon),
            },
        )
    if phase in ("Build", "Peak") and km >= 24:
        return Session(
            LONG,
            f"Long run {km:.0f} km — easy aerobic, final 8–10 km at M-pace",
            distance_km=km,
            targets={
                "main_pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}",
                "finish_M-pace": fmt_pace(z.marathon),
            },
        )
    return Session(
        LONG,
        f"Long run {km:.0f} km — easy aerobic throughout, stay in Z1–Z2",
        distance_km=km,
        targets={"pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}"},
    )


def _medium_long(km: float, phase: str, z: Zones) -> Session:
    if phase in ("Build", "Peak") and km >= 18:
        return Session(
            MEDIUM_LONG,
            f"Medium-long {km:.0f} km — first 12 km easy, final {int(km)-12} km at M-pace",
            distance_km=km,
            targets={
                "easy_section": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}",
                "M-pace_section": fmt_pace(z.marathon),
            },
        )
    return Session(
        MEDIUM_LONG,
        f"Medium-long {km:.0f} km — steady aerobic, slightly brisker than easy",
        distance_km=km,
        targets={"pace": f"{fmt_pace(z.easy_hi)} – {fmt_pace(z.marathon)}"},
    )


def _quality(phase: str, wk: int, z: Zones, injury: str, q_num: int = 1, phase_wk: int = 1) -> Session:
    """Quality session — cycle driven by phase_wk so Build always starts with VO₂max."""
    if injury == "moderate":
        return _easy(10, z)

    cycle = (phase_wk - 1) % 4

    if phase == "Base":
        # From week 2: alternate short intervals and threshold work
        base_cycle = (phase_wk - 2) % 4
        base_variants = [
            Session(SVC_INTERVAL,
                "Short intervals — 2 km WU + 6×400 m at I-pace (90 s walk recovery) + 3 km CD",
                distance_km=10.0,
                targets={"I-pace": fmt_pace(z.interval), "recovery": "90 s walk"}),
            Session(TEMPO,
                "Cruise intervals — 2 km WU + 5×1000 m at T-pace (60 s jog) + 2 km CD",
                distance_km=11.0,
                targets={"T-pace": fmt_pace(z.threshold), "recovery": "60 s jog"}),
            Session(SVC_INTERVAL,
                "Short intervals — 2 km WU + 5×600 m at I-pace (90 s jog) + 3 km CD",
                distance_km=11.0,
                targets={"I-pace": fmt_pace(z.interval), "recovery": "90 s jog"}),
            Session(PROGRESSION,
                "Progression — 5 km easy + 5 km at M-pace + 3 km at T-pace + 1 km CD",
                distance_km=14.0,
                targets={"M-pace": fmt_pace(z.marathon), "T-pace": fmt_pace(z.threshold)}),
        ]
        return base_variants[base_cycle]

    if phase == "Build":
        if q_num == 2:
            return Session(MP_RUN,
                "M-pace block — 2 km WU + 10 km at M-pace + 2 km CD",
                distance_km=14.0,
                targets={"M-pace": fmt_pace(z.marathon), "WU/CD": fmt_pace(z.easy_hi)})
        # cycle 0 = first week of Build → VO₂max intervals from day 1
        build_variants = [
            Session(SVC_INTERVAL,
                "VO₂max intervals — 3 km WU + 6×800 m at I-pace (90 s jog) + 2 km CD",
                distance_km=12.0,
                targets={"I-pace": fmt_pace(z.interval), "recovery": "90 s jog"}),
            Session(TEMPO,
                "Lactate threshold — 3 km WU + 2×15 min at T-pace (2 min jog) + 2 km CD",
                distance_km=12.0,
                targets={"T-pace": fmt_pace(z.threshold)}),
            Session(SVC_INTERVAL,
                "VO₂max intervals — 3 km WU + 5×1200 m at I-pace (2 min jog) + 2 km CD",
                distance_km=13.0,
                targets={"I-pace": fmt_pace(z.interval), "recovery": "2 min jog"}),
            Session(MP_RUN,
                "M-pace run — 3 km WU + 10 km at M-pace + 2 km CD",
                distance_km=15.0,
                targets={"M-pace": fmt_pace(z.marathon), "WU/CD": fmt_pace(z.easy_hi)}),
        ]
        return build_variants[cycle]

    if phase == "Peak":
        if q_num == 2:
            return Session(PROGRESSION,
                "Progression run — 4 km easy + 6 km at M-pace + 4 km at T-pace + 2 km CD",
                distance_km=16.0,
                targets={
                    "easy": fmt_pace(z.easy_hi),
                    "M-pace": fmt_pace(z.marathon),
                    "T-pace": fmt_pace(z.threshold),
                })
        peak_variants = [
            Session(SVC_INTERVAL,
                "VO₂max race-prep — 3 km WU + 4×1600 m at I-pace (2 min jog) + 2 km CD",
                distance_km=15.0,
                targets={"I-pace": fmt_pace(z.interval), "recovery": "2 min jog"}),
            Session(MP_RUN,
                "Race simulation — 3 km WU + 14 km at M-pace + 3 km easy",
                distance_km=20.0,
                targets={"M-pace": fmt_pace(z.marathon)}),
            Session(SVC_INTERVAL,
                "Sharpener — 3 km WU + 5×1000 m at I-pace (90 s jog) + 2 km CD",
                distance_km=12.0,
                targets={"I-pace": fmt_pace(z.interval), "recovery": "90 s jog"}),
            Session(TEMPO,
                "Threshold top-up — 3 km WU + 2×15 min at T-pace (2 min jog) + 2 km CD",
                distance_km=12.0,
                targets={"T-pace": fmt_pace(z.threshold)}),
        ]
        return peak_variants[cycle]

    # Taper — short and sharp
    if cycle % 2 == 0:
        return Session(MP_RUN,
            "Taper sharpener — 2 km WU + 20 min at M-pace + 2 km CD",
            distance_km=9.0,
            targets={"M-pace": fmt_pace(z.marathon)})
    return Session(TEMPO,
        "Taper tune-up — 2 km WU + 4×1 km at T-pace (90 s jog) + 2 km CD",
        distance_km=9.0,
        targets={"T-pace": fmt_pace(z.threshold)})


def _lrp_sat_quality(phase: str, phase_wk: int, z: Zones) -> Session:
    """Saturday quality override — distinct workout for each override within a phase."""
    if phase == "Build":
        if phase_wk % 6 == 3:  # first override in Build (week 3 of Build)
            return Session(MP_RUN,
                "LRP Sat quality — 3 km WU + 12 km at M-pace + 3 km CD",
                distance_km=18.0,
                targets={"M-pace": fmt_pace(z.marathon), "WU/CD": fmt_pace(z.easy_hi)})
        # second override (week 6 of Build) — progression, harder
        return Session(PROGRESSION,
            "LRP Sat quality — 3 km easy + 8 km at M-pace + 5 km at T-pace + 2 km CD",
            distance_km=18.0,
            targets={"M-pace": fmt_pace(z.marathon), "T-pace": fmt_pace(z.threshold)})
    # Peak — VO₂max race-prep (phase_wk == 2)
    return Session(SVC_INTERVAL,
        "LRP Sat quality — 3 km WU + 8×1000 m at I-pace (90 s jog) + 2 km CD",
        distance_km=15.0,
        targets={"I-pace": fmt_pace(z.interval), "recovery": "90 s jog"})


# ---------------------------------------------------------------------------
# Week builder
# ---------------------------------------------------------------------------

def _build_week(
    week_start: date, wk: int, total: int, phase: str, zones: Zones,
    target_km: float, run_days: list, lrp_sessions: list,
    strength_days: list, cycling_days: list, injury: str,
    phase_wk: int = 1,
) -> list:
    sessions: dict[int, Session] = {}

    # Calendar-relative offset (0 = week_start, ..., 6 = 6 days later)
    # Use offsets for distance checks so Sunday → Monday is 1, not 6.
    start_wd = week_start.weekday()
    def _off(wd: int) -> int:
        return (wd - start_wd) % 7
    def _day_dist(a: int, b: int) -> int:
        d = abs(_off(a) - _off(b))
        return min(d, 7 - d)
    # Sort run days by calendar position this week
    run_days_cal = sorted(run_days, key=_off)

    # Cross-training: if the day already has a run, note it; otherwise own slot
    _xtra: dict[int, list[str]] = {}
    for d in strength_days:
        if d in run_days:
            _xtra.setdefault(d, []).append("Strength 45 min PM")
        elif d not in sessions:
            sessions[d] = Session(STRENGTH, "Strength — squats, deadlifts, lunges, core, hip mobility", 0, 45)
    for d in cycling_days:
        if d in run_days:
            _xtra.setdefault(d, []).append("Cycling / Zwift 60 min PM")
        elif d not in sessions:
            sessions[d] = Session(CYCLING, "Cycling / Zwift — Z1–Z2 aerobic cross-training", 0, 60)

    def _note(d: int) -> str:
        return " + ".join(_xtra.get(d, []))

    # 1. Lock LRP club sessions
    lrp_long_day: Optional[int] = None
    for s in lrp_sessions:
        day = s.get("day")
        if day is None:
            continue
        stype = s.get("type", "easy")
        note  = _note(day)
        extra = f" + {note}" if note else ""

        if stype == "long":
            lrp_long_day = day
            # Quality override at phase weeks 3 & 6 (Build) or week 2 (Peak) — not every week
            _sat_override = (
                phase == "Build" and phase_wk % 3 == 0
            ) or (
                phase == "Peak" and phase_wk == 2
            )
            if _sat_override:
                sessions[day] = _lrp_sat_quality(phase, phase_wk, zones)
            else:
                # Use caller-supplied km if non-zero (decide.py passes user's override);
                # fall back to periodised suggestion when km is 0/None (plan generation).
                km_passed = float(s.get("km") or 0)
                km = km_passed if km_passed > 0 else _long_run_km(wk, total)
                sessions[day] = Session(
                    CLUB_RUN,
                    f"LRP Sat long run — {km:.0f} km suggested (easy aerobic){extra}",
                    distance_km=km,
                    targets={
                        "easy_pace":    f"{fmt_pace(zones.easy_lo)} – {fmt_pace(zones.easy_hi)}",
                        "adjust":       "adapt to how you feel on the day",
                    },
                )
        else:
            km = float(s.get("km", 12.0))
            pace_target = fmt_pace(zones.threshold) if stype == "tempo" else fmt_pace(zones.easy_hi)
            sessions[day] = Session(
                CLUB_RUN,
                f"LRP club run — {km:.0f} km ({stype}){extra}",
                distance_km=km,
                targets={"pace": pace_target},
            )

    # 2. Long run — skipped when LRP Saturday covers it; otherwise weekends only
    if lrp_long_day is not None:
        long_day = lrp_long_day
    else:
        # Never put a standalone long run on a weekday
        weekend = [d for d in run_days if d in (5, 6)]
        long_day = 6 if 6 in run_days else (5 if 5 in run_days else max(run_days, key=_off))
        if long_day not in sessions:
            sessions[long_day] = _long(_long_run_km(wk, total), phase, zones)

    # 3. Day before long run → easy/recovery if it's a run day
    long_off = _off(long_day)
    for rd in run_days:
        if _off(rd) == long_off - 1 and rd not in sessions:
            sessions[rd] = _recovery(zones, _note(rd)) \
                if phase in ("Build", "Peak") else _easy(8.0, zones, _note(rd))
            break

    def _hard_days_so_far():
        # Include Club Run days so quality sessions are spaced away from LRP runs too
        return {d for d, s in sessions.items() if s.type in _HARD or s.type == CLUB_RUN}

    def _place_first_free(candidate_fn, min_dist: int = 2, strict: bool = False):
        """Place candidate on first available run day ≥ min_dist from any hard/club day.
        strict=True: never fall back below min_dist (session is skipped if no good slot)."""
        hard_placed = _hard_days_so_far()
        for d in run_days_cal:
            if d in sessions:
                continue
            if not hard_placed or all(_day_dist(d, h) >= min_dist for h in hard_placed):
                sessions[d] = candidate_fn(d)
                return True
        if min_dist > 1 and not strict:
            return _place_first_free(candidate_fn, min_dist - 1, strict=False)
        return False

    # 4. Primary quality session — skip Base week 1 to let the body adapt first
    _pw = phase_wk
    if not (phase == "Base" and phase_wk == 1):
        _place_first_free(lambda _, pw=_pw: _quality(phase, wk, zones, injury, 1, pw))

    # 5. Second quality in Build/Peak with ≥4 run days — must fit; skip if no clean slot
    if phase in ("Build", "Peak") and len(run_days) >= 4:
        _place_first_free(lambda _, pw=_pw: _quality(phase, wk, zones, injury, 2, pw), strict=True)

    # 6. Medium-long run in Build/Peak — strict: skip rather than land on adjacent day
    if phase in ("Build", "Peak") and len(run_days) >= 4:
        ml_km = _medium_long_km(_long_run_km(wk, total))
        _place_first_free(lambda _: _medium_long(ml_km, phase, zones), strict=True)

    # 6. Fill remaining run days
    hard_placed = _hard_days_so_far()
    day_after_hard = {rd for rd in run_days if any(_off(rd) == _off(h) + 1 for h in hard_placed)}
    used_km = sum(s.distance_km for s in sessions.values())
    remaining = [d for d in run_days_cal if d not in sessions]

    for d in remaining:
        note = _note(d)
        share = max(6.0, min(13.0, (target_km - used_km) / max(1, len(remaining))))
        if d in day_after_hard or injury != "none":
            s = _recovery(zones, note)
        elif phase in ("Build", "Peak") and wk % 2 == 0:
            s = _easy_strides(round(share, 1), zones, note)
        else:
            s = _easy(round(share, 1), zones, note)
        sessions[d] = s
        used_km += s.distance_km

    # Calendar-align each weekday to its actual date within the 7-day block
    def _cal_date(wd: int) -> date:
        return week_start + timedelta(days=(wd - week_start.weekday()) % 7)

    return sorted(
        [
            DayPlan(
                date=_cal_date(wd),
                weekday=wd,
                session=sessions.get(wd, Session(REST, "Rest or active recovery — walk, stretch, foam roll")),
            )
            for wd in range(7)
        ],
        key=lambda dp: dp.date,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_plan(
    marathon_date: date,
    goal_time_s: int,
    zones: Zones,
    run_days: list,
    lrp_sessions: list,
    strength_days: list,
    cycling_days: list,
    injury: str = "none",
    start_km: Optional[float] = None,
    runs_per_week: int = 0,
    max_runs_per_week: int = 0,
    allow_volume_increase: bool = True,
    lrp_day: Optional[int] = None,
    lrp_km: float = 12.0,
    lrp_type: str = "easy",
) -> list:
    if not run_days:
        return []

    if not lrp_sessions and lrp_day is not None:
        lrp_sessions = [{"day": lrp_day, "km": lrp_km, "type": lrp_type}]

    # M-pace = goal pace, not VDOT-predicted pace — athletes train at their target
    goal_mp_s = int(goal_time_s / 42.195)
    if goal_mp_s > zones.marathon:
        from dataclasses import replace as _dc_replace
        zones = _dc_replace(zones, marathon=goal_mp_s)

    tomorrow = date.today() + timedelta(days=1)
    weeks = max(4, min(20, math.ceil((marathon_date - tomorrow).days / 7)))
    peak = _peak_km(goal_time_s)
    base = start_km if start_km else peak * 0.55

    if injury == "light":
        peak *= 0.85
        base *= 0.85
    elif injury == "moderate":
        peak *= 0.70
        base *= 0.70

    for s in lrp_sessions:
        day = s.get("day")
        if day is not None and day not in run_days:
            run_days = sorted(run_days + [day])

    # Progressive run count: Base=min_runs, Build/Peak ramps to max_runs
    lrp_days = sorted({s["day"] for s in lrp_sessions if s.get("day") is not None})
    min_r = max(len(lrp_days), runs_per_week) if runs_per_week else len(run_days)
    max_r = max_runs_per_week if max_runs_per_week else len(run_days)
    min_r = min(min_r, len(run_days))
    max_r = max(min_r, min(max_r, len(run_days)))
    non_lrp_days = [d for d in run_days if d not in lrp_days]

    # Calendar-align weeks to Mon-Sun: week 1 starts from the Monday
    # of the week containing tomorrow (may be a partial week).
    week1_mon = tomorrow - timedelta(days=tomorrow.weekday())
    taper_start = weeks - 1  # last 2 weeks: taper week + race week
    plan = []

    prev_phase = None
    phase_wk_counters: dict = {}

    for wk in range(1, weeks + 1):
        pct = wk / weeks
        if wk > weeks - 2:
            phase = "Taper"
        elif pct < 0.35:
            phase = "Base"
        elif pct < 0.70:
            phase = "Build"
        else:
            phase = "Peak"

        if phase != prev_phase:
            prev_phase = phase
        phase_wk_counters[phase] = phase_wk_counters.get(phase, 0) + 1
        phase_wk = phase_wk_counters[phase]

        if wk >= taper_start:
            target_km = peak * (0.70, 0.35)[min(wk - taper_start, 1)]
        elif not allow_volume_increase:
            target_km = base
        else:
            target_km = base + (peak - base) * ((wk - 1) / max(1, taper_start - 1))

        # Progressive run count per week
        if max_r > min_r:
            if phase == "Base":
                target_r = min_r
            elif phase in ("Build", "Peak"):
                pct_bp = max(0.0, (pct - 0.35) / 0.50)
                target_r = min_r + round((max_r - min_r) * min(1.0, pct_bp))
            else:
                target_r = max(min_r, max_r - 1)
        else:
            target_r = min_r

        extra_slots = max(0, target_r - len(lrp_days))
        if extra_slots >= len(non_lrp_days):
            active_run_days = list(run_days)
        else:
            step = len(non_lrp_days) / max(1, extra_slots)
            extra = [non_lrp_days[int(i * step)] for i in range(extra_slots)]
            active_run_days = sorted(set(lrp_days + extra))

        week_start = week1_mon + timedelta(weeks=wk - 1)
        all_days = _build_week(
            week_start, wk, weeks, phase, zones,
            target_km, active_run_days, lrp_sessions,
            list(strength_days), list(cycling_days), injury,
            phase_wk=phase_wk,
        )

        # Week 1 may be partial: drop days that are already in the past
        if wk == 1:
            all_days = [dp for dp in all_days if dp.date >= tomorrow]

        plan.append(WeekPlan(
            week_num=wk,
            start_date=week_start if wk > 1 else tomorrow,
            phase=phase,
            focus=_focus(phase, wk, weeks),
            target_km=round(target_km, 1),
            days=all_days,
        ))

    return plan


def _focus(phase: str, wk: int, total: int) -> str:
    taper_wk = wk - (total - 2)
    if phase == "Base":
        return "Build aerobic base — short intervals and tempo layered onto easy volume"
    if phase == "Build":
        return "Race fitness — SVC intervals, VO₂max work, medium-long runs with M-pace finish"
    if phase == "Peak":
        return "Peak load — race-specific workouts, long runs with M-pace blocks, absorb the fatigue"
    if taper_wk >= 2:
        return "Race week — final sharpener, easy jogs, trust your training"
    return f"Taper week {taper_wk}/2 — cut volume, keep sharpeners, arrive fresh and confident"


# Public alias — decide.py uses this to build a single week without calling generate_plan
build_week = _build_week


def plan_to_rows(plan: list) -> list:
    rows = []
    for w in plan:
        for d in w.days:
            s = d.session
            rows.append({
                "Week": w.week_num,
                "Phase": w.phase,
                "Date": f"{d.date.day} {d.date.strftime('%B %Y')}",
                "Day": d.date.strftime("%a"),
                "Session": s.type,
                "Detail": s.description,
                "Km": f"{s.distance_km:.0f}" if s.distance_km else "—",
                "Targets": ", ".join(f"{k}: {v}" for k, v in s.targets.items()) if s.targets else "—",
            })
    return rows
