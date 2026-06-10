"""
Marathon training plan generator — Daniels + SVC hybrid periodization.

Phases : Base → Build → Peak → Taper
Hard/easy principle: quality/long days always separated by ≥1 easy/rest day.
Cross-training combines with run days (noted in description) rather than blocking them.
"""

from __future__ import annotations

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
    return 55.0


def _long_run_km(wk: int, total: int) -> float:
    taper_start = total - 3
    if wk >= taper_start:
        return max(16.0, 32.0 - (wk - taper_start) * 6.0)
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


def _quality(phase: str, wk: int, z: Zones, injury: str, q_num: int = 1) -> Session:
    if injury == "moderate":
        return _easy(10, z)

    cycle = wk % 3  # rotate through 3 session variants per phase

    if phase == "Base":
        variants = [
            Session(TEMPO,
                "Tempo — 2 km WU + 2×15 min at T-pace (2 min jog) + 2 km CD",
                distance_km=11.0,
                targets={"T-pace": fmt_pace(z.threshold), "WU/CD_pace": fmt_pace(z.easy_hi)}),
            Session(TEMPO,
                "Cruise intervals — 2 km WU + 5×1 km at T-pace (60 s jog) + 2 km CD",
                distance_km=10.0,
                targets={"T-pace": fmt_pace(z.threshold), "recovery": "60 s jog"}),
            Session(TEMPO,
                "Tempo — 2 km WU + 20 min continuous at T-pace + 2 km CD",
                distance_km=10.0,
                targets={"T-pace": fmt_pace(z.threshold)}),
        ]
        return variants[cycle]

    if phase == "Build":
        if q_num == 2:
            # Second quality in Build = medium aerobic with strides
            return Session(EASY,
                "General aerobic 13 km + 8×100 m strides — steady effort, Z2 heart rate",
                distance_km=13.0,
                targets={"pace": f"{fmt_pace(z.easy_hi)} – {fmt_pace(z.marathon)}", "strides": "near 5 K effort"})
        variants = [
            Session(SVC_INTERVAL,
                "SVC Intervals — 3 km WU + 5×1000 m at SVC pace (90 s jog) + 2 km CD",
                distance_km=12.0,
                targets={"SVC-pace": fmt_pace(z.cv_interval), "recovery": "90 s jog"}),
            Session(TEMPO,
                "Threshold — 3 km WU + 2×20 min at T-pace (2 min jog) + 2 km CD",
                distance_km=13.0,
                targets={"T-pace": fmt_pace(z.threshold)}),
            Session(SVC_INTERVAL,
                "VO₂max — 3 km WU + 6×800 m at I-pace (2 min jog) + 2 km CD",
                distance_km=12.0,
                targets={"I-pace": fmt_pace(z.interval), "recovery": "2 min jog"}),
        ]
        return variants[cycle]

    if phase == "Peak":
        if q_num == 2:
            return Session(PROGRESSION,
                "Progression run 16 km — 6 km easy, 6 km at M-pace, 4 km at T-pace",
                distance_km=16.0,
                targets={
                    "easy_section": fmt_pace(z.easy_hi),
                    "M-pace_section": fmt_pace(z.marathon),
                    "T-pace_section": fmt_pace(z.threshold),
                })
        variants = [
            Session(SVC_INTERVAL,
                "SVC Intervals — 3 km WU + 6×1000 m at SVC pace (75 s jog) + 2 km CD",
                distance_km=13.0,
                targets={"SVC-pace": fmt_pace(z.cv_interval), "recovery": "75 s jog"}),
            Session(MP_RUN,
                "Race-specific — 3 km WU + 16 km at marathon pace + 2 km CD",
                distance_km=21.0,
                targets={"M-pace": fmt_pace(z.marathon)}),
            Session(TEMPO,
                "Threshold — 3 km WU + 2×20 min at T-pace (90 s jog) + 2 km CD",
                distance_km=13.0,
                targets={"T-pace": fmt_pace(z.threshold)}),
        ]
        return variants[cycle]

    # Taper
    if cycle == 0:
        return Session(MP_RUN,
            "Taper sharpener — 2 km WU + 20 min at marathon pace + 2 km CD",
            distance_km=9.0,
            targets={"M-pace": fmt_pace(z.marathon)})
    return Session(TEMPO,
        "Taper tune-up — 2 km WU + 4×1 km at T-pace (90 s jog) + 2 km CD",
        distance_km=9.0,
        targets={"T-pace": fmt_pace(z.threshold)})


# ---------------------------------------------------------------------------
# Week builder
# ---------------------------------------------------------------------------

def _build_week(
    week_start: date, wk: int, total: int, phase: str, zones: Zones,
    target_km: float, run_days: list, lrp_sessions: list,
    strength_days: list, cycling_days: list, injury: str,
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
    for s in lrp_sessions:
        day = s.get("day")
        if day is None:
            continue
        km    = float(s.get("km", 12.0))
        stype = s.get("type", "easy")
        if stype == "tempo":
            pace_target = fmt_pace(zones.threshold)
        elif stype == "long":
            pace_target = fmt_pace(zones.easy_lo)
        else:
            pace_target = fmt_pace(zones.easy_hi)
        note = _note(day)
        extra = f" + {note}" if note else ""
        sessions[day] = Session(
            CLUB_RUN,
            f"LRP club run — {km:.0f} km ({stype}){extra}",
            distance_km=km,
            targets={"pace": pace_target},
        )

    # 2. Long run — prefer Sunday, otherwise latest calendar day
    long_day = 6 if 6 in run_days else max(run_days, key=_off)
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
        return {d for d, s in sessions.items() if s.type in _HARD}

    def _place_first_free(candidate_fn, min_dist: int = 2):
        hard_placed = _hard_days_so_far()
        for d in run_days_cal:
            if d in sessions:
                continue
            if not hard_placed or all(_day_dist(d, h) >= min_dist for h in hard_placed):
                sessions[d] = candidate_fn(d)
                return True
        if min_dist > 1:
            return _place_first_free(candidate_fn, min_dist - 1)
        return False

    # 4. Primary quality session
    _place_first_free(lambda _: _quality(phase, wk, zones, injury, 1))

    # 5. Medium-long run in Build/Peak with ≥4 run days
    if phase in ("Build", "Peak") and len(run_days) >= 4:
        ml_km = _medium_long_km(_long_run_km(wk, total))
        _place_first_free(lambda _: _medium_long(ml_km, phase, zones))

    # 6. Second quality (strides/aerobic) in Build/Peak with ≥5 run days
    if phase in ("Build", "Peak") and len(run_days) >= 5:
        _place_first_free(lambda _: _quality(phase, wk, zones, injury, 2))

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
    allow_volume_increase: bool = True,
    lrp_day: Optional[int] = None,
    lrp_km: float = 12.0,
    lrp_type: str = "easy",
) -> list:
    if not run_days:
        return []

    if not lrp_sessions and lrp_day is not None:
        lrp_sessions = [{"day": lrp_day, "km": lrp_km, "type": lrp_type}]

    today = date.today() + timedelta(days=1)
    weeks = max(4, min(20, (marathon_date - today).days // 7))
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

    if runs_per_week and 0 < runs_per_week < len(run_days):
        lrp_days = [s["day"] for s in lrp_sessions if s.get("day") is not None]
        non_lrp = [d for d in run_days if d not in lrp_days]
        slots_left = max(0, runs_per_week - len(lrp_days))
        if slots_left and non_lrp:
            step = len(non_lrp) / slots_left
            extra = [non_lrp[int(i * step)] for i in range(slots_left)]
        else:
            extra = []
        run_days = sorted(set(lrp_days + extra))

    taper_start = weeks - 3
    plan = []

    for wk in range(1, weeks + 1):
        pct = wk / weeks
        if pct < 0.35:
            phase = "Base"
        elif pct < 0.70:
            phase = "Build"
        elif pct < 0.85:
            phase = "Peak"
        else:
            phase = "Taper"

        if wk >= taper_start:
            target_km = max(peak * 0.40, peak * (1 - (wk - taper_start) * 0.22))
        elif not allow_volume_increase:
            target_km = base
        else:
            target_km = base + (peak - base) * ((wk - 1) / max(1, taper_start - 1))

        week_start = today + timedelta(weeks=wk - 1)
        days = _build_week(
            week_start, wk, weeks, phase, zones,
            target_km, list(run_days), lrp_sessions,
            list(strength_days), list(cycling_days), injury,
        )
        plan.append(WeekPlan(
            week_num=wk,
            start_date=week_start,
            phase=phase,
            focus=_focus(phase, wk, weeks),
            target_km=round(target_km, 1),
            days=days,
        ))

    return plan


def _focus(phase: str, wk: int, total: int) -> str:
    taper_wk = wk - (total - 3)
    if phase == "Base":
        return "Build aerobic base — tempo runs to establish lactate threshold, easy volume"
    if phase == "Build":
        return "Race fitness — SVC intervals, VO₂max work, medium-long runs with M-pace finish"
    if phase == "Peak":
        return "Peak load — race-specific workouts, long runs with M-pace blocks, absorb the fatigue"
    return f"Taper week {taper_wk}/3 — cut volume, keep sharpeners, arrive fresh and confident"


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
