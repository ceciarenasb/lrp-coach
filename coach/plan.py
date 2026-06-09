"""
Marathon training plan generator — Daniels + SVC hybrid periodization.

Phases : Base → Build → Peak → Taper
Sessions: REST, RECOVERY, EASY, LONG, TEMPO, SVC_INTERVAL, MP_RUN,
          STRENGTH, CYCLING, CLUB_RUN
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .zones import Zones, fmt_pace

# Session type labels
REST         = "Rest"
RECOVERY     = "Recovery"
EASY         = "Easy"
LONG         = "Long Run"
TEMPO        = "Tempo"
SVC_INTERVAL = "SVC Intervals"
MP_RUN       = "Marathon Pace"
STRENGTH     = "Strength"
CYCLING      = "Cycling / Zwift"
CLUB_RUN     = "Club Run (LRP)"

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


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
    weekday: int   # 0 = Mon … 6 = Sun
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
# Volume targets by goal time
# ---------------------------------------------------------------------------

def _peak_km(goal_s: int) -> float:
    h = goal_s / 3600
    if h <= 3.0:
        return 90.0
    if h <= 3.5:
        return 78.0
    if h <= 4.0:
        return 65.0
    return 55.0


def _long_run_km(wk: int, total: int) -> float:
    """Progressive long run: 18 km week 1 → 32 km at peak → 16 km race week."""
    taper_start = total - 3
    if wk >= taper_start:
        return max(16.0, 32.0 - (wk - taper_start) * 6.0)
    prog = (wk - 1) / max(1, taper_start - 1)
    return round(18.0 + prog * 14.0, 1)


# ---------------------------------------------------------------------------
# Session builders
# ---------------------------------------------------------------------------

def _easy(km: float, z: Zones) -> Session:
    return Session(
        EASY,
        f"Easy {km:.0f} km — conversational pace, nasal breathing",
        distance_km=round(km, 1),
        targets={"pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}"},
    )


def _recovery(z: Zones) -> Session:
    return Session(
        RECOVERY,
        "Recovery 6 km — very easy, no pace pressure",
        distance_km=6.0,
        targets={"pace": f"slower than {fmt_pace(z.easy_lo)}"},
    )


def _long(km: float, phase: str, z: Zones) -> Session:
    if phase in ("Build", "Peak") and km >= 24:
        return Session(
            LONG,
            f"Long run {km:.0f} km — easy aerobic, final 8–10 km at M-pace",
            distance_km=km,
            targets={
                "main_pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}",
                "finish_at_M_pace": fmt_pace(z.marathon),
            },
        )
    return Session(
        LONG,
        f"Long run {km:.0f} km — easy aerobic throughout",
        distance_km=km,
        targets={"pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}"},
    )


def _quality(phase: str, wk: int, z: Zones, injury: str) -> Session:
    if injury == "moderate":
        return _easy(10, z)

    if phase == "Base":
        return Session(
            TEMPO,
            "Tempo — 10 min warm-up + 2×15 min at T-pace (2 min jog) + 10 min cool-down",
            distance_km=10.0,
            targets={"T_pace": fmt_pace(z.threshold)},
        )

    if phase == "Build":
        if wk % 2 == 0:
            return Session(
                SVC_INTERVAL,
                "SVC Intervals — 10 min warm-up + 5×1000 m at SVC pace (90 s jog) + 10 min cool-down",
                distance_km=11.0,
                targets={"SVC_pace": fmt_pace(z.cv_interval), "recovery": "90 s jog"},
            )
        return Session(
            TEMPO,
            "Tempo — 10 min warm-up + 30 min continuous at T-pace + 10 min cool-down",
            distance_km=12.0,
            targets={"T_pace": fmt_pace(z.threshold)},
        )

    if phase == "Peak":
        return Session(
            SVC_INTERVAL,
            "SVC Intervals — 10 min warm-up + 6×1000 m at SVC pace (75 s jog) + 10 min cool-down",
            distance_km=13.0,
            targets={"SVC_pace": fmt_pace(z.cv_interval), "recovery": "75 s jog"},
        )

    # Taper
    return Session(
        MP_RUN,
        "M-pace run — 10 min warm-up + 20 min at marathon pace + 10 min cool-down",
        distance_km=9.0,
        targets={"M_pace": fmt_pace(z.marathon)},
    )


# ---------------------------------------------------------------------------
# Week builder
# ---------------------------------------------------------------------------

def _build_week(
    week_start: date, wk: int, total: int, phase: str, zones: Zones,
    target_km: float, run_days: list, lrp_sessions: list,
    strength_days: list, cycling_days: list, injury: str,
) -> list:
    sessions: dict[int, Session] = {}

    # 1. Lock all LRP club sessions
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
        sessions[day] = Session(
            CLUB_RUN,
            f"LRP club run — {km:.0f} km ({stype})",
            distance_km=km,
            targets={"pace": pace_target},
        )

    # 2. Long run — prefer Sunday, else last available run day
    long_day = 6 if 6 in run_days else max(run_days)
    if long_day not in sessions:
        sessions[long_day] = _long(_long_run_km(wk, total), phase, zones)

    # 3. Quality session — prefer day ≥ 2 apart from long run
    quality_done = False
    for d in run_days:
        if d in sessions:
            continue
        if abs(d - long_day) >= 2:
            sessions[d] = _quality(phase, wk, zones, injury)
            quality_done = True
            break
    # Fallback: place quality adjacent if no non-adjacent day available
    if not quality_done:
        for d in run_days:
            if d not in sessions and d != long_day:
                sessions[d] = _quality(phase, wk, zones, injury)
                quality_done = True
                break

    # 4. Cross-training
    for d in strength_days:
        if d not in sessions:
            sessions[d] = Session(STRENGTH, "Strength — core, squats, lunges, hip mobility", 0, 45)
    for d in cycling_days:
        if d not in sessions:
            sessions[d] = Session(CYCLING, "Cycling / Zwift — steady Z2 aerobic cross-training", 0, 60)

    # 5. Fill remaining run days with easy runs
    used_km = sum(s.distance_km for s in sessions.values())
    remaining = [d for d in run_days if d not in sessions]
    for d in remaining:
        share = max(6.0, min(14.0, (target_km - used_km) / max(1, len(remaining))))
        sessions[d] = _easy(round(share, 1), zones)
        used_km += share

    # Place each weekday on its actual calendar date within this 7-day block.
    # week_start may be any weekday; (wd - week_start.weekday()) % 7 gives the
    # correct offset so Monday always lands on a real Monday, Saturday on a real
    # Saturday, etc. — regardless of what day the plan started on.
    def _cal_date(wd: int) -> date:
        return week_start + timedelta(days=(wd - week_start.weekday()) % 7)

    return sorted(
        [
            DayPlan(
                date=_cal_date(wd),
                weekday=wd,
                session=sessions.get(wd, Session(REST, "Rest or active recovery (walk, stretch)")),
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
    # Legacy single-session params — kept for backward compat, ignored if lrp_sessions non-empty
    lrp_day: Optional[int] = None,
    lrp_km: float = 12.0,
    lrp_type: str = "easy",
) -> list:
    if not run_days:
        return []

    # Back-compat: if caller passes old single-session params, wrap them
    if not lrp_sessions and lrp_day is not None:
        lrp_sessions = [{"day": lrp_day, "km": lrp_km, "type": lrp_type}]

    today = date.today() + timedelta(days=1)  # plan starts tomorrow
    weeks = max(4, min(20, (marathon_date - today).days // 7))
    peak = _peak_km(goal_time_s)
    base = start_km if start_km else peak * 0.55

    if injury == "light":
        peak *= 0.85
        base *= 0.85
    elif injury == "moderate":
        peak *= 0.70
        base *= 0.70

    # Merge LRP days into available run days
    for s in lrp_sessions:
        day = s.get("day")
        if day is not None and day not in run_days:
            run_days = sorted(run_days + [day])

    # Trim to desired runs_per_week, always keeping LRP days
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
        return "Build aerobic base — easy volume, weekly tempo to establish threshold"
    if phase == "Build":
        return "Develop race fitness — SVC intervals, long runs with M-pace finish"
    if phase == "Peak":
        return "Peak load week — absorb the fatigue, trust the training"
    return f"Taper week {taper_wk}/3 — reduce volume, keep a touch of intensity, arrive fresh"


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
