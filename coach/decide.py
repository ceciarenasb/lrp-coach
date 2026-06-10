"""
decide.py — the coaching decision engine.

For each weekly check-in:
  1. Run the PROGRESS/MAINTAIN/CONSOLIDATE/RECOVER ladder (adapt.py) as a CLAMP.
  2. Enumerate N candidate weekly load levels within the clamp band.
  3. For each candidate, simulate fitness-fatigue forward to race day (state_model.py).
  4. Score projected race-day Form.
  5. Hard-filter violations: volume cap, ACWR band, pain gate.
  6. Pick the best surviving candidate.
  7. Apply per-week availability override to the chosen session structure.
  8. Return a WeekProposal.

Club-run rules (hard constraints, never overridden by the optimizer):
  Monday LRP   — day, distance, and type are all FIXED. Never moved, resized, or dropped
                 unless the user explicitly sets club_run_decision = "skip".
  Saturday LRP — day is pinned by default. Distance comes from the user's input this week;
                 the optimizer takes it as given and fills the other sessions around it.

All logic is deterministic Python. The LLM never touches this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .adapt import WeekMetrics, score_week
from .plan import (
    CLUB_RUN, EASY, LONG, MEDIUM_LONG, MP_RUN, PROGRESSION,
    RECOVERY, REST, SVC_INTERVAL, STRENGTH, CYCLING, TEMPO,
    Session, DayPlan, build_week,
    _focus, _long_run_km, _quality, _easy, _recovery, _medium_long,
)
from .readiness import score as readiness_score, label as readiness_label
from .state_model import ModelState, from_dict as sm_from_dict, simulate_forward, score_race_day_form
from .zones import Zones, build_zones

_N_CANDIDATES  = 7
_ACWR_LO       = 0.8
_ACWR_HI       = 1.3
_ACWR_REHAB_HI = 1.1
_LONG_RUN_MAX_KM = 35.0
_LONG_RAMP_PCT   = 0.10   # max 10 % week-over-week long-run increase

# Typical HR ratio (avg_HR − hr_rest) / (hr_max − hr_rest) by session type
# Used to estimate planned-session TRIMP before the FIT arrives
_HR_RATIO = {
    RECOVERY: 0.50, EASY: 0.58, MEDIUM_LONG: 0.65, LONG: 0.63,
    TEMPO: 0.80, SVC_INTERVAL: 0.86, MP_RUN: 0.76, PROGRESSION: 0.72,
    CLUB_RUN: 0.60, STRENGTH: 0.0, CYCLING: 0.0, REST: 0.0,
}
_PACE_MIN_PER_KM = {
    RECOVERY: 7.2, EASY: 6.2, MEDIUM_LONG: 5.8, LONG: 6.5,
    TEMPO: 5.0, SVC_INTERVAL: 5.2, MP_RUN: 5.5, PROGRESSION: 5.8,
    CLUB_RUN: 6.2,
}


# ---------------------------------------------------------------------------
# Public dataclass returned to callers (app.py, llm.py)
# ---------------------------------------------------------------------------

@dataclass
class WeekProposal:
    week_iso: str
    phase: str
    focus: str
    target_km: float
    saturday_km: float         # the saturday long-run distance used (user or suggested)
    sessions: list             # list[DayPlan] — serialised by app.py
    load_target: float
    readiness: float
    readiness_label: str
    acwr: float
    ladder_decision: str
    ladder_score: int
    warnings: list[str] = field(default_factory=list)
    # Context dict passed straight into llm.build_context
    coaching_context: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load estimation helpers
# ---------------------------------------------------------------------------

def _session_trimp(session: Session, hr_max: int, hr_rest: int) -> float:
    """Rough planned-session TRIMP (pre-FIT, HR assumed from session type)."""
    import math
    if not session.distance_km or session.type in (REST, STRENGTH, CYCLING):
        return 0.0
    ratio = _HR_RATIO.get(session.type, 0.60)
    if ratio == 0.0:
        return 0.0
    mins = session.distance_km * _PACE_MIN_PER_KM.get(session.type, 6.0)
    return round(mins * ratio * math.exp(1.67 * ratio), 2)


def _chronic_avg(history: list, days: int = 42) -> float:
    """Average daily TRIMP over the last `days` days."""
    if not history:
        return 0.0
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    loads  = [r.get("training_load") or 0.0 for r in history if r.get("date", "") >= cutoff]
    return sum(loads) / days


def _acute_avg(history: list, days: int = 7) -> float:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    loads  = [r.get("training_load") or 0.0 for r in history if r.get("date", "") >= cutoff]
    return sum(loads) / days


def _acwr(history: list) -> float:
    c = _chronic_avg(history, 42) or 1.0
    a = _acute_avg(history, 7)
    return round(a / c, 3)


def _days_to_race(race_date_str: str) -> int:
    try:
        return max(0, (date.fromisoformat(race_date_str) - date.today()).days)
    except Exception:
        return 84


# ---------------------------------------------------------------------------
# Availability override logic
# ---------------------------------------------------------------------------

def _apply_override(
    day_plans: list,
    config: dict,
    override: dict,
    saturday_km: float,
) -> tuple[list, list]:
    """
    Apply a per-week availability override to a list of DayPlan objects.
    Returns (modified_day_plans, warnings).

    Rules (priority order):
      1. Saturday long run: if day unavailable, move to nearest available day.
      2. Monday club run: keep / skip / substitute per club_run_decision.
      3. Other run days marked unavailable: drop easy/recovery first; keep quality.
      4. Never compensate for skipped volume by adding to other days.
    """
    if not override:
        return day_plans, []

    warnings: list[str] = []
    club_decisions: dict = override.get("club_run_decisions", {})
    unavailable: set     = set(override.get("unavailable_days", []))
    available: set       = set(override.get("available_days", list(range(7)))) - unavailable
    day_caps: dict       = override.get("day_caps", {})

    # Index by weekday for easy lookup
    by_wd: dict[int, DayPlan] = {dp.weekday: dp for dp in day_plans}

    # --- Saturday long run ---
    sat_wd = 5
    sat_decision = club_decisions.get("lrp_saturday", "keep")
    if sat_wd in unavailable and isinstance(sat_decision, dict) and sat_decision.get("type") == "substitute_to_day":
        new_day = sat_decision["day"]
        if sat_wd in by_wd:
            old_dp = by_wd.pop(sat_wd)
            new_date = old_dp.date + timedelta(days=(new_day - sat_wd) % 7)
            by_wd[new_day] = DayPlan(date=new_date, weekday=new_day, session=old_dp.session)
            warnings.append(f"Saturday long run moved to {new_date.strftime('%A')}")
    elif sat_wd in unavailable and sat_decision == "keep":
        # Auto-flex to nearest available day (Friday preferred, then Sunday)
        for candidate in [4, 6, 3]:
            if candidate in available and candidate not in by_wd:
                if sat_wd in by_wd:
                    old_dp = by_wd.pop(sat_wd)
                    delta = (candidate - sat_wd) % 7
                    new_date = old_dp.date + timedelta(days=delta)
                    by_wd[candidate] = DayPlan(date=new_date, weekday=candidate, session=old_dp.session)
                    warnings.append(f"Saturday long run moved to {new_date.strftime('%A')} (Saturday unavailable)")
                break

    # --- Monday club run ---
    mon_wd = 0
    mon_decision = club_decisions.get("lrp_monday", "keep")
    if mon_decision == "skip":
        if mon_wd in by_wd and by_wd[mon_wd].session.type == CLUB_RUN:
            by_wd.pop(mon_wd)
            warnings.append("Monday LRP club run skipped this week")
    elif isinstance(mon_decision, dict) and mon_decision.get("type") == "substitute_to_day":
        new_day = mon_decision["day"]
        if mon_wd in by_wd:
            old_dp = by_wd.pop(mon_wd)
            new_date = old_dp.date + timedelta(days=(new_day - mon_wd) % 7)
            by_wd[new_day] = DayPlan(date=new_date, weekday=new_day, session=old_dp.session)
            warnings.append(f"Monday LRP substituted on {new_date.strftime('%A')}")

    # --- Drop sessions on remaining unavailable days (priority: drop easy first) ---
    DROPPABLE = {EASY, RECOVERY, REST}
    HARD_TYPES = {LONG, MEDIUM_LONG, TEMPO, SVC_INTERVAL, MP_RUN, PROGRESSION}
    for wd in list(unavailable):
        if wd in by_wd:
            s_type = by_wd[wd].session.type
            if s_type in DROPPABLE:
                by_wd.pop(wd)
            elif s_type in HARD_TYPES:
                # Try to reschedule to an available free day
                moved = False
                for cand in sorted(available):
                    if cand not in by_wd and abs(cand - wd) >= 2:
                        old_dp = by_wd.pop(wd)
                        delta = (cand - wd) % 7
                        new_date = old_dp.date + timedelta(days=delta)
                        by_wd[cand] = DayPlan(date=new_date, weekday=cand, session=old_dp.session)
                        warnings.append(f"Quality session moved from {old_dp.date.strftime('%A')} to {new_date.strftime('%A')}")
                        moved = True
                        break
                if not moved:
                    by_wd.pop(wd)
                    warnings.append(f"Quality session on {wd} dropped (no suitable available day)")

    # --- Apply per-day caps ---
    for wd_str, cap in day_caps.items():
        wd = int(wd_str)
        if wd in by_wd and cap.get("type") == "easy":
            dp = by_wd[wd]
            if dp.session.type not in (REST, STRENGTH, CYCLING, CLUB_RUN):
                max_min = cap.get("max_duration_min", 60)
                capped_km = min(dp.session.distance_km, max_min / 6.5)
                by_wd[wd] = DayPlan(
                    date=dp.date, weekday=dp.weekday,
                    session=Session(EASY, f"Easy {capped_km:.0f} km (capped)", capped_km),
                )

    result = sorted(by_wd.values(), key=lambda dp: dp.date)
    return result, warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def propose_week(
    cycle: dict,
    history: list,
    feeling: float = 3.0,
    saturday_km: Optional[float] = None,
    availability: Optional[dict] = None,
) -> WeekProposal:
    """
    Propose the next week's training for `cycle`.

    Parameters
    ----------
    cycle        : active race-cycle dict from state.json
    history      : full run history list from state.json
    feeling      : 1–5 subjective score from the weekly check-in
    saturday_km  : user-entered Saturday long-run distance (None → use suggestion)
    availability : per-week override dict (None → use default schedule)
    """
    config     = cycle.get("config", {})
    race_date  = cycle["race"]["date"]
    injury     = config.get("injury_level", "none")
    vol_cap    = float(config.get("volume_cap_km", 75.0))
    hr_max     = int(cycle.get("athlete", {}).get("hr_max") or 177)
    hr_rest    = int(cycle.get("athlete", {}).get("hr_rest") or 50)

    # --- Zones ---
    zd = cycle.get("zones", {})
    zones = (
        Zones(**{k: v for k, v in zd.items() if k in Zones.__dataclass_fields__})
        if zd else build_zones(40.0, 0.0)
    )

    # --- Readiness ---
    cutoff7  = (date.today() - timedelta(days=7)).isoformat()
    cutoff14 = (date.today() - timedelta(days=14)).isoformat()
    recent   = [r for r in history if r.get("date", "") >= cutoff7]
    prev_wk  = [r for r in history if cutoff14 <= r.get("date", "") < cutoff7]

    def _easy_hr(runs):
        hrs = [r["avg_hr"] for r in runs
               if r.get("avg_hr") and (r.get("avg_pace_s") or 0) > 330]
        return sum(hrs) / len(hrs) if hrs else None

    easy_hr   = _easy_hr(recent)
    prev_hr   = _easy_hr(prev_wk)
    drifts    = [r["hr_drift_pct"] for r in recent if r.get("hr_drift_pct") is not None]
    avg_drift = sum(drifts) / len(drifts) if drifts else None

    readiness  = readiness_score(feeling, easy_hr, prev_hr, avg_drift)
    r_label    = readiness_label(readiness)

    # --- Adaptation ladder (clamp) ---
    actual_km_7d = sum(
        r.get("distance_km", 0) for r in recent
        if r.get("activity_type", "Run") == "Run"
    )
    plan_weeks   = cycle.get("plan", [])
    history_log  = cycle.get("check_in_history", [])
    last_target  = history_log[-1].get("load_target_km", 50.0) if history_log else 50.0

    q_paces    = [r["avg_pace_s"] for r in recent if r.get("avg_pace_s") and r["avg_pace_s"] < 360]
    q_pace     = min(q_paces) if q_paces else None

    metrics = WeekMetrics(
        actual_km             = actual_km_7d,
        target_km             = last_target,
        quality_avg_pace_s    = q_pace,
        quality_target_pace_s = zones.threshold if q_pace else None,
        easy_avg_hr           = easy_hr,
        prev_easy_hr          = prev_hr,
        avg_hr_drift          = avg_drift,
        avg_feeling           = feeling,
    )
    ladder = score_week(metrics)

    # --- Volume band from clamp ---
    chronic_daily = _chronic_avg(history, 42) or 5.0
    base_km_week  = chronic_daily * 7

    # Clamp multipliers from ladder decision
    lo_mult = {"RECOVER": 0.75, "CONSOLIDATE": 0.88, "MAINTAIN": 0.95, "PROGRESS": 1.00}[ladder.decision]
    hi_mult = {"RECOVER": 0.88, "CONSOLIDATE": 1.00, "MAINTAIN": 1.06, "PROGRESS": 1.10}[ladder.decision]

    # Pain gate (adjustments.py override) — tighter band
    if injury == "moderate":
        hi_mult = min(hi_mult, 0.80)
        lo_mult = min(lo_mult, 0.70)

    lo_km = max(15.0, base_km_week * lo_mult)
    hi_km = min(vol_cap, base_km_week * hi_mult)
    if lo_km > hi_km:
        lo_km, hi_km = hi_km * 0.9, hi_km

    # --- Plan position ---
    days_left   = _days_to_race(race_date)
    total_weeks = max(4, days_left // 7)
    current_wk  = len(history_log) + 1
    pct         = current_wk / total_weeks
    phase       = ("Base" if pct < 0.35 else "Build" if pct < 0.70
                   else "Peak" if pct < 0.85 else "Taper")

    # --- Saturday distance ---
    last_long = max(
        (r.get("distance_km", 0) for r in history[-14:] if r.get("distance_km", 0) > 14),
        default=14.0,
    )
    suggested_sat = min(_LONG_RUN_MAX_KM, _long_run_km(current_wk, total_weeks))
    warnings: list[str] = []

    if saturday_km is None:
        saturday_km = suggested_sat
    else:
        sat_max = min(_LONG_RUN_MAX_KM, last_long * (1 + _LONG_RAMP_PCT))
        if saturday_km > sat_max:
            warnings.append(
                f"Saturday {saturday_km:.0f} km exceeds safe ramp limit "
                f"({sat_max:.0f} km max). Clamped."
            )
            saturday_km = sat_max

    # --- ACWR ---
    cur_acwr     = _acwr(history)
    rehab        = injury != "none"
    acwr_ceiling = _ACWR_REHAB_HI if rehab else _ACWR_HI

    # --- State model for forward simulation ---
    sm = sm_from_dict(cycle.get("state_model", {}))

    # --- Enumerate and score candidates ---
    candidates = [lo_km + (hi_km - lo_km) * i / (_N_CANDIDATES - 1)
                  for i in range(_N_CANDIDATES)]

    best_score = -1.0
    best_km    = candidates[len(candidates) // 2]

    for cand_km in candidates:
        # Quick ACWR projection for this candidate
        cand_daily = cand_km * 6.0 / 7.0
        proj_acwr  = ((_acute_avg(history, 7) * 6 + cand_daily) / 7) / (_chronic_avg(history, 42) or 1.0)
        if proj_acwr > acwr_ceiling or proj_acwr < _ACWR_LO:
            continue
        if cand_km > vol_cap:
            continue

        # Simulate forward to race day with a simplified taper schedule
        wks_left = max(1, days_left // 7)
        load_sched = []
        for w in range(wks_left):
            day_offset = (w + 1) * 7
            taper_f    = max(0.4, 1.0 - max(0, w - (wks_left - 4)) * 0.2)
            wk_load    = cand_km * 6.0 * taper_f
            load_sched.append((day_offset, wk_load / 7.0))  # daily average

        snaps = simulate_forward(sm, load_sched)
        form_score = score_race_day_form(snaps[-1]) if snaps else 0.0

        if form_score > best_score:
            best_score = form_score
            best_km    = cand_km

    # Apply readiness scaling
    target_km = round(max(15.0, min(vol_cap, best_km * readiness)), 1)

    # --- Build concrete sessions ---
    run_days     = list(config.get("default_run_days", [0, 1, 3, 5]))
    strength_d   = list(config.get("strength_days", []))
    cycling_d    = list(config.get("cycling_days", []))
    club_runs    = config.get("club_runs", [])

    lrp_sessions = []
    for cr in club_runs:
        cr_id     = cr.get("id", "")
        cr_day    = cr.get("day")
        if cr_day is None:
            continue
        if cr_id == "lrp_saturday":
            km = saturday_km
        else:
            km = float(cr.get("distance_km", 10.0))
        lrp_sessions.append({"day": cr_day, "km": km, "type": cr.get("type", "easy")})

    week_start = date.today() + timedelta(days=1)
    # Snap to week-start (Monday)
    week_start = week_start - timedelta(days=week_start.weekday())

    day_plans = build_week(
        week_start    = week_start,
        wk            = current_wk,
        total         = total_weeks,
        phase         = phase,
        zones         = zones,
        target_km     = target_km,
        run_days      = run_days,
        lrp_sessions  = lrp_sessions,
        strength_days = strength_d,
        cycling_days  = cycling_d,
        injury        = injury,
    )

    # Apply availability override
    if availability:
        day_plans, override_warnings = _apply_override(day_plans, config, availability, saturday_km)
        warnings.extend(override_warnings)

    # Recompute planned load
    load_target = sum(_session_trimp(dp.session, hr_max, hr_rest) for dp in day_plans)

    week_iso = f"{date.today().isocalendar()[0]}-W{date.today().isocalendar()[1]:02d}"
    focus    = _focus(phase, current_wk, total_weeks)

    return WeekProposal(
        week_iso         = week_iso,
        phase            = phase,
        focus            = focus,
        target_km        = target_km,
        saturday_km      = saturday_km,
        sessions         = day_plans,
        load_target      = round(load_target, 1),
        readiness        = readiness,
        readiness_label  = r_label,
        acwr             = cur_acwr,
        ladder_decision  = ladder.decision,
        ladder_score     = ladder.score,
        warnings         = warnings,
        coaching_context = {
            "ladder":     ladder,
            "metrics":    metrics,
            "readiness":  readiness,
            "easy_hr":    easy_hr,
            "prev_hr":    prev_hr,
            "avg_drift":  avg_drift,
            "acwr":       cur_acwr,
            "days_left":  days_left,
            "phase":      phase,
        },
    )
