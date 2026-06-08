"""
Apply mid-plan overrides to stored plan weeks.

Rules decide what changes; the LLM explains why and coaches through it.
"""

from __future__ import annotations

from .plan import (
    CLUB_RUN, EASY, LONG, MP_RUN, RECOVERY, REST,
    SVC_INTERVAL, TEMPO,
)
from .zones import Zones, fmt_pace

QUALITY_TYPES = {TEMPO, SVC_INTERVAL, MP_RUN}


def apply(
    plan: list,
    from_week: int,
    num_weeks: int,        # 0 = all remaining weeks
    no_club_run: bool,
    easy_only: bool,
    volume_pct: float,     # 1.0 = no change, 0.8 = −20 %
    zones_data: dict,
) -> tuple[list, list]:
    """
    Modify plan weeks in-place and return (updated_plan, change_log).
    Only affects weeks in [from_week, from_week + num_weeks).
    """
    z = Zones(**zones_data)
    easy_pace_str = f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}"
    log = []

    to_week = (from_week + num_weeks - 1) if num_weeks > 0 else len(plan)

    for wk in plan:
        wk_num = wk["week_num"]
        if wk_num < from_week or wk_num > to_week:
            continue

        for d in wk["days"]:
            st = d["session_type"]

            # Replace club run with easy run
            if no_club_run and st == CLUB_RUN:
                d["session_type"]  = EASY
                d["description"]   = f"Easy {d['distance_km']:.0f} km — replacing LRP (physio / restricted)"
                d["targets"]       = {"pace": easy_pace_str}
                log.append(f"Week {wk_num} {d['date']}: LRP club run → easy run")

            # Replace all quality sessions with easy runs
            if easy_only and st in QUALITY_TYPES:
                km = max(8.0, d["distance_km"])
                d["session_type"]  = EASY
                d["description"]   = f"Easy {km:.0f} km — quality session removed (recovery period)"
                d["distance_km"]   = km
                d["targets"]       = {"pace": easy_pace_str}
                log.append(f"Week {wk_num} {d['date']}: {st} → easy run")

            # Scale run distances
            if volume_pct != 1.0 and d["distance_km"] > 0 and st not in (REST, RECOVERY):
                old = d["distance_km"]
                new = round(max(5.0, old * volume_pct), 1)
                d["distance_km"]  = new
                # Patch km number in description string
                d["description"]  = d["description"].replace(
                    f"{old:.0f} km", f"{new:.0f} km", 1
                )

        if volume_pct != 1.0:
            wk["target_km"] = round(wk["target_km"] * volume_pct, 1)

        if no_club_run or easy_only or volume_pct != 1.0:
            wk["focus"] = wk["focus"].rstrip() + "  ⚠ adjusted"

    return plan, log


def build_adjustment_context(
    athlete_name: str,
    goal_race: str,
    weeks_left: int,
    phase: str,
    user_message: str,
    no_club_run: bool,
    easy_only: bool,
    volume_pct: float,
    from_week: int,
    num_weeks: int,
    change_log: list,
) -> str:
    duration = f"weeks {from_week}–{from_week + num_weeks - 1}" if num_weeks > 0 else f"week {from_week} onwards"
    changes_str = "\n".join(f"  • {c}" for c in change_log) if change_log else "  • No structural changes"
    flags = []
    if no_club_run: flags.append("LRP club runs replaced with easy runs")
    if easy_only:   flags.append("All quality sessions (tempo/SVC/M-pace) replaced with easy runs")
    if volume_pct != 1.0:
        pct = int((volume_pct - 1) * 100)
        flags.append(f"Volume adjusted {'+' if pct > 0 else ''}{pct}%")
    flags_str = "\n".join(f"  • {f}" for f in flags) if flags else "  • No flags"

    return f"""ATHLETE: {athlete_name}
GOAL: {goal_race} | {weeks_left} weeks to race | Phase: {phase}

ATHLETE'S MESSAGE:
"{user_message}"

ADJUSTMENTS APPLIED ({duration}):
{flags_str}

CHANGES TO PLAN:
{changes_str}

Write a coaching response that:
1. Acknowledges the situation honestly (physio / fatigue / life — whatever applies)
2. Explains what was changed in the plan and why it's the right call
3. Gives 1–2 specific things to focus on during this restricted period
4. Ends with a clear message about what happens when they're ready to return to full training
Keep it to 200–230 words. Be direct and supportive, not generic."""
