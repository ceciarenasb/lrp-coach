"""Plan editing — move sessions between dates, validate week structure."""

from __future__ import annotations

import copy
from typing import Optional

_QUALITY = {"Tempo", "SVC Intervals", "Marathon Pace", "Progression Run"}
_HARD = _QUALITY | {"Long Run", "Medium-Long", "Club Run (LRP)"}
_REST = {"Rest"}


def move_session(
    plan: list, from_date: str, to_date: str
) -> tuple[list, list[str]]:
    """Swap session content between from_date and to_date.

    Returns (new_plan, warnings). The plan is deep-copied; original is unchanged.
    Dates are kept in place — only session_type/description/distance_km/targets swap.
    """
    if from_date == to_date:
        return plan, ["Cannot move a session to the same day."]

    plan = copy.deepcopy(plan)
    src_wi, src_di = _find_day(plan, from_date)
    tgt_wi, tgt_di = _find_day(plan, to_date)

    if src_wi is None:
        return plan, [f"Date {from_date} not found in plan."]
    if tgt_wi is None:
        return plan, [f"Date {to_date} not found in plan."]

    src = plan[src_wi]["days"][src_di]
    tgt = plan[tgt_wi]["days"][tgt_di]

    _FIELDS = ("session_type", "description", "distance_km", "targets")
    src_vals = {k: src[k] for k in _FIELDS}
    tgt_vals = {k: tgt[k] for k in _FIELDS}

    for k in _FIELDS:
        plan[src_wi]["days"][src_di][k] = tgt_vals[k]
        plan[tgt_wi]["days"][tgt_di][k] = src_vals[k]

    warnings: list[str] = []
    for wi in sorted(set([src_wi, tgt_wi])):
        warnings.extend(validate_week(plan[wi]["days"]))

    return plan, warnings


def validate_week(week_days: list) -> list[str]:
    """Return warning strings for hard-session spacing violations. Empty = OK."""
    warnings: list[str] = []
    hard = [d["weekday"] for d in week_days if d["session_type"] in _HARD]
    for i, d1 in enumerate(hard):
        for d2 in hard[i + 1 :]:
            gap = min(abs(d1 - d2), 7 - abs(d1 - d2))
            if gap < 2:
                warnings.append(
                    f"Hard sessions only {gap} day(s) apart (weekday {d1} & {d2})."
                )
    return warnings


def _find_day(plan: list, date_iso: str) -> tuple[Optional[int], Optional[int]]:
    for wi, week in enumerate(plan):
        for di, day in enumerate(week.get("days", [])):
            if day["date"] == date_iso:
                return wi, di
    return None, None
