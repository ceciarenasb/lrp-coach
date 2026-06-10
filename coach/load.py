"""
Training-load scoring — Banister HR-based TRIMP with pace/distance fallbacks.

Consumes a run dict as produced by fit.py.summarize().
Public API: compute(run, hr_max, hr_rest, zones=None) -> float | None
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .zones import Zones

_TRIMP_B = 1.67   # Banister exponent — unisex value (original paper: 1.92 male / 1.67 female)
_KM_PROXY = 6.0   # TRIMP units per km when no HR and no zones available (easy-run proxy)


def compute(
    run: dict,
    hr_max: int,
    hr_rest: int,
    zones: "Optional[Zones]" = None,
) -> Optional[float]:
    """
    Return Banister TRIMP for one run dict (from fit.py.summarize()).

    Fallback chain:
      1. HR-TRIMP   — used when avg_hr is present and plausible
      2. Pace-based — used when HR is absent/noisy but zones are available
      3. Distance   — last resort: distance_km × constant

    Returns None if the run dict has an error key or zero duration.
    """
    if run.get("error"):
        return None

    duration_s = run.get("duration_s") or 0
    if duration_s <= 0:
        return None
    duration_min = duration_s / 60.0

    avg_hr = run.get("avg_hr")
    hrr = hr_max - hr_rest

    # --- Path 1: Banister HR-TRIMP ---
    if (
        avg_hr is not None
        and hrr > 10
        and avg_hr > hr_rest + 10   # clearly above resting
        and avg_hr < hr_max + 15    # not wildly over max (sensor spike guard)
    ):
        ratio = max(0.01, min(0.99, (avg_hr - hr_rest) / hrr))
        return round(duration_min * ratio * math.exp(_TRIMP_B * ratio), 2)

    # --- Path 2: pace-based intensity ---
    avg_pace_s = run.get("avg_pace_s")
    if avg_pace_s and avg_pace_s > 0 and zones is not None and zones.threshold > 0:
        # intensity_factor: >1 → faster than threshold (harder)
        intensity = zones.threshold / avg_pace_s
        intensity = max(0.2, min(3.0, intensity))
        return round(duration_min * 0.45 * intensity ** 2, 2)

    # --- Path 3: distance proxy ---
    dist_km = run.get("distance_km") or 0.0
    if dist_km > 0:
        return round(dist_km * _KM_PROXY, 2)

    return None


def weekly_load(history: list, anchor_date: str, days: int = 7) -> float:
    """Sum of training_load for runs in the `days`-day window ending on anchor_date."""
    from datetime import date, timedelta
    try:
        end = date.fromisoformat(anchor_date)
    except ValueError:
        end = date.today()
    start = (end - timedelta(days=days - 1)).isoformat()
    end_s = end.isoformat()
    return sum(
        (r.get("training_load") or 0.0)
        for r in history
        if start <= r.get("date", "") <= end_s
    )
