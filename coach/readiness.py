"""
Readiness multiplier — folds subjective feeling + objective HR signals into
a single 0.60–1.10 scalar that scales prescribed session load/distance.

  base  = 0.85
  Δ feeling  = (feeling − 3) × 0.05          range  ±0.10
  Δ HR trend = −0.20 / −0.10 / 0 / +0.05    based on week-over-week easy HR ratio
  Δ HR drift = −0.15 / −0.08 / 0            based on avg cardiac drift %
  result     = clamp(base + Δ, 0.60, 1.10)
"""

from __future__ import annotations

from typing import Optional

_BASE = 0.85
_LO   = 0.60
_HI   = 1.10


def score(
    feeling: float,
    easy_hr: Optional[float],
    prev_easy_hr: Optional[float],
    avg_hr_drift: Optional[float],
) -> float:
    """
    Return a readiness multiplier in [0.60, 1.10].

    Parameters
    ----------
    feeling       : subjective rating 1 (exhausted) – 5 (great)
    easy_hr       : average HR on this week's easy/recovery runs (bpm)
    prev_easy_hr  : same for the previous week
    avg_hr_drift  : average cardiac drift % across recent sessions (positive = drifting up)
    """
    delta = (feeling - 3.0) * 0.05

    if easy_hr and prev_easy_hr and prev_easy_hr > 0:
        ratio = easy_hr / prev_easy_hr
        if ratio > 1.06:
            delta -= 0.20
        elif ratio > 1.03:
            delta -= 0.10
        elif ratio < 0.97:
            delta += 0.05

    if avg_hr_drift is not None:
        if avg_hr_drift > 8.0:
            delta -= 0.15
        elif avg_hr_drift > 5.0:
            delta -= 0.08

    return round(max(_LO, min(_HI, _BASE + delta)), 3)


def label(multiplier: float) -> str:
    if multiplier >= 1.05:
        return "Excellent"
    if multiplier >= 0.95:
        return "Good"
    if multiplier >= 0.85:
        return "Moderate"
    if multiplier >= 0.75:
        return "Low"
    return "Recovery needed"
