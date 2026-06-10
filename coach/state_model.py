"""
Banister fitness-fatigue (impulse-response) model.

  Fitness(t) = Fitness(t-1) × exp(−1/k1) + load(t)    k1 ≈ 42 days
  Fatigue(t) = Fatigue(t-1) × exp(−1/k2) + load(t)    k2 ≈  7 days
  Form(t)    = Fitness(t) − Fatigue(t)

Optimal race-day Form ≈ +8 to +25 (Banister et al. literature range).

Bounded constant update: k1/k2 may be nudged at most ±15 % per monthly cycle
against observed quality-session pace ratios, and only when ≥ 12 data points exist.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from typing import Optional

_K1_DEFAULT = 42.0          # fitness time constant (days)
_K2_DEFAULT = 7.0           # fatigue time constant (days)
_K1_BOUNDS  = (25.0, 60.0)
_K2_BOUNDS  = (4.0,  12.0)
_FORM_LO    = 8.0           # optimal race-day form — lower bound
_FORM_HI    = 25.0          # optimal race-day form — upper bound
_MAX_CHANGE = 0.15          # max fractional constant update per monthly cycle


@dataclass
class ModelState:
    fitness: float = 20.0
    fatigue: float = 10.0
    form: float    = 10.0
    k1_days: float = _K1_DEFAULT
    k2_days: float = _K2_DEFAULT
    last_updated: str = ""


def from_dict(d: dict) -> ModelState:
    return ModelState(
        fitness      = float(d.get("fitness",      20.0)),
        fatigue      = float(d.get("fatigue",      10.0)),
        form         = float(d.get("form",         10.0)),
        k1_days      = float(d.get("k1_days",  _K1_DEFAULT)),
        k2_days      = float(d.get("k2_days",  _K2_DEFAULT)),
        last_updated = str(d.get("last_updated", "")),
    )


def to_dict(m: ModelState) -> dict:
    return asdict(m)


def decay_only(state: ModelState, days: int) -> ModelState:
    """Public alias for pure exponential decay (no new load)."""
    return _decay(state, days)


def _decay(state: ModelState, days: int) -> ModelState:
    """Pure exponential decay for `days` elapsed days (no new load)."""
    if days <= 0:
        return state
    d1 = math.exp(-days / state.k1_days)
    d2 = math.exp(-days / state.k2_days)
    f  = state.fitness * d1
    fa = state.fatigue * d2
    return ModelState(
        fitness=round(f, 4), fatigue=round(fa, 4), form=round(f - fa, 4),
        k1_days=state.k1_days, k2_days=state.k2_days,
        last_updated=state.last_updated,
    )


def update(state: ModelState, load: float, days_elapsed: int = 1) -> ModelState:
    """
    Apply decay for `days_elapsed` days then add one load impulse.
    Handles multi-day gaps between check-ins correctly.
    """
    s = _decay(state, max(0, days_elapsed - 1))   # decay all-but-last day
    f  = s.fitness * math.exp(-1.0 / s.k1_days) + load
    fa = s.fatigue * math.exp(-1.0 / s.k2_days) + load
    return ModelState(
        fitness=round(f, 4), fatigue=round(fa, 4), form=round(f - fa, 4),
        k1_days=s.k1_days, k2_days=s.k2_days,
        last_updated=date.today().isoformat(),
    )


def apply_history(state: ModelState, history: list) -> ModelState:
    """
    Replay `history` records (sorted by date) through the model.
    Each record must have 'date' (ISO) and 'training_load' (float).
    Records without training_load are treated as rest days (pure decay).
    """
    records = sorted(
        [r for r in history if r.get("training_load") is not None],
        key=lambda r: r.get("date", ""),
    )
    if not records:
        return state

    prev_date: Optional[date] = None
    for r in records:
        try:
            cur = date.fromisoformat(r["date"])
        except (ValueError, KeyError):
            continue
        gap = (cur - prev_date).days if prev_date else 1
        state = update(state, float(r["training_load"]), days_elapsed=max(1, gap))
        prev_date = cur

    return state


def simulate_forward(
    state: ModelState,
    daily_loads: list,      # [(days_from_now: int, load: float), ...]
) -> list:
    """
    Simulate the model forward through a schedule of load events.
    Returns a list of ModelState snapshots, one per event.
    Events are processed in ascending days_from_now order.
    """
    snapshots: list[ModelState] = []
    current = state
    cursor  = 0

    for days_ahead, load in sorted(daily_loads, key=lambda x: x[0]):
        gap = days_ahead - cursor
        current = _decay(current, gap)
        f  = current.fitness * math.exp(-1.0 / current.k1_days) + load
        fa = current.fatigue * math.exp(-1.0 / current.k2_days) + load
        current = ModelState(
            fitness=round(f, 4), fatigue=round(fa, 4), form=round(f - fa, 4),
            k1_days=current.k1_days, k2_days=current.k2_days,
            last_updated=current.last_updated,
        )
        snapshots.append(current)
        cursor = days_ahead

    return snapshots


def score_race_day_form(state: ModelState) -> float:
    """
    Score 0–100 for projected race-day Form.
    Peak band [8, 25] → ≥ 90; steep penalty below (still fatigued), softer above (detraining).
    """
    form = state.form
    mid  = (_FORM_LO + _FORM_HI) / 2
    if _FORM_LO <= form <= _FORM_HI:
        return max(60.0, 100.0 - abs(form - mid) * 2.5)
    if form < _FORM_LO:
        return max(0.0, 60.0 + (form - _FORM_LO) * 6.0)   # steep fall below 8
    return max(0.0, 60.0 - (form - _FORM_HI) * 3.0)       # gradual fall above 25


def bounded_update_constants(
    state: ModelState,
    pace_ratios: list,   # [actual_pace_s / target_pace_s] from quality sessions
) -> ModelState:
    """
    Nudge k1 (and optionally k2) based on recent quality-session pace ratios.
    Only fires with ≥ 12 data points; clamps to ±_MAX_CHANGE per call.
    """
    if len(pace_ratios) < 12:
        return state

    mean_ratio = sum(pace_ratios) / len(pace_ratios)
    # ratio < 1 → athlete running faster than target → fitness higher than model thinks → k1 underestimated
    delta_pct = max(-_MAX_CHANGE, min(_MAX_CHANGE, (1.0 - mean_ratio) * 0.5))

    new_k1 = max(_K1_BOUNDS[0], min(_K1_BOUNDS[1], state.k1_days * (1 + delta_pct)))

    new_k2 = state.k2_days
    if len(pace_ratios) >= 20:
        n = len(pace_ratios)
        mu = mean_ratio
        cv = (sum((r - mu) ** 2 for r in pace_ratios) / n) ** 0.5 / (mu or 1.0)
        if cv < 0.15 and mean_ratio < 0.98:
            new_k2 = max(_K2_BOUNDS[0], min(_K2_BOUNDS[1], state.k2_days * 0.95))

    return ModelState(
        fitness=state.fitness, fatigue=state.fatigue, form=state.form,
        k1_days=round(new_k1, 2), k2_days=round(new_k2, 2),
        last_updated=state.last_updated,
    )
