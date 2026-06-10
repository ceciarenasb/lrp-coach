"""Tests for coach/state_model.py — Banister impulse-response model."""

import math
import pytest
from coach.state_model import (
    ModelState, from_dict, to_dict,
    update, decay_only, simulate_forward,
    score_race_day_form, bounded_update_constants,
    _K1_DEFAULT, _K2_DEFAULT, _K1_BOUNDS, _K2_BOUNDS, _MAX_CHANGE,
)


def _fresh():
    return ModelState(fitness=40.0, fatigue=35.0, form=5.0)


# ── Round-trip ─────────────────────────────────────────────────────────────

def test_from_to_dict_round_trip():
    s = _fresh()
    assert from_dict(to_dict(s)).fitness == pytest.approx(s.fitness)
    assert from_dict(to_dict(s)).k1_days == pytest.approx(s.k1_days)

def test_from_dict_uses_defaults_for_missing_keys():
    s = from_dict({})
    assert s.k1_days == pytest.approx(_K1_DEFAULT)
    assert s.k2_days == pytest.approx(_K2_DEFAULT)


# ── Update / decay ─────────────────────────────────────────────────────────

def test_load_impulse_raises_fitness_and_fatigue():
    s = _fresh()
    s2 = update(s, load=50.0, days_elapsed=1)
    assert s2.fitness > s.fitness
    assert s2.fatigue > s.fatigue

def test_zero_load_is_pure_decay():
    s = _fresh()
    s2 = update(s, load=0.0, days_elapsed=1)
    assert s2.fitness < s.fitness
    assert s2.fatigue < s.fatigue

def test_form_equals_fitness_minus_fatigue():
    s = update(_fresh(), load=30.0)
    assert s.form == pytest.approx(s.fitness - s.fatigue, abs=0.01)

def test_multi_day_gap_decays_more_than_one_day():
    s = _fresh()
    one  = update(s, load=0.0, days_elapsed=1)
    five = update(s, load=0.0, days_elapsed=5)
    assert five.fitness < one.fitness

def test_fatigue_decays_faster_than_fitness():
    """k2 < k1 → fatigue half-life is shorter."""
    s   = ModelState(fitness=50.0, fatigue=50.0, form=0.0)
    s7  = update(s, load=0.0, days_elapsed=7)
    # After 7 days: fitness decays by exp(-7/42) ≈ 0.846; fatigue by exp(-7/7) ≈ 0.368
    assert s7.fatigue / s.fatigue < s7.fitness / s.fitness


# ── simulate_forward ───────────────────────────────────────────────────────

def test_simulate_returns_one_snapshot_per_event():
    s = _fresh()
    snaps = simulate_forward(s, [(7, 20.0), (14, 20.0), (21, 20.0)])
    assert len(snaps) == 3

def test_simulate_forward_fitness_grows_with_consistent_load():
    s = ModelState(fitness=10.0, fatigue=5.0, form=5.0)
    loads  = [(d * 7, 50.0) for d in range(1, 9)]   # 8 weeks of 50-unit loads
    snaps  = simulate_forward(s, loads)
    assert snaps[-1].fitness > s.fitness

def test_simulate_forward_order_independent():
    """Events given out of order should produce the same result as sorted."""
    s = _fresh()
    ordered   = simulate_forward(s, [(7, 30.0), (14, 20.0)])
    unordered = simulate_forward(s, [(14, 20.0), (7, 30.0)])
    assert ordered[-1].fitness == pytest.approx(unordered[-1].fitness, abs=0.01)


# ── score_race_day_form ────────────────────────────────────────────────────

def test_optimal_form_scores_highest():
    optimal   = score_race_day_form(ModelState(fitness=50.0, fatigue=33.0, form=17.0))
    too_tired = score_race_day_form(ModelState(fitness=50.0, fatigue=60.0, form=-10.0))
    detraining= score_race_day_form(ModelState(fitness=20.0, fatigue=0.0,  form=20.0))
    assert optimal >= too_tired
    assert optimal >= detraining

def test_very_negative_form_scores_zero_or_close():
    s = ModelState(fitness=10.0, fatigue=50.0, form=-40.0)
    assert score_race_day_form(s) == pytest.approx(0.0, abs=5.0)


# ── bounded_update_constants ───────────────────────────────────────────────

def test_constants_unchanged_below_threshold():
    s = _fresh()
    # Fewer than 12 points → no update
    updated = bounded_update_constants(s, pace_ratios=[1.0] * 8)
    assert updated.k1_days == pytest.approx(s.k1_days)

def test_k1_increases_when_athlete_faster_than_target():
    """pace_ratio < 1 → athlete running faster → k1 underestimated → increase k1."""
    s       = _fresh()
    ratios  = [0.95] * 15          # consistently 5% faster
    updated = bounded_update_constants(s, ratios)
    assert updated.k1_days > s.k1_days

def test_k1_stays_within_bounds():
    s = ModelState(fitness=40.0, fatigue=35.0, form=5.0, k1_days=_K1_BOUNDS[1] - 1)
    updated = bounded_update_constants(s, [0.80] * 20)   # very fast — push limit
    assert updated.k1_days <= _K1_BOUNDS[1]

def test_update_clamped_to_max_change():
    s       = _fresh()
    ratios  = [0.50] * 20          # extreme — would push k1 far
    updated = bounded_update_constants(s, ratios)
    max_delta = s.k1_days * _MAX_CHANGE
    assert abs(updated.k1_days - s.k1_days) <= max_delta + 0.01
