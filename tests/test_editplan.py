"""Tests for coach/editplan.py — move_session and validate_week."""

from __future__ import annotations

from typing import Optional

import pytest
from coach.editplan import move_session, validate_week

# ── Helpers ────────────────────────────────────────────────────────────────


def _day(date: str, weekday: int, session_type: str, km: float = 10.0,
         desc: str = "", targets: Optional[dict] = None):
    return {
        "date": date, "weekday": weekday,
        "session_type": session_type, "description": desc,
        "distance_km": km, "targets": targets or {},
    }


def _week(week_num: int, phase: str, days: list, target_km: float = 50.0):
    return {"week_num": week_num, "phase": phase, "focus": "",
            "target_km": target_km, "days": days}


def _plan_one_week():
    """Mon Easy, Tue Tempo, Wed Rest, Thu Easy, Fri SVC Intervals, Sat Club Run, Sun Rest"""
    return [_week(1, "Base", [
        _day("2026-06-15", 0, "Easy",           km=8),
        _day("2026-06-16", 1, "Tempo",          km=12),
        _day("2026-06-17", 2, "Rest",            km=0),
        _day("2026-06-18", 3, "Easy",           km=8),
        _day("2026-06-19", 4, "SVC Intervals",  km=10),
        _day("2026-06-20", 5, "Club Run (LRP)", km=10),
        _day("2026-06-21", 6, "Rest",            km=0),
    ])]


def _plan_two_weeks():
    w1 = _plan_one_week()[0]
    w2 = _week(2, "Build", [
        _day("2026-06-22", 0, "Easy",           km=9),
        _day("2026-06-23", 1, "Rest",            km=0),
        _day("2026-06-24", 2, "Tempo",          km=12),
        _day("2026-06-25", 3, "Easy",           km=9),
        _day("2026-06-26", 4, "Rest",            km=0),
        _day("2026-06-27", 5, "Long Run",       km=25),
        _day("2026-06-28", 6, "Rest",            km=0),
    ])
    return [w1, w2]


# ── move_session ───────────────────────────────────────────────────────────


class TestMoveSession:

    def test_swap_within_week(self):
        plan = _plan_one_week()
        new_plan, warnings = move_session(plan, "2026-06-15", "2026-06-17")
        days = new_plan[0]["days"]
        assert days[0]["session_type"] == "Rest"     # Mon gets Rest
        assert days[2]["session_type"] == "Easy"     # Wed gets Easy

    def test_move_to_rest_day(self):
        plan = _plan_one_week()
        new_plan, _ = move_session(plan, "2026-06-16", "2026-06-17")  # Tempo ↔ Rest
        days = new_plan[0]["days"]
        assert days[1]["session_type"] == "Rest"     # Tue → Rest
        assert days[2]["session_type"] == "Tempo"    # Wed → Tempo

    def test_cross_week_move(self):
        plan = _plan_two_weeks()
        # Move Tempo (w1 Tue) to Rest (w2 Tue)
        new_plan, _ = move_session(plan, "2026-06-16", "2026-06-23")
        assert new_plan[0]["days"][1]["session_type"] == "Rest"
        assert new_plan[1]["days"][1]["session_type"] == "Tempo"

    def test_same_day_rejected(self):
        plan = _plan_one_week()
        new_plan, warnings = move_session(plan, "2026-06-15", "2026-06-15")
        assert new_plan[0]["days"][0]["session_type"] == "Easy"   # unchanged
        assert warnings

    def test_original_plan_not_mutated(self):
        plan = _plan_one_week()
        original_type = plan[0]["days"][0]["session_type"]
        move_session(plan, "2026-06-15", "2026-06-17")
        assert plan[0]["days"][0]["session_type"] == original_type

    def test_inverse_move_restores_original(self):
        plan = _plan_one_week()
        moved, _ = move_session(plan, "2026-06-15", "2026-06-17")
        restored, _ = move_session(moved, "2026-06-17", "2026-06-15")
        for orig_day, rest_day in zip(plan[0]["days"], restored[0]["days"]):
            assert orig_day["session_type"] == rest_day["session_type"]

    def test_date_not_found_returns_warning(self):
        plan = _plan_one_week()
        _, warnings = move_session(plan, "2099-01-01", "2026-06-15")
        assert warnings

    def test_distance_km_and_targets_swap_too(self):
        plan = _plan_one_week()
        plan[0]["days"][0]["distance_km"] = 8.0
        plan[0]["days"][0]["targets"] = {"pace": "5:30/km"}
        plan[0]["days"][2]["distance_km"] = 0.0
        plan[0]["days"][2]["targets"] = {}
        new_plan, _ = move_session(plan, "2026-06-15", "2026-06-17")
        assert new_plan[0]["days"][2]["distance_km"] == 8.0
        assert new_plan[0]["days"][2]["targets"] == {"pace": "5:30/km"}
        assert new_plan[0]["days"][0]["distance_km"] == 0.0


# ── validate_week ──────────────────────────────────────────────────────────


class TestValidateWeek:

    def test_valid_week_no_warnings(self):
        # Tempo Mon, rest Tue, SVC Thu, Club Run Sat — all ≥2 days apart
        days = [
            _day("2026-06-15", 0, "Tempo"),
            _day("2026-06-16", 1, "Easy"),
            _day("2026-06-17", 2, "Rest"),
            _day("2026-06-18", 3, "SVC Intervals"),
            _day("2026-06-19", 4, "Easy"),
            _day("2026-06-20", 5, "Club Run (LRP)"),
            _day("2026-06-21", 6, "Rest"),
        ]
        assert validate_week(days) == []

    def test_adjacent_quality_sessions_warns(self):
        days = [
            _day("2026-06-15", 0, "Tempo"),
            _day("2026-06-16", 1, "SVC Intervals"),  # adjacent → warn
        ]
        warnings = validate_week(days)
        assert warnings

    def test_two_days_apart_ok(self):
        days = [
            _day("2026-06-15", 0, "Tempo"),
            _day("2026-06-17", 2, "SVC Intervals"),  # 2 days gap → OK
        ]
        assert validate_week(days) == []

    def test_rest_days_not_counted_as_hard(self):
        days = [
            _day("2026-06-15", 0, "Rest"),
            _day("2026-06-16", 1, "Rest"),
        ]
        assert validate_week(days) == []

    def test_quality_adjacent_to_long_run_warns(self):
        days = [
            _day("2026-06-15", 0, "Long Run"),
            _day("2026-06-16", 1, "Tempo"),
        ]
        warnings = validate_week(days)
        assert warnings

    def test_empty_week_no_warnings(self):
        assert validate_week([]) == []
