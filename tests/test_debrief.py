"""Tests for coach/debrief.py — pure logic, no Gradio."""

import pytest
from coach.debrief import (
    RATING_EXTRA, RATING_MISSED, RATING_MODERATE, RATING_PARTIAL,
    RATING_REST, RATING_SUCCESSFUL,
    SessionDebrief, compliance, match_week, rate_session,
)

# ── Shared fixtures ────────────────────────────────────────────────────────

ZONES = {
    "easy_lo": 420,       # 7:00/km  (slow)
    "easy_hi": 380,       # 6:20/km  (fast)
    "marathon": 360,      # 6:00/km
    "threshold": 330,     # 5:30/km
    "cv_interval": 310,   # 5:10/km
    "interval": 295,      # 4:55/km
    "rep": 280,
}

def _planned(date="2026-06-09", weekday=0, s_type="Easy", km=10.0, desc="Easy run"):
    return {"date": date, "weekday": weekday, "session_type": s_type,
            "distance_km": km, "description": desc}

def _actual(date="2026-06-09", km=10.0, pace=395, hr=140, drift=3.0,
            activity_type="Run", duration_s=3960):
    return {"date": date, "activity_type": activity_type, "distance_km": km,
            "avg_pace_s": pace, "avg_hr": hr, "hr_drift_pct": drift,
            "duration_s": duration_s}


# ── match_week ─────────────────────────────────────────────────────────────

class TestMatchWeek:

    def test_exact_date_match(self):
        planned = [_planned("2026-06-09", s_type="Easy")]
        hist    = [_actual("2026-06-09")]
        pairs   = match_week(planned, hist)
        assert len(pairs) == 1
        p, a = pairs[0]
        assert p is not None
        assert a is not None

    def test_category_mismatch_not_matched(self):
        planned = [_planned("2026-06-09", s_type="Easy")]
        hist    = [_actual("2026-06-09", activity_type="Strength")]
        pairs   = match_week(planned, hist)
        p, a = pairs[0]
        assert a is None   # strength shouldn't match a run session

    def test_two_same_day_runs_closest_distance_wins(self):
        planned = [_planned("2026-06-09", s_type="Easy", km=10.0)]
        hist    = [
            _actual("2026-06-09", km=7.0),
            _actual("2026-06-09", km=9.8),
        ]
        pairs   = match_week(planned, hist)
        p, a = pairs[0]
        assert a["distance_km"] == 9.8   # closest to 10.0

    def test_each_activity_matched_only_once(self):
        planned = [
            _planned("2026-06-09", s_type="Easy",    km=10.0),
            _planned("2026-06-10", s_type="Easy",    km=8.0),
        ]
        hist = [_actual("2026-06-09", km=10.0)]
        pairs = match_week(planned, hist)
        matched_actuals = [a for _, a in pairs if a is not None]
        assert len(matched_actuals) == 1   # only one activity, used once

    def test_unmatched_run_becomes_extra(self):
        planned = [_planned("2026-06-09", s_type="Rest")]
        hist    = [_actual("2026-06-09", km=5.0)]
        pairs   = match_week(planned, hist)
        extra_pairs = [(p, a) for p, a in pairs if p is None]
        assert len(extra_pairs) == 1
        assert extra_pairs[0][1]["distance_km"] == 5.0

    def test_rest_day_untouched(self):
        planned = [_planned("2026-06-09", s_type="Rest")]
        hist    = []
        pairs   = match_week(planned, hist)
        p, a = pairs[0]
        assert p["session_type"] == "Rest"
        assert a is None

    def test_empty_week_returns_empty(self):
        assert match_week([], []) == []


# ── rate_session ───────────────────────────────────────────────────────────

class TestRateSession:

    def test_completed_easy_in_window_successful(self):
        # pace 395 is within easy window (380*0.96=364 fast, 420*1.10=462 slow)
        d = rate_session(_planned(s_type="Easy", km=10), _actual(km=10, pace=400), ZONES)
        assert d.rating == RATING_SUCCESSFUL
        assert any("Distance" in s for s in d.strengths)
        assert any("Pace" in s for s in d.strengths)

    def test_easy_run_too_fast_adds_weakness(self):
        # pace 340 is faster than easy_hi * 0.96 = 364 — weakness present even if still Successful
        d = rate_session(_planned(s_type="Easy", km=10), _actual(km=10, pace=340), ZONES)
        assert any("easy" in w.lower() or "fast" in w.lower() for w in d.weaknesses)
        assert d.score < 100   # score is penalised even if rating stays Successful

    def test_missed_quality_missed_rating_and_quality_advice(self):
        d = rate_session(_planned(s_type="Tempo", km=12), None, ZONES)
        assert d.rating == RATING_MISSED
        assert d.score == 0
        assert "quality" in d.advice.lower()

    def test_missed_long_run_specific_advice(self):
        d = rate_session(_planned(s_type="Long Run", km=25), None, ZONES)
        assert d.rating == RATING_MISSED
        assert "cornerstone" in d.advice.lower() or "long run" in d.advice.lower()

    def test_short_run_60_to_90_pct_moderate_or_partial(self):
        # 7 km of 12 km planned = 58% → partial
        d = rate_session(_planned(s_type="Easy", km=12), _actual(km=7, pace=400), ZONES)
        assert d.rating in (RATING_MODERATE, RATING_PARTIAL)
        assert any("short" in w.lower() or "cut" in w.lower() for w in d.weaknesses)

    def test_high_hr_drift_adds_weakness(self):
        d = rate_session(_planned(s_type="Easy", km=10), _actual(km=10, pace=400, drift=10.0), ZONES)
        assert any("drift" in w.lower() for w in d.weaknesses)

    def test_rpe_8_on_easy_adds_weakness(self):
        d = rate_session(_planned(s_type="Easy", km=10), _actual(km=10, pace=400), ZONES, rpe=8)
        assert any("rpe" in w.lower() or "effort" in w.lower() for w in d.weaknesses)

    def test_rpe_3_on_quality_adds_strength(self):
        # pace 315 is within SVC window (310*0.92=285 fast, 310*1.28=396 slow)
        d = rate_session(
            _planned(s_type="SVC Intervals", km=12),
            _actual(km=12, pace=315),
            ZONES,
            rpe=3,
        )
        assert any("rpe" in s.lower() or "fitness" in s.lower() for s in d.strengths)

    def test_rest_day_returns_rest_rating(self):
        d = rate_session(_planned(s_type="Rest", km=0), None, ZONES)
        assert d.rating == RATING_REST
        assert d.score == 100

    def test_extra_activity_rating(self):
        d = rate_session(None, _actual(km=5), ZONES)
        assert d.rating == RATING_EXTRA
        assert any("extra" in s.lower() for s in d.strengths)

    def test_strength_completed(self):
        d = rate_session(
            _planned(s_type="Strength", km=0),
            _actual(activity_type="Strength", km=0, pace=None, drift=None),
            ZONES,
        )
        assert d.rating == RATING_SUCCESSFUL

    def test_strength_missed(self):
        d = rate_session(_planned(s_type="Strength", km=0), None, ZONES)
        assert d.rating == RATING_MISSED


# ── compliance ─────────────────────────────────────────────────────────────

class TestCompliance:

    def _make_debriefs(self, ratings, planned_kms=None, actual_kms=None):
        kms  = planned_kms or [10] * len(ratings)
        akms = actual_kms  or [10] * len(ratings)
        out  = []
        for i, (r, pkm, akm) in enumerate(zip(ratings, kms, akms)):
            a = {"distance_km": akm} if r != RATING_MISSED else None
            out.append(SessionDebrief(
                date=f"2026-06-0{i+1}", weekday=i,
                session_type="Easy", description="",
                planned_km=pkm, actual=a,
                rating=r, score=80 if r == RATING_SUCCESSFUL else 60,
            ))
        return out

    def test_all_successful_100_pct(self):
        debriefs = self._make_debriefs([RATING_SUCCESSFUL] * 4)
        c = compliance(debriefs)
        assert c["pct_sessions"] == 100
        assert c["planned"] == 4
        assert c["successful"] == 4

    def test_one_missed_reduces_pct(self):
        debriefs = self._make_debriefs(
            [RATING_SUCCESSFUL, RATING_SUCCESSFUL, RATING_SUCCESSFUL, RATING_MISSED],
        )
        c = compliance(debriefs)
        assert c["pct_sessions"] < 100
        assert c["missed"] == 1

    def test_km_percentages(self):
        debriefs = self._make_debriefs(
            [RATING_SUCCESSFUL, RATING_SUCCESSFUL],
            planned_kms=[10, 10],
            actual_kms=[8, 9],
        )
        c = compliance(debriefs)
        assert c["planned_km"] == 20.0
        assert c["actual_km"] == 17.0
        assert c["pct_km"] == 85

    def test_rest_days_excluded_from_denominator(self):
        debriefs = [
            SessionDebrief("2026-06-09", 0, "Easy", "", 10, {"distance_km": 10},
                           RATING_SUCCESSFUL, 90),
            SessionDebrief("2026-06-10", 1, "Rest", "", 0, None,
                           RATING_REST, 100),
        ]
        c = compliance(debriefs)
        assert c["planned"] == 1   # Rest excluded

    def test_mutated_plan_distances_used(self):
        # compliance should use whatever planned_km is in the debrief
        # (which reflects the current stored plan, not the original)
        d = SessionDebrief("2026-06-09", 0, "Easy", "", planned_km=15,
                           actual={"distance_km": 15}, rating=RATING_SUCCESSFUL, score=90)
        c = compliance([d])
        assert c["planned_km"] == 15.0

    def test_empty_week_no_division_by_zero(self):
        c = compliance([])
        assert c["pct_sessions"] == 0
        assert c["planned_km"] == 0
        assert c["actual_km"] == 0
