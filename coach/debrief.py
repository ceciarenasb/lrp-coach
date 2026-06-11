"""Session-by-session debrief: matching, rating, compliance, LLM context."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Mirrors coach/plan.py constants (no circular import)
_REST        = "Rest"
_RECOVERY    = "Recovery"
_EASY        = "Easy"
_MEDIUM_LONG = "Medium-Long"
_LONG        = "Long Run"
_TEMPO       = "Tempo"
_SVC         = "SVC Intervals"
_MP_RUN      = "Marathon Pace"
_PROGRESSION = "Progression Run"
_STRENGTH    = "Strength"
_CYCLING     = "Cycling / Zwift"
_CLUB_RUN    = "Club Run (LRP)"

_EASY_TYPES    = {_EASY, _RECOVERY, _CLUB_RUN}
_QUALITY_TYPES = {_TEMPO, _SVC, _MP_RUN, _PROGRESSION}
_LONG_TYPES    = {_LONG, _MEDIUM_LONG}
_KEY_TYPES     = _QUALITY_TYPES | _LONG_TYPES

RATING_SUCCESSFUL = "Successful"
RATING_MODERATE   = "Moderately successful"
RATING_PARTIAL    = "Partially completed"
RATING_MISSED     = "Missed"
RATING_EXTRA      = "Extra (unplanned)"
RATING_REST       = "Rest day"


@dataclass
class SessionDebrief:
    date: str
    weekday: int
    session_type: str
    description: str
    planned_km: float
    actual: Optional[dict]
    rating: str
    score: int
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    advice: str = ""
    rpe: Optional[float] = None
    comment: str = ""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _activity_category(activity_type: str) -> str:
    at = (activity_type or "").lower()
    if "cycling" in at:
        return "cycling"
    if at in ("strength", "yoga", "gym", "weights"):
        return "strength"
    return "run"


def _plan_category(session_type: str) -> str:
    if session_type == _CYCLING:
        return "cycling"
    if session_type == _STRENGTH:
        return "strength"
    if session_type == _REST:
        return "rest"
    return "run"


def _pace_window(session_type: str, zones: dict) -> Optional[tuple]:
    """Return (fast_bound_s, slow_bound_s). None for non-pace sessions."""
    el = zones.get("easy_lo") or 0
    eh = zones.get("easy_hi") or 0
    mp = zones.get("marathon") or 0
    tp = zones.get("threshold") or 0
    cv = zones.get("cv_interval") or 0

    if not el or not eh:
        return None

    if session_type in (_EASY, _CLUB_RUN):
        return int(eh * 0.96), int(el * 1.10)
    if session_type == _RECOVERY:
        return int(eh), int(el * 1.18)
    if session_type in _LONG_TYPES:
        return int(mp) if mp else int(el * 0.95), int(el * 1.10)
    if session_type == _TEMPO:
        return int(tp * 0.95), int(tp * 1.20)
    if session_type == _SVC:
        return int(cv * 0.92), int(cv * 1.28)
    if session_type == _MP_RUN:
        return int(mp * 0.96), int(mp * 1.16)
    if session_type == _PROGRESSION:
        return int(mp * 0.98) if mp else int(el * 0.90), int(el * 1.08)
    return None


def _fmt_pace(sec_per_km) -> str:
    if not sec_per_km or sec_per_km <= 0:
        return "—"
    m, s = divmod(int(sec_per_km), 60)
    return f"{m}:{s:02d}/km"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_week(week_days: list, history: list) -> list:
    """
    Pair each planned day with the best matching activity from history.
    Returns list of (planned_dict_or_None, actual_dict_or_None).
    Unmatched history entries are appended as (None, actual).
    """
    if not week_days:
        return []

    dates = [d["date"] for d in week_days if d.get("date")]
    if not dates:
        return []
    week_start, week_end = min(dates), max(dates)

    week_hist = [
        a for a in history
        if week_start <= (a.get("date") or "") <= week_end
    ]

    used: set = set()
    result = []

    for day in week_days:
        s_type   = day.get("session_type", _REST)
        plan_cat = _plan_category(s_type)

        if plan_cat == "rest":
            result.append((day, None))
            continue

        planned_km = day.get("distance_km") or 0
        day_date   = day.get("date", "")

        candidates = [
            (i, a) for i, a in enumerate(week_hist)
            if i not in used
            and (a.get("date") or "") == day_date
            and _activity_category(a.get("activity_type", "")) == plan_cat
        ]

        if not candidates:
            result.append((day, None))
            continue

        best_i, best_a = min(
            candidates,
            key=lambda x: abs((x[1].get("distance_km") or 0) - planned_km),
        )
        used.add(best_i)
        result.append((day, best_a))

    # Unmatched history → Extra
    for i, a in enumerate(week_hist):
        if i not in used and a.get("activity_type"):
            result.append((None, a))

    return result


def rate_session(
    planned: Optional[dict],
    actual: Optional[dict],
    zones: dict,
    rpe: Optional[float] = None,
    comment: str = "",
) -> SessionDebrief:
    """Score one session. zones is the stored dict with int sec/km values."""

    # ── Extra (unplanned) ──────────────────────────────────────────────────
    if planned is None and actual is not None:
        km  = actual.get("distance_km") or 0
        dur = (actual.get("duration_s") or 0) // 60
        desc = f"{km:.1f} km" if km else f"{dur} min"
        return SessionDebrief(
            date=actual.get("date", ""),
            weekday=-1,
            session_type=RATING_EXTRA,
            description=f"{actual.get('activity_type', 'Activity')} — {desc}",
            planned_km=0.0,
            actual=actual,
            rating=RATING_EXTRA,
            score=0,
            strengths=["Extra training added to the week ✓"],
            weaknesses=[],
            advice="Make sure extras don't compromise the next day's quality session.",
            rpe=rpe,
            comment=comment,
        )

    s_type    = planned.get("session_type", _REST)
    planned_km = planned.get("distance_km") or 0
    plan_cat  = _plan_category(s_type)

    # ── Rest day ───────────────────────────────────────────────────────────
    if plan_cat == "rest":
        return SessionDebrief(
            date=planned.get("date", ""),
            weekday=planned.get("weekday", -1),
            session_type=_REST,
            description=planned.get("description", "Rest"),
            planned_km=0.0,
            actual=actual,
            rating=RATING_REST,
            score=100,
            strengths=[],
            weaknesses=[],
            advice="",
            rpe=rpe,
            comment=comment,
        )

    # ── Strength / Cycling ─────────────────────────────────────────────────
    if plan_cat in ("strength", "cycling"):
        if actual is None:
            return SessionDebrief(
                date=planned.get("date", ""),
                weekday=planned.get("weekday", -1),
                session_type=s_type,
                description=planned.get("description", ""),
                planned_km=planned_km,
                actual=None,
                rating=RATING_MISSED,
                score=0,
                strengths=[],
                weaknesses=["Session not completed"],
                advice="Try to reschedule — consistency in cross-training builds durability.",
                rpe=rpe,
                comment=comment,
            )
        dur_min = (actual.get("duration_s") or 0) // 60
        return SessionDebrief(
            date=planned.get("date", ""),
            weekday=planned.get("weekday", -1),
            session_type=s_type,
            description=planned.get("description", ""),
            planned_km=planned_km,
            actual=actual,
            rating=RATING_SUCCESSFUL,
            score=90,
            strengths=[f"Completed — {dur_min} min ✓"],
            weaknesses=[],
            advice="Bank it and recover well.",
            rpe=rpe,
            comment=comment,
        )

    # ── Missed run ─────────────────────────────────────────────────────────
    if actual is None:
        if s_type in _QUALITY_TYPES:
            advice = ("Quality sessions drive the biggest gains — "
                      "sacrifice an easy day instead next time.")
            weakness = f"Missed quality session ({s_type})"
        elif s_type == _LONG:
            advice = ("The long run is the cornerstone of marathon prep — "
                      "move it rather than skip it.")
            weakness = "Missed long run"
        else:
            advice = "Consistency builds fitness — even a shorter version is better than nothing."
            weakness = f"Missed {s_type} session"
        return SessionDebrief(
            date=planned.get("date", ""),
            weekday=planned.get("weekday", -1),
            session_type=s_type,
            description=planned.get("description", ""),
            planned_km=planned_km,
            actual=None,
            rating=RATING_MISSED,
            score=0,
            strengths=[],
            weaknesses=[weakness],
            advice=advice,
            rpe=rpe,
            comment=comment,
        )

    # ── Completed run ──────────────────────────────────────────────────────
    score      = 100
    strengths: list = []
    weaknesses: list = []
    advice     = "Bank it and recover well."

    actual_km  = actual.get("distance_km") or 0
    actual_pace = actual.get("avg_pace_s")
    hr_drift   = actual.get("hr_drift_pct")

    # Distance
    if planned_km > 0:
        ratio = actual_km / planned_km
        if ratio >= 0.90:
            strengths.append("Distance done ✓")
        elif ratio >= 0.60:
            score -= 20
            weaknesses.append(f"Cut short by {round((1 - ratio) * 100)}%")
        else:
            score -= 40
            weaknesses.append(
                f"Significantly short — {actual_km:.1f} of {planned_km:.0f} km "
                f"({round((1 - ratio) * 100)}% cut)"
            )

    # Pace
    window = _pace_window(s_type, zones)
    if window and actual_pace:
        fast_b, slow_b = window
        in_window = fast_b <= actual_pace <= slow_b
        if in_window:
            strengths.append("Pace on target ✓")
        elif actual_pace < fast_b:
            if s_type in _EASY_TYPES:
                score -= 15
                weaknesses.append(
                    f"Too fast ({_fmt_pace(actual_pace)} vs {_fmt_pace(fast_b)}–{_fmt_pace(slow_b)}) "
                    "— easy days must stay easy; same aerobic gain, fresher for quality days"
                )
                advice = "Slow down on easy days — the easy/hard contrast is what drives adaptation."
            elif s_type in _QUALITY_TYPES:
                strengths.append(
                    f"Pace ahead of target ({_fmt_pace(actual_pace)}) — fitness ahead of plan ✓"
                )
            else:
                strengths.append("Pace on target ✓")
        else:
            score -= 15
            weaknesses.append(
                f"Pace below target ({_fmt_pace(actual_pace)} vs {_fmt_pace(fast_b)}–{_fmt_pace(slow_b)})"
            )
            if s_type in _QUALITY_TYPES:
                advice = ("If target pace feels impossible, shorten reps "
                          "rather than grinding slow ones.")

    # HR drift
    if hr_drift is not None:
        if hr_drift <= 5:
            strengths.append("HR drift normal ✓")
        elif hr_drift <= 8:
            score -= 5
            weaknesses.append(f"HR drift slightly elevated ({hr_drift:.1f}%) — mild fatigue signal")
        else:
            score -= 15
            weaknesses.append(f"HR drift high ({hr_drift:.1f}%) — fatigue or dehydration signal")
            if advice == "Bank it and recover well.":
                advice = ("Prioritise sleep and fuelling — hydration and nutrition "
                          "are key recovery levers.")

    # RPE cross-check
    if rpe is not None:
        if rpe >= 7 and s_type in _EASY_TYPES:
            score -= 10
            weaknesses.append(
                f"Effort felt hard (RPE {int(rpe)}) for an easy session — first fatigue flag"
            )
        elif rpe <= 4 and s_type in _QUALITY_TYPES:
            strengths.append(f"Felt easy for a quality session (RPE {int(rpe)}) — fitness ahead of plan ✓")

    score = max(0, min(100, score))
    if score >= 80:
        rating = RATING_SUCCESSFUL
    elif score >= 55:
        rating = RATING_MODERATE
    else:
        rating = RATING_PARTIAL

    # Fill advice if still generic despite weaknesses
    if advice == "Bank it and recover well." and weaknesses:
        advice = "Keep the pattern — consistency compounds over weeks."

    return SessionDebrief(
        date=planned.get("date", ""),
        weekday=planned.get("weekday", -1),
        session_type=s_type,
        description=planned.get("description", ""),
        planned_km=planned_km,
        actual=actual,
        rating=rating,
        score=score,
        strengths=strengths,
        weaknesses=weaknesses,
        advice=advice,
        rpe=rpe,
        comment=comment,
    )


def compliance(debriefs: list) -> dict:
    """Aggregate compliance stats from a list of SessionDebriefs."""
    planned = [d for d in debriefs if d.rating not in (RATING_REST, RATING_EXTRA)]
    extras  = [d for d in debriefs if d.rating == RATING_EXTRA]

    n_planned    = len(planned)
    n_successful = sum(1 for d in planned if d.rating == RATING_SUCCESSFUL)
    n_moderate   = sum(1 for d in planned if d.rating == RATING_MODERATE)
    n_partial    = sum(1 for d in planned if d.rating == RATING_PARTIAL)
    n_missed     = sum(1 for d in planned if d.rating == RATING_MISSED)

    pct_sessions = 0
    if n_planned > 0:
        pct_sessions = round(
            100 * (n_successful + 0.75 * n_moderate + 0.4 * n_partial) / n_planned
        )

    planned_km = sum(d.planned_km for d in planned)
    actual_km  = sum(
        (d.actual.get("distance_km") or 0)
        for d in debriefs
        if d.actual and d.rating not in (RATING_REST,)
    )
    pct_km = round(100 * actual_km / planned_km) if planned_km > 0 else 0

    key_planned = [d for d in planned if d.session_type in _KEY_TYPES]
    key_done    = [d for d in key_planned if d.rating in (RATING_SUCCESSFUL, RATING_MODERATE)]

    return {
        "planned":      n_planned,
        "successful":   n_successful,
        "moderate":     n_moderate,
        "partial":      n_partial,
        "missed":       n_missed,
        "extra":        len(extras),
        "planned_km":   round(planned_km, 1),
        "actual_km":    round(actual_km, 1),
        "pct_sessions": pct_sessions,
        "pct_km":       pct_km,
        "key_planned":  len(key_planned),
        "key_done":     len(key_done),
    }


def build_llm_context(
    name: str,
    phase: str,
    debriefs: list,
    comp: dict,
    week_label: str,
) -> str:
    lines = []
    for d in debriefs:
        if d.rating == RATING_REST:
            continue
        km_str = ""
        if d.actual:
            km  = d.actual.get("distance_km") or 0
            pace = d.actual.get("avg_pace_s")
            km_str = f" → {km:.1f} km"
            if pace:
                km_str += f" @ {_fmt_pace(pace)}"
        rpe_str     = f" [RPE {int(d.rpe)}]" if d.rpe else ""
        comment_str = f' — "{d.comment}"' if d.comment else ""
        lines.append(
            f"  {d.date} · {d.session_type} (plan {d.planned_km:.0f} km)"
            f"{km_str} → {d.rating}{rpe_str}{comment_str}"
        )

    sessions_block = "\n".join(lines) if lines else "  (no sessions recorded)"

    return (
        f"ATHLETE: {name}\n"
        f"WEEK: {week_label}  PHASE: {phase}\n"
        f"\nSESSION LOG:\n{sessions_block}\n"
        f"\nCOMPLIANCE:\n"
        f"  Sessions {comp['pct_sessions']}% "
        f"({comp['successful']} done · {comp['moderate']} moderate · "
        f"{comp['missed']} missed of {comp['planned']} planned)\n"
        f"  Volume {comp['actual_km']:.1f} / {comp['planned_km']:.1f} km ({comp['pct_km']}%)\n"
        f"  Key sessions (quality + long): {comp['key_done']}/{comp['key_planned']}\n"
        f"\nWrite a weekly debrief note (~150–180 words). "
        f"Name the week's main strength, main weakness, and ONE concrete improvement. "
        f"React to athlete comments where given. Cite specific numbers. "
        f"Sign off as 'LRP Coach'."
    )
