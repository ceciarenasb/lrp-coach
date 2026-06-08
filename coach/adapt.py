"""
Weekly performance scoring and adaptation rules.

Inputs : actual FIT metrics + feeling score vs this week's plan target
Outputs: score (0–100), decision (PROGRESS/MAINTAIN/CONSOLIDATE/RECOVER),
         specific flags, and adjustments to next week's plan
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WeekMetrics:
    actual_km: float
    target_km: float
    quality_avg_pace_s: float | None       # sec/km — actual quality session
    quality_target_pace_s: float | None    # sec/km — target from plan
    easy_avg_hr: float | None              # bpm — this week's easy runs
    prev_easy_hr: float | None             # bpm — last week's easy runs
    avg_hr_drift: float | None             # % drift across sessions
    avg_feeling: float                     # 1–5


@dataclass
class AdaptationResult:
    score: int
    decision: str          # PROGRESS / MAINTAIN / CONSOLIDATE / RECOVER
    flags: list            # signal descriptions
    volume_adj: float      # multiplier for next week's target_km
    drop_quality: bool     # replace quality session with easy run
    add_recovery: bool     # insert an extra recovery day
    reasoning: str         # one-sentence summary for the LLM context


def score_week(m: WeekMetrics) -> AdaptationResult:
    score = 50
    flags = []

    # ── Volume ───────────────────────────────────────────────────────────
    vol_ratio = m.actual_km / max(m.target_km, 1)
    if vol_ratio >= 0.95:
        score += 10
        flags.append(f"Volume on target ({m.actual_km:.0f} / {m.target_km:.0f} km ✓)")
    elif vol_ratio >= 0.80:
        score += 3
        flags.append(f"Slightly below volume target ({m.actual_km:.0f} / {m.target_km:.0f} km)")
    else:
        score -= 12
        flags.append(f"Low volume week — {m.actual_km:.0f} of {m.target_km:.0f} km completed")

    # ── Quality session pace ──────────────────────────────────────────────
    if m.quality_avg_pace_s and m.quality_target_pace_s:
        dev = (m.quality_avg_pace_s - m.quality_target_pace_s) / m.quality_target_pace_s
        if dev <= 0.02:
            score += 18
            flags.append("Quality session at or ahead of target pace ✓")
        elif dev <= 0.05:
            score += 6
            flags.append(f"Quality session slightly below target pace (+{dev*100:.0f}%)")
        else:
            score -= 15
            flags.append(f"Quality session significantly below target (+{dev*100:.0f}%) — check recovery")

    # ── HR trend on easy runs ─────────────────────────────────────────────
    if m.easy_avg_hr and m.prev_easy_hr and m.prev_easy_hr > 0:
        hr_trend = m.easy_avg_hr / m.prev_easy_hr
        if hr_trend > 1.06:
            score -= 22
            flags.append(
                f"HR rising significantly on easy runs (+{(hr_trend-1)*100:.0f}% vs last week) "
                "— clear fatigue signal"
            )
        elif hr_trend > 1.03:
            score -= 8
            flags.append(f"HR slightly elevated on easy runs (+{(hr_trend-1)*100:.0f}%) — monitor closely")
        elif hr_trend < 0.97:
            score += 12
            flags.append(f"HR dropping at similar effort — aerobic fitness improving ✓")
        else:
            flags.append("HR stable on easy runs ✓")

    # ── HR drift ─────────────────────────────────────────────────────────
    if m.avg_hr_drift is not None:
        if m.avg_hr_drift > 8:
            score -= 22
            flags.append(f"High HR drift ({m.avg_hr_drift:.1f}%) — significant fatigue or dehydration")
        elif m.avg_hr_drift > 5:
            score -= 10
            flags.append(f"Moderate HR drift ({m.avg_hr_drift:.1f}%) — build in extra recovery")
        else:
            score += 5
            flags.append(f"HR drift within normal range ({m.avg_hr_drift:.1f}%) ✓")

    # ── Feeling ───────────────────────────────────────────────────────────
    feeling_delta = (m.avg_feeling - 3.0) * 9
    score += feeling_delta
    if m.avg_feeling >= 4:
        flags.append(f"Strong subjective feeling ({m.avg_feeling:.1f}/5) ✓")
    elif m.avg_feeling <= 2:
        flags.append(f"Low subjective feeling ({m.avg_feeling:.1f}/5) — body is sending a signal")

    score = max(0, min(100, round(score)))

    # ── Decision ──────────────────────────────────────────────────────────
    if score >= 78:
        decision, volume_adj, drop_quality, add_recovery = "PROGRESS",     1.06, False, False
        reasoning = "Strong week across all indicators — ready to absorb more load."
    elif score >= 58:
        decision, volume_adj, drop_quality, add_recovery = "MAINTAIN",     1.00, False, False
        reasoning = "Solid week, plan is working — hold the course."
    elif score >= 38:
        decision, volume_adj, drop_quality, add_recovery = "CONSOLIDATE",  0.95, True,  False
        reasoning = "Fatigue signals present — consolidate volume and protect recovery."
    else:
        decision, volume_adj, drop_quality, add_recovery = "RECOVER",      0.85, True,  True
        reasoning = "Multiple stress signals — cut load this week, prioritize sleep and nutrition."

    return AdaptationResult(
        score=score,
        decision=decision,
        flags=flags,
        volume_adj=volume_adj,
        drop_quality=drop_quality,
        add_recovery=add_recovery,
        reasoning=reasoning,
    )
