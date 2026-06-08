"""
Local LLM coaching narrative via mlx-lm (Apple Silicon / M-chip optimised).

Model  : mlx-community/Qwen2.5-7B-Instruct-4bit (~4 GB, one-time download)
Speed  : ~60–80 tok/s on M4 → 300-token note in ~4–6 seconds
Fallback: returns a plain text summary if mlx-lm is unavailable
"""

from __future__ import annotations

_model = None
_tokenizer = None

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"

SYSTEM_PROMPT = (
    "You are an experienced marathon running coach specialising in the Jack Daniels "
    "Running Formula and Critical Velocity (SVC/vitesse critique) training. "
    "You analyse athlete data and write personalised weekly coaching notes. "
    "Rules: always cite specific numbers from the data; be honest about fatigue signals; "
    "end with one concrete focus for the coming week. Length: 200–250 words."
)


def _load():
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        _model, _tokenizer = load(MODEL_ID)


def coaching_note(context: str) -> str:
    """Generate a coaching note from a structured context string."""
    try:
        _load()
        from mlx_lm import generate
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": context},
        ]
        prompt = _tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(_model, _tokenizer, prompt=prompt, max_tokens=380, verbose=False)
    except Exception as exc:
        return (
            f"[LLM unavailable — {exc}]\n\n"
            "Tip: run `pip install mlx-lm` in your venv to enable coaching notes."
        )


def build_context(
    name: str,
    goal_race: str,
    goal_time: str,
    weeks_left: int,
    phase: str,
    zones_dict: dict,
    metrics,    # WeekMetrics
    result,     # AdaptationResult
    next_focus: str,
) -> str:
    zone_lines = "\n".join(f"  {k}: {v}" for k, v in zones_dict.items())
    flag_lines = "\n".join(f"  • {f}" for f in result.flags)

    vol_pct = (metrics.actual_km / max(metrics.target_km, 1)) * 100
    q_pace  = _fmt(metrics.quality_avg_pace_s)
    q_tgt   = _fmt(metrics.quality_target_pace_s)
    hr_str  = f"{metrics.easy_avg_hr:.0f} bpm" if metrics.easy_avg_hr else "n/a"
    prev_hr = f"{metrics.prev_easy_hr:.0f} bpm" if metrics.prev_easy_hr else "n/a"
    drift   = f"{metrics.avg_hr_drift:.1f}%" if metrics.avg_hr_drift is not None else "n/a"

    adj_sign = "+" if result.volume_adj >= 1 else ""
    adj_pct  = f"{adj_sign}{(result.volume_adj - 1) * 100:.0f}%"

    return f"""ATHLETE: {name}
GOAL: {goal_race} — target {goal_time} | {weeks_left} weeks to race
CURRENT PHASE: {phase}

TRAINING ZONES:
{zone_lines}

THIS WEEK — ACTUAL vs PLAN:
  Volume: {metrics.actual_km:.1f} km vs {metrics.target_km:.1f} km target ({vol_pct:.0f}%)
  Quality session pace: {q_pace} (target {q_tgt})
  Easy run avg HR: {hr_str} (prev week: {prev_hr})
  HR drift avg: {drift}
  Feeling score: {metrics.avg_feeling:.1f} / 5

SIGNALS DETECTED:
{flag_lines}

PERFORMANCE SCORE: {result.score} / 100
ADAPTATION: {result.decision} — {result.reasoning}
  Next week volume adjustment: {adj_pct}
  Quality session: {"replaced with easy run" if result.drop_quality else "kept as planned"}
  Extra recovery day: {"yes" if result.add_recovery else "no"}

NEXT WEEK FOCUS: {next_focus}

Write the weekly coaching note for {name}. Reference the specific numbers above."""


def _fmt(sec: float | None) -> str:
    if not sec:
        return "n/a"
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d} /km"
