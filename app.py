"""LRP Coach — Marathon Training Assistant"""

from __future__ import annotations

from datetime import datetime

import gradio as gr
import pandas as pd

from coach import adjustments as adj_mod
from coach import fit as fit_mod
from coach import llm as llm_mod
from coach import state as state_mod
from coach.adapt import WeekMetrics, score_week
from coach.plan import generate_plan, plan_to_rows
from coach.zones import (
    Zones, build_zones, cv_from_two_efforts, cv_from_vdot,
    fmt_pace, vdot_from_race, zones_summary,
)

# ── Constants ──────────────────────────────────────────────────────────────

DISTANCES = {
    "5 km": 5_000,
    "10 km": 10_000,
    "Half-marathon": 21_097,
    "Marathon": 42_195,
}
DIST_KEYS = list(DISTANCES.keys())
WEEKDAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_MAP = {v: i for i, v in enumerate(WEEKDAY_LABELS)}

# ── Pre-load saved state for form defaults ─────────────────────────────────

_saved   = state_mod.load()
_prof    = _saved.get("profile", {})
_sched   = _saved.get("schedule", {})
_has_plan = bool(_saved.get("plan"))

def _time_parts(total_s: int) -> tuple:
    h, r = divmod(int(total_s), 3600)
    m, s = divmod(r, 60)
    return h, m, s

_g_h, _g_m, _g_s = _time_parts(_prof.get("goal_time_s", 13500))

_b1_dist_saved = _prof.get("b1_dist", "Half-marathon")
_b1_h, _b1_m, _b1_s = _time_parts(_prof.get("b1_time_s", 6300))
_b2_dist_saved = _prof.get("b2_dist", "10 km")
_b2_h, _b2_m, _b2_s = _time_parts(_prof.get("b2_time_s", 2850))

_saved_run_days = [WEEKDAY_LABELS[i] for i in _sched.get("run_days", [1, 3, 4, 5, 6])]
_saved_lrp_day  = (WEEKDAY_LABELS[_sched["lrp_day"]]
                   if _sched.get("lrp_day") is not None else "None")
_saved_strength = [WEEKDAY_LABELS[i] for i in _sched.get("strength_days", [])]
_saved_cycling  = [WEEKDAY_LABELS[i] for i in _sched.get("cycling_days", [])]

# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_time(h, m, s):
    try:
        return int(h) * 3600 + int(m) * 60 + int(s)
    except (ValueError, TypeError):
        return None


def _fmt_duration(total_s: int) -> str:
    h, r = divmod(int(total_s), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _plan_to_df(plan: list) -> pd.DataFrame:
    rows = []
    for w in plan:
        for d in w["days"]:
            rows.append({
                "Week":    w["week_num"],
                "Phase":   w["phase"],
                "Date":    d["date"],
                "Day":     ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][d["weekday"]],
                "Session": d["session_type"],
                "Detail":  d["description"],
                "Km":      f"{d['distance_km']:.0f}" if d.get("distance_km") else "—",
                "Targets": ", ".join(f"{k}: {v}" for k, v in d.get("targets", {}).items()) or "—",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Startup loaders (called by demo.load) ─────────────────────────────────

def load_plan_on_start():
    plan = state_mod.load().get("plan", [])
    return _plan_to_df(plan)


def load_history_on_start():
    hist = state_mod.load().get("history", [])
    return pd.DataFrame(hist) if hist else pd.DataFrame()


def load_status_on_start():
    s = state_mod.load()
    parts = []
    if s.get("profile", {}).get("goal_race"):
        p = s["profile"]
        parts.append(f"Goal: **{p['goal_race']}** on {p.get('marathon_date','?')}  |  "
                     f"target {_fmt_duration(p.get('goal_time_s',0))}")
    if s.get("plan"):
        parts.append(f"Plan: **{len(s['plan'])} weeks** saved")
    if s.get("history"):
        parts.append(f"History: **{len(s['history'])} runs** saved")
    return " · ".join(parts) if parts else "No saved data yet — fill in Setup & Plan to get started."


# ── Tab 1: Setup & plan generation ────────────────────────────────────────

def compute_and_generate(
    name, goal_race, marathon_date_str,
    goal_h, goal_m, goal_s,
    b1_dist, b1_h, b1_m, b1_s,
    b2_dist, b2_h, b2_m, b2_s,
    injury_level, injury_notes,
    run_days_labels, lrp_day_label, lrp_km, lrp_type,
    strength_labels, cycling_labels,
):
    goal_total = _parse_time(goal_h, goal_m, goal_s)
    if not goal_total:
        return {"error": "Invalid goal time"}, None, "Fix errors above."

    try:
        marathon_date = datetime.strptime(marathon_date_str, "%Y-%m-%d").date()
    except Exception:
        return {"error": "Invalid date — use YYYY-MM-DD"}, None, "Fix errors above."

    b1_time  = _parse_time(b1_h, b1_m, b1_s)
    b1_m_val = DISTANCES.get(b1_dist)
    if not b1_time or not b1_m_val:
        return {"error": "Invalid benchmark 1"}, None, "Fix errors above."

    vdot     = vdot_from_race(b1_m_val, b1_time)
    b2_time  = _parse_time(b2_h, b2_m, b2_s)
    b2_m_val = DISTANCES.get(b2_dist)

    if b2_time and b2_m_val and b2_m_val != b1_m_val:
        efforts   = sorted([(b1_m_val, b1_time), (b2_m_val, b2_time)])
        cv        = cv_from_two_efforts(efforts[0][0], efforts[0][1], efforts[1][0], efforts[1][1])
        cv_source = "exact (Monod-Billat)"
    else:
        cv        = cv_from_vdot(vdot)
        cv_source = "estimated from VDOT"

    zones    = build_zones(vdot, cv)
    run_days = sorted([WEEKDAY_MAP[d] for d in (run_days_labels or [])])
    lrp_day  = WEEKDAY_MAP.get(lrp_day_label) if lrp_day_label and lrp_day_label != "None" else None
    strength = [WEEKDAY_MAP[d] for d in (strength_labels or [])]
    cycling  = [WEEKDAY_MAP[d] for d in (cycling_labels or [])]

    plan = generate_plan(
        marathon_date=marathon_date,
        goal_time_s=goal_total,
        zones=zones,
        run_days=run_days,
        lrp_day=lrp_day,
        lrp_km=float(lrp_km or 10),
        lrp_type=lrp_type or "easy",
        strength_days=strength,
        cycling_days=cycling,
        injury=injury_level,
    )

    existing = state_mod.load()
    existing.update({
        "profile": {
            "name": name, "goal_race": goal_race,
            "marathon_date": str(marathon_date),
            "goal_time_s": goal_total,
            "injury_level": injury_level,
            "injury_notes": injury_notes,
            # Save benchmark inputs so form re-populates on next launch
            "b1_dist": b1_dist, "b1_time_s": b1_time,
            "b2_dist": b2_dist, "b2_time_s": b2_time or 0,
        },
        "zones": zones.__dict__,
        "schedule": {
            "run_days": run_days, "lrp_day": lrp_day,
            "lrp_km": float(lrp_km or 10), "lrp_type": lrp_type or "easy",
            "strength_days": strength, "cycling_days": cycling,
        },
        "plan": [
            {
                "week_num": w.week_num, "phase": w.phase,
                "focus": w.focus, "target_km": w.target_km,
                "days": [
                    {
                        "date": str(d.date), "weekday": d.weekday,
                        "session_type": d.session.type,
                        "description": d.session.description,
                        "distance_km": d.session.distance_km,
                        "targets": d.session.targets,
                    }
                    for d in w.days
                ],
            }
            for w in plan
        ],
    })
    state_mod.save(existing)

    zs = zones_summary(zones)
    zs["CV source"] = cv_source
    msg = (f"Plan saved — {len(plan)} weeks to {goal_race}  |  "
           f"VDOT {zones.vdot}  |  SVC {zones.cv_mps * 3.6:.1f} km/h  |  "
           f"Target {_fmt_duration(goal_total)}")
    return zs, _plan_to_df(plan), msg


# ── Tab 2: Run history ─────────────────────────────────────────────────────

def process_history(files):
    if not files:
        return load_history_on_start(), "No files uploaded — showing saved history."

    new_records = [
        r for f in files
        if (r := fit_mod.summarize(f.name)) and "error" not in r
    ]
    if not new_records:
        return load_history_on_start(), "No valid FIT records in uploaded files."

    existing = state_mod.load().get("history", [])
    existing_dates = {r["date"] for r in existing}
    added = [r for r in new_records if r["date"] not in existing_dates]
    merged = sorted(existing + added, key=lambda r: r["date"], reverse=True)

    state_mod.update("history", merged)
    msg = f"{len(added)} new run{'s' if len(added) != 1 else ''} added  ({len(merged)} total saved)"
    return pd.DataFrame(merged), msg


def clear_history():
    state_mod.update("history", [])
    return pd.DataFrame(), "History cleared."


# ── Tab 4: Adjustments ────────────────────────────────────────────────────

def apply_adjustments_ui(user_message, from_week, num_weeks,
                          no_club_run, easy_only, volume_slider):
    state      = state_mod.load()
    plan       = state.get("plan", [])
    profile    = state.get("profile", {})
    zones_data = state.get("zones", {})

    if not plan:
        return "No plan found — generate your plan first in Setup & Plan.", {}, pd.DataFrame()

    updated_plan, change_log = adj_mod.apply(
        plan=plan,
        from_week=int(from_week),
        num_weeks=int(num_weeks),
        no_club_run=no_club_run,
        easy_only=easy_only,
        volume_pct=volume_slider / 100.0,
        zones_data=zones_data,
    )
    state["plan"] = updated_plan
    state_mod.save(state)

    weeks_left = len(plan) - int(from_week) + 1
    phase = next((w["phase"] for w in plan if w["week_num"] == int(from_week)), "Unknown")

    context = adj_mod.build_adjustment_context(
        athlete_name=profile.get("name", "Athlete"),
        goal_race=profile.get("goal_race", "your marathon"),
        weeks_left=weeks_left, phase=phase,
        user_message=user_message or "(no message provided)",
        no_club_run=no_club_run, easy_only=easy_only,
        volume_pct=volume_slider / 100.0,
        from_week=int(from_week), num_weeks=int(num_weeks),
        change_log=change_log,
    )
    note    = llm_mod.coaching_note(context)
    summary = {
        "Weeks modified": (f"{from_week} → {int(from_week)+int(num_weeks)-1}"
                           if int(num_weeks) > 0 else f"{from_week} → end of plan"),
        "Changes applied": len(change_log),
        "Log": change_log,
    }
    return note, summary, _plan_to_df(updated_plan)


# ── Tab 5: Weekly check-in ─────────────────────────────────────────────────

def _apply_checkin_adaptation(plan, wk_idx, result, zones_data):
    next_idx = wk_idx + 1
    if next_idx >= len(plan):
        return plan
    z        = Zones(**zones_data)
    easy_pace = f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}"
    next_wk  = plan[next_idx]

    next_wk["target_km"] = round(next_wk["target_km"] * result.volume_adj, 1)

    if result.drop_quality:
        from coach.adjustments import QUALITY_TYPES
        for d in next_wk["days"]:
            if d["session_type"] in QUALITY_TYPES:
                d["session_type"] = "Easy"
                d["description"]  = f"Easy {max(8, d['distance_km']):.0f} km — quality removed (adaptation)"
                d["targets"]      = {"pace": easy_pace}

    if result.add_recovery:
        candidates = [d for d in next_wk["days"]
                      if d["session_type"] == "Easy" and d["distance_km"] <= 10]
        if candidates:
            lightest = min(candidates, key=lambda d: d["distance_km"])
            lightest.update({
                "session_type": "Recovery",
                "description":  "Recovery 6 km — adaptation week, protect the body",
                "distance_km":  6.0,
                "targets":      {"pace": f"slower than {fmt_pace(z.easy_lo)}"},
            })

    next_wk["focus"] = next_wk["focus"].rstrip() + f"  [adapted: {result.decision} {result.score}/100]"
    return plan


def checkin(week_num, files, feeling, prev_hr_input):
    state      = state_mod.load()
    plan       = state.get("plan", [])
    profile    = state.get("profile", {})
    zones_data = state.get("zones", {})

    if not plan:
        return None, "No plan found — go to Setup & Plan first.", pd.DataFrame()

    wk_idx = int(week_num) - 1
    if wk_idx < 0 or wk_idx >= len(plan):
        return None, f"Week {week_num} not in plan ({len(plan)} weeks total).", pd.DataFrame()

    week      = plan[wk_idx]
    target_km = week["target_km"]

    quality_target_s = None
    for d in week["days"]:
        if d.get("session_type") in ("Tempo", "SVC Intervals", "Marathon Pace"):
            raw = (d.get("targets", {}).get("T_pace")
                   or d.get("targets", {}).get("SVC_pace")
                   or d.get("targets", {}).get("M_pace"))
            if raw and ":" in raw:
                try:
                    mm, ss = raw.replace(" /km", "").split(":")
                    quality_target_s = int(mm) * 60 + int(ss)
                except Exception:
                    pass
            break

    actual_km, quality_pace_s = 0.0, None
    hr_list, drift_list = [], []

    if files:
        for f in files:
            s = fit_mod.summarize(f.name)
            if not s or "error" in s:
                continue
            actual_km += s.get("distance_km", 0)
            if s.get("avg_hr"):      hr_list.append(s["avg_hr"])
            if s.get("hr_drift_pct") is not None: drift_list.append(s["hr_drift_pct"])
            if quality_target_s and s.get("avg_pace_s"):
                if quality_target_s - 30 <= s["avg_pace_s"] <= quality_target_s + 90:
                    quality_pace_s = s["avg_pace_s"]

    metrics = WeekMetrics(
        actual_km=actual_km, target_km=target_km,
        quality_avg_pace_s=quality_pace_s, quality_target_pace_s=quality_target_s,
        easy_avg_hr=sum(hr_list)/len(hr_list) if hr_list else None,
        prev_easy_hr=float(prev_hr_input) if prev_hr_input else None,
        avg_hr_drift=sum(drift_list)/len(drift_list) if drift_list else None,
        avg_feeling=float(feeling),
    )
    result = score_week(metrics)

    state["plan"] = _apply_checkin_adaptation(plan, wk_idx, result, zones_data)
    state_mod.save(state)

    assessment = {
        "Performance score": f"{result.score} / 100",
        "Decision":          result.decision,
        "Volume":            f"{actual_km:.1f} km  (target {target_km:.1f} km)",
        "Quality session pace": (f"{fmt_pace(quality_pace_s)}  (target {fmt_pace(quality_target_s)})"
                                  if quality_pace_s else "not detected in FIT files"),
        "Signals": result.flags,
        "Next week volume": f"×{result.volume_adj:.2f}",
        "Quality session":  "→ replaced with easy run" if result.drop_quality else "kept as planned",
        "Extra recovery":   "yes" if result.add_recovery else "no",
    }

    z          = Zones(**zones_data)
    weeks_left = len(plan) - wk_idx
    next_focus = plan[wk_idx+1]["focus"] if wk_idx+1 < len(plan) else "Race week — stay calm"

    note = llm_mod.coaching_note(llm_mod.build_context(
        name=profile.get("name", "Athlete"),
        goal_race=profile.get("goal_race", "your marathon"),
        goal_time=_fmt_duration(profile.get("goal_time_s", 0)),
        weeks_left=weeks_left, phase=week["phase"],
        zones_dict=zones_summary(z),
        metrics=metrics, result=result, next_focus=next_focus,
    ))

    return assessment, note, _plan_to_df(state["plan"])


# ── UI ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="LRP Coach", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# LRP Coach — Marathon Training Assistant\n"
        "Daniels VDOT + Critical Velocity plan · weekly FIT-based adaptation · local AI coaching"
    )
    status_bar = gr.Markdown("")

    # ── Tab 1: Setup & Plan ───────────────────────────────────────────────
    with gr.Tab("Setup & Plan"):
        gr.Markdown(
            "Fill in your profile and click **Update Plan**. "
            "Your settings are saved automatically and will reload next time you open the app."
        )
        gr.Markdown("## Profile & Goals")
        with gr.Row():
            name_in          = gr.Textbox(label="Your name", value=_prof.get("name", "Cecilia"))
            goal_race_in     = gr.Textbox(label="Target race", value=_prof.get("goal_race", ""),
                                          placeholder="Paris Marathon 2027")
            marathon_date_in = gr.Textbox(label="Race date (YYYY-MM-DD)",
                                          value=_prof.get("marathon_date", ""),
                                          placeholder="2027-04-11")

        gr.Markdown("### Target finish time")
        with gr.Row():
            goal_h = gr.Number(label="Hours",   value=_g_h, precision=0, minimum=2, maximum=7)
            goal_m = gr.Number(label="Minutes", value=_g_m, precision=0, minimum=0, maximum=59)
            goal_s = gr.Number(label="Seconds", value=_g_s, precision=0, minimum=0, maximum=59)

        gr.Markdown("### Benchmark 1 — required (recent race or time trial)")
        with gr.Row():
            b1_dist = gr.Dropdown(DIST_KEYS, label="Distance", value=_b1_dist_saved)
            b1_h    = gr.Number(label="h",   value=_b1_h, precision=0, minimum=0, maximum=5)
            b1_m    = gr.Number(label="min", value=_b1_m, precision=0, minimum=0, maximum=59)
            b1_s    = gr.Number(label="sec", value=_b1_s, precision=0, minimum=0, maximum=59)

        gr.Markdown(
            "### Benchmark 2 — optional\n"
            "A *different* distance gives an exact SVC via the Monod-Billat formula. "
            "Leave at zero if you only have one result."
        )
        with gr.Row():
            b2_dist = gr.Dropdown(DIST_KEYS, label="Distance", value=_b2_dist_saved)
            b2_h    = gr.Number(label="h",   value=_b2_h, precision=0, minimum=0, maximum=5)
            b2_m    = gr.Number(label="min", value=_b2_m, precision=0, minimum=0, maximum=59)
            b2_s    = gr.Number(label="sec", value=_b2_s, precision=0, minimum=0, maximum=59)

        gr.Markdown("### Physical status")
        with gr.Row():
            injury_in       = gr.Radio(
                ["none", "light", "moderate"], label="Rehab / injury level",
                value=_prof.get("injury_level", "none"),
                info="none = full training  ·  light = reduced intensity  ·  moderate = significant restriction",
            )
            injury_notes_in = gr.Textbox(label="Notes", value=_prof.get("injury_notes", ""), lines=2)

        gr.Markdown("## Weekly Schedule")
        run_days_in = gr.CheckboxGroup(
            WEEKDAY_LABELS, label="Days available for running",
            value=_saved_run_days,
        )
        gr.Markdown("### LRP club runs (locked into plan)")
        with gr.Row():
            lrp_day_in  = gr.Dropdown(["None"] + WEEKDAY_LABELS, label="Club run day",
                                       value=_saved_lrp_day)
            lrp_km_in   = gr.Number(label="Approx. distance (km)",
                                     value=_sched.get("lrp_km", 12), minimum=5, maximum=35)
            lrp_type_in = gr.Radio(["easy", "tempo", "long"], label="Session type",
                                    value=_sched.get("lrp_type", "easy"))

        gr.Markdown("### Cross-training")
        with gr.Row():
            strength_in = gr.CheckboxGroup(WEEKDAY_LABELS, label="Strength days",
                                            value=_saved_strength)
            cycling_in  = gr.CheckboxGroup(WEEKDAY_LABELS, label="Cycling / Zwift days",
                                            value=_saved_cycling)

        gen_btn   = gr.Button("Update Plan" if _has_plan else "Calculate Zones & Generate Plan",
                               variant="primary", size="lg")
        gen_msg   = gr.Textbox(label="Status", interactive=False)
        zones_out = gr.JSON(label="Training Zones")

    # ── Tab 2: Run History ────────────────────────────────────────────────
    with gr.Tab("Run History"):
        gr.Markdown(
            "Upload .fit files to add runs to your history log. "
            "**Already-saved runs are not duplicated** — only new dates are added. "
            "Metrics: distance, duration, pace, avg HR, max HR, HR drift, cadence, elevation gain."
        )
        hist_files = gr.File(file_count="multiple", file_types=[".fit"], label="Add FIT files")
        with gr.Row():
            hist_btn       = gr.Button("Add to history", variant="primary")
            hist_clear_btn = gr.Button("Clear all history", variant="stop")
        hist_msg = gr.Textbox(label="Status", interactive=False)
        hist_df  = gr.Dataframe(label="Run log", wrap=True)
        hist_btn.click(process_history, inputs=hist_files, outputs=[hist_df, hist_msg])
        hist_clear_btn.click(clear_history, outputs=[hist_df, hist_msg])

    # ── Tab 3: My Plan ────────────────────────────────────────────────────
    with gr.Tab("My Plan"):
        gr.Markdown(
            "Your plan loads automatically. "
            "After any adjustment or check-in the plan here reflects the latest state."
        )
        plan_df = gr.Dataframe(label="Week-by-week plan", wrap=True)

    # ── Tab 4: Adjustments ───────────────────────────────────────────────
    with gr.Tab("Adjustments"):
        gr.Markdown(
            "## Tell your coach what's changed\n"
            "Physio, travel, illness, extra fatigue — describe the situation, "
            "set how many weeks are affected, and the plan updates immediately. "
            "The AI writes a coaching note explaining the change."
        )
        adj_message_in = gr.Textbox(
            label="What's going on?",
            placeholder="e.g. Still doing physio for my knee — skipping LRP for 3 weeks, easy runs only.",
            lines=3,
        )
        with gr.Row():
            adj_from_in  = gr.Number(label="Starting from week #", value=1, precision=0, minimum=1)
            adj_weeks_in = gr.Number(label="For how many weeks  (0 = rest of plan)",
                                      value=3, precision=0, minimum=0)

        gr.Markdown("### What to change")
        with gr.Row():
            adj_no_lrp_in    = gr.Checkbox(label="Skip LRP club runs → replace with easy")
            adj_easy_only_in = gr.Checkbox(label="Easy runs only → remove all quality sessions")
        adj_volume_in = gr.Slider(60, 110, value=100, step=5, label="Volume (% of planned km)")

        adj_btn = gr.Button("Apply to plan & get coaching note", variant="primary")
        gr.Markdown("### Coach response")
        adj_note_out    = gr.Textbox(label="From your coach", lines=10, interactive=False)
        adj_changes_out = gr.JSON(label="Changes applied")
        adj_plan_out    = gr.Dataframe(label="Updated plan", wrap=True)

        adj_btn.click(
            apply_adjustments_ui,
            inputs=[adj_message_in, adj_from_in, adj_weeks_in,
                    adj_no_lrp_in, adj_easy_only_in, adj_volume_in],
            outputs=[adj_note_out, adj_changes_out, adj_plan_out],
        )

    # ── Tab 5: Weekly Check-in ────────────────────────────────────────────
    with gr.Tab("Weekly Check-in"):
        gr.Markdown(
            "## End-of-week coaching session\n"
            "Upload this week's FIT files and rate how you felt. "
            "The app scores pace, HR trend, HR drift, volume, and feeling — "
            "then adapts next week's plan and writes your coaching note."
        )
        with gr.Row():
            checkin_week = gr.Number(label="Plan week number", value=1, precision=0, minimum=1)
            feeling_in   = gr.Slider(1, 5, value=3, step=0.5,
                                     label="Overall feeling  (1 = rough · 5 = excellent)")
            prev_hr_in   = gr.Number(label="Last week avg easy HR (bpm, optional)", value=None)

        checkin_files = gr.File(file_count="multiple", file_types=[".fit"],
                                label="This week's FIT files")
        checkin_btn   = gr.Button("Analyse week & get coaching note", variant="primary")

        gr.Markdown("### Performance assessment")
        checkin_json = gr.JSON(label="Week summary & adaptations")
        gr.Markdown("### Your coaching note")
        coaching_out = gr.Textbox(label="From your coach", lines=12, interactive=False)
        gr.Markdown("### Updated plan")
        checkin_plan_out = gr.Dataframe(label="Plan (next week adapted)", wrap=True)

        checkin_btn.click(
            checkin,
            inputs=[checkin_week, checkin_files, feeling_in, prev_hr_in],
            outputs=[checkin_json, coaching_out, checkin_plan_out],
        )

    # ── Wire generate button ──────────────────────────────────────────────
    gen_btn.click(
        compute_and_generate,
        inputs=[
            name_in, goal_race_in, marathon_date_in,
            goal_h, goal_m, goal_s,
            b1_dist, b1_h, b1_m, b1_s,
            b2_dist, b2_h, b2_m, b2_s,
            injury_in, injury_notes_in,
            run_days_in, lrp_day_in, lrp_km_in, lrp_type_in,
            strength_in, cycling_in,
        ],
        outputs=[zones_out, plan_df, gen_msg],
    )

    # ── Load saved data on page open ─────────────────────────────────────
    demo.load(load_status_on_start,  outputs=status_bar)
    demo.load(load_plan_on_start,    outputs=plan_df)
    demo.load(load_history_on_start, outputs=hist_df)


if __name__ == "__main__":
    demo.launch()
