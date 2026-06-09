"""LRP Coach — Marathon Training Assistant"""

from __future__ import annotations

from datetime import datetime

import gradio as gr
import pandas as pd

from coach import adjustments as adj_mod
from coach import fit as fit_mod
from coach import llm as llm_mod
from coach import state as state_mod
from coach import garmin as garmin_mod
from coach import scheduler as sched_mod
from coach.adapt import WeekMetrics, score_week
from coach.plan import generate_plan
from coach.zones import (
    Zones, build_zones, cv_from_two_efforts, cv_from_vdot,
    fmt_pace, infer_vdot_adjustment, vdot_from_race, vdot_recency_factor,
    zones_summary,
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


def _iso_to_dmy(iso: str) -> str:
    if not iso:
        return ""
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%m-%Y")
    except Exception:
        return iso


def _dmy_to_iso(dmy: str) -> str:
    if not dmy:
        return ""
    try:
        return datetime.strptime(dmy.strip(), "%d-%m-%Y").strftime("%Y-%m-%d")
    except Exception:
        return dmy


_g_h, _g_m, _g_s = _time_parts(_prof.get("goal_time_s", 13500))

_b1_dist_saved = _prof.get("b1_dist", "Half-marathon")
_b1_h, _b1_m, _b1_s = _time_parts(_prof.get("b1_time_s", 6300))
_b1_date_saved = _iso_to_dmy(_prof.get("b1_date", ""))
_b2_dist_saved = _prof.get("b2_dist", "10 km")
_b2_h, _b2_m, _b2_s = _time_parts(_prof.get("b2_time_s", 2850))
_b2_date_saved = _iso_to_dmy(_prof.get("b2_date", ""))

_saved_run_days = [WEEKDAY_LABELS[i] for i in _sched.get("run_days", [1, 3, 4, 5, 6])]
_saved_strength = [WEEKDAY_LABELS[i] for i in _sched.get("strength_days", [])]
_saved_cycling  = [WEEKDAY_LABELS[i] for i in _sched.get("cycling_days", [])]

# Migrate old single lrp_day/km/type → new lrp_sessions list
def _load_lrp_sessions(sched: dict) -> list:
    if sched.get("lrp_sessions"):
        return sched["lrp_sessions"]
    # Legacy single-session state
    if sched.get("lrp_day") is not None:
        return [{"day": sched["lrp_day"], "km": sched.get("lrp_km", 12.0),
                 "type": sched.get("lrp_type", "easy")}]
    return [{"day": None, "km": 12.0, "type": "easy"}]

_saved_lrp_sessions = _load_lrp_sessions(_sched)
_lrp1 = _saved_lrp_sessions[0] if len(_saved_lrp_sessions) > 0 else {}
_lrp2 = _saved_lrp_sessions[1] if len(_saved_lrp_sessions) > 1 else {}
_lrp3 = _saved_lrp_sessions[2] if len(_saved_lrp_sessions) > 2 else {}
_lrp4 = _saved_lrp_sessions[3] if len(_saved_lrp_sessions) > 3 else {}

def _lrp_day_label(s): return WEEKDAY_LABELS[s["day"]] if s.get("day") is not None else "None"

_lrp1_day  = _lrp_day_label(_lrp1);  _lrp1_km = _lrp1.get("km", 12.0); _lrp1_type = _lrp1.get("type", "easy")
_lrp2_day  = _lrp_day_label(_lrp2);  _lrp2_km = _lrp2.get("km", 10.0); _lrp2_type = _lrp2.get("type", "easy")
_lrp3_day  = _lrp_day_label(_lrp3);  _lrp3_km = _lrp3.get("km", 10.0); _lrp3_type = _lrp3.get("type", "easy")
_lrp4_day  = _lrp_day_label(_lrp4);  _lrp4_km = _lrp4.get("km", 10.0); _lrp4_type = _lrp4.get("type", "easy")
_has_lrp2  = bool(_lrp2.get("day") is not None)
_has_lrp3  = bool(_lrp3.get("day") is not None)
_has_lrp4  = bool(_lrp4.get("day") is not None)

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


def _format_history_df(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw metric keys and format values for display."""
    if df.empty:
        return df
    d = df.copy()
    if "duration_s" in d.columns:
        def _dur(s):
            try:
                s = int(float(s))
            except (TypeError, ValueError):
                return "—"
            h, r = divmod(s, 3600)
            m = r // 60
            return f"{h}h{m:02d}m" if h else f"{m}m"
        d["duration_s"] = d["duration_s"].apply(_dur)
    if "avg_pace_s" in d.columns:
        def _pace(s):
            try:
                s = int(float(s))
                return "—" if s <= 0 else f"{s // 60}:{s % 60:02d}/km"
            except (TypeError, ValueError):
                return "—"
        d["avg_pace_s"] = d["avg_pace_s"].apply(_pace)
    if "distance_km" in d.columns:
        d["distance_km"] = d["distance_km"].apply(
            lambda x: f"{float(x):.1f}" if pd.notna(x) and x else "—")
    if "hr_drift_pct" in d.columns:
        d["hr_drift_pct"] = d["hr_drift_pct"].apply(
            lambda x: f"{float(x):.1f}%" if pd.notna(x) and x else "—")
    d = d.rename(columns={
        "date": "Date", "distance_km": "Distance (km)", "duration_s": "Duration",
        "avg_pace_s": "Avg Pace", "avg_hr": "Avg HR", "max_hr": "Max HR",
        "hr_drift_pct": "HR Drift", "avg_cadence_spm": "Cadence",
        "elevation_gain_m": "Elevation (m)",
    })
    order = ["Date", "Distance (km)", "Duration", "Avg Pace", "Avg HR",
             "HR Drift", "Elevation (m)", "Cadence", "Max HR"]
    cols = [c for c in order if c in d.columns] + \
           [c for c in d.columns if c not in order]
    return d[cols]


def _plan_to_df(plan: list) -> pd.DataFrame:
    from datetime import date as _date
    rows = []
    for w in plan:
        for d in w["days"]:
            _d = _date.fromisoformat(d["date"])
            rows.append({
                "Week":    w["week_num"],
                "Phase":   w["phase"],
                "Date":    f"{_d.day} {_d.strftime('%B %Y')}",
                "Day":     _d.strftime("%a"),
                "Session": d["session_type"],
                "Detail":  d["description"],
                "Km":      f"{d['distance_km']:.0f}" if d.get("distance_km") else "—",
                "Targets": ", ".join(f"{k}: {v}" for k, v in d.get("targets", {}).items()) or "—",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


_PHASE_CONFIG = {
    "Base":  {"accent": "#059669", "badge_bg": "#DCFCE7", "badge_text": "#166534"},
    "Build": {"accent": "#2563EB", "badge_bg": "#DBEAFE", "badge_text": "#1E40AF"},
    "Peak":  {"accent": "#EA580C", "badge_bg": "#FFEDD5", "badge_text": "#9A3412"},
    "Taper": {"accent": "#64748B", "badge_bg": "#F1F5F9", "badge_text": "#475569"},
}

_SESSION_COLOR = {
    "Easy":          "#10B981",
    "Recovery":      "#34D399",
    "Long":          "#2563EB",
    "Tempo":         "#F5871F",
    "SVC Intervals": "#DC2626",
    "Marathon Pace": "#7C3AED",
    "Rest":          "#9CA3AF",
    "Strength":      "#D97706",
    "Cycling":       "#0891B2",
    "LRP Easy":      "#0D9488",
    "LRP Tempo":     "#EA580C",
    "LRP Long":      "#1D4ED8",
}


def _plan_to_html(plan: list) -> str:
    if not plan:
        return (
            "<div style='text-align:center;padding:56px 0;color:#9CA3AF;"
            "font-family:-apple-system,sans-serif'>"
            "<div style='font-size:44px;margin-bottom:12px'>🏃</div>"
            "<div style='font-size:15px;font-weight:600;color:#374151'>No plan yet</div>"
            "<div style='font-size:13px;margin-top:6px'>Go to Setup & Plan to generate your training plan.</div>"
            "</div>"
        )

    rows_html = ""
    prev_week = None
    row_idx = 0

    for w in plan:
        phase = w["phase"]
        cfg = _PHASE_CONFIG.get(phase, {"accent": "#64748B", "badge_bg": "#F1F5F9", "badge_text": "#475569"})
        for d in w["days"]:
            wk_num = w["week_num"]
            from datetime import date as _date
            _d = _date.fromisoformat(d["date"])
            day    = _d.strftime("%a")
            sess   = d["session_type"]
            badge_color = _SESSION_COLOR.get(sess, "#6B7280")
            km     = f"{d['distance_km']:.0f} km" if d.get("distance_km") else "—"
            desc   = d["description"]
            targets = " · ".join(f"{k}: {v}" for k, v in d.get("targets", {}).items()) or "—"
            week_sep = "border-top:2px solid #E5E7EB;" if wk_num != prev_week else ""
            prev_week = wk_num
            row_bg = "#ffffff" if row_idx % 2 == 0 else "#FAFAFA"
            row_idx += 1
            rows_html += (
                f"<tr style='background:{row_bg};{week_sep}'>"
                f"<td style='padding:9px 12px;font-weight:700;color:#1B2874;"
                f"border-left:3px solid {cfg['accent']};white-space:nowrap'>{wk_num}</td>"
                f"<td style='padding:9px 10px;white-space:nowrap'>"
                f"<span style='background:{cfg['badge_bg']};color:{cfg['badge_text']};"
                f"padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600'>{phase}</span></td>"
                f"<td style='padding:9px 10px;color:#9CA3AF;font-size:12px;white-space:nowrap'>{_d.day} {_d.strftime('%B %Y')}</td>"
                f"<td style='padding:9px 10px;font-weight:600;color:#374151;white-space:nowrap'>{day}</td>"
                f"<td style='padding:9px 10px;white-space:nowrap'>"
                f"<span style='background:{badge_color};color:#fff;padding:2px 9px;"
                f"border-radius:99px;font-size:11px;font-weight:700'>{sess}</span></td>"
                f"<td style='padding:9px 10px;color:#374151;font-size:13px'>{desc}</td>"
                f"<td style='padding:9px 10px;text-align:center;font-weight:700;"
                f"color:#1B2874;white-space:nowrap'>{km}</td>"
                f"<td style='padding:9px 10px;color:#9CA3AF;font-size:12px'>{targets}</td>"
                f"</tr>"
            )

    _th = ("padding:12px 10px;text-align:left;font-weight:600;"
           "font-size:11px;letter-spacing:0.05em;text-transform:uppercase;color:#fff")
    return (
        "<div style='overflow-x:auto;border-radius:12px;"
        "border:1px solid #F5871F;"
        "box-shadow:0 2px 8px rgba(0,0,0,0.06);"
        "font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif'>"
        "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
        "<thead>"
        f"<tr style='background:#F5871F'>"
        f"<th style='{_th};white-space:nowrap;padding-left:12px'>Wk</th>"
        f"<th style='{_th}'>Phase</th>"
        f"<th style='{_th}'>Date</th>"
        f"<th style='{_th}'>Day</th>"
        f"<th style='{_th}'>Session</th>"
        f"<th style='{_th}'>Details</th>"
        f"<th style='{_th};text-align:center'>Km</th>"
        f"<th style='{_th}'>Targets</th>"
        "</tr>"
        "</thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table>"
        "</div>"
    )


# ── Startup loaders ────────────────────────────────────────────────────────

def load_plan_on_start():
    plan = state_mod.load().get("plan", [])
    return _plan_to_html(plan)


def load_history_on_start():
    hist = state_mod.load().get("history", [])
    return _format_history_df(pd.DataFrame(hist)) if hist else pd.DataFrame()


def load_status_on_start():
    s = state_mod.load()
    parts = []
    if s.get("profile", {}).get("goal_race"):
        p = s["profile"]
        parts.append(
            f"<b>Goal:</b> {p['goal_race']} on {_iso_to_dmy(p.get('marathon_date','')) or '?'}"
            f"&nbsp; | &nbsp;target {_fmt_duration(p.get('goal_time_s', 0))}"
        )
    if s.get("plan"):
        parts.append(f"<b>Plan:</b> {len(s['plan'])} weeks saved")
    if s.get("history"):
        parts.append(f"<b>History:</b> {len(s['history'])} runs saved")
    text = " &nbsp;·&nbsp; ".join(parts) if parts else "No saved data yet — fill in Setup &amp; Plan to get started."
    return (
        "<div style='background:#1B2874;border-radius:8px;padding:9px 16px;"
        "color:#ffffff;font-size:13px;"
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;line-height:1.4'>"
        + text + "</div>"
    )


# ── Tab 1: Setup & plan generation ────────────────────────────────────────

def _parse_lrp_sessions(
    lrp1_day, lrp1_km, lrp1_type,
    lrp2_day, lrp2_km, lrp2_type, lrp2_visible,
    lrp3_day, lrp3_km, lrp3_type, lrp3_visible,
    lrp4_day, lrp4_km, lrp4_type, lrp4_visible,
) -> list:
    sessions = []
    slot_defaults = [12, 10, 10, 10]
    for i, (day_v, km_v, type_v, visible) in enumerate([
        (lrp1_day, lrp1_km, lrp1_type, True),
        (lrp2_day, lrp2_km, lrp2_type, lrp2_visible),
        (lrp3_day, lrp3_km, lrp3_type, lrp3_visible),
        (lrp4_day, lrp4_km, lrp4_type, lrp4_visible),
    ]):
        if not visible:
            continue
        d = WEEKDAY_MAP.get(day_v) if day_v and day_v != "None" else None
        if d is not None:
            sessions.append({"day": d, "km": float(km_v or slot_defaults[i]), "type": type_v or "easy"})
    return sessions


def _profile_summary_html(profile: dict, schedule: dict, zones_data: dict) -> str:
    if not profile:
        return ""
    p = profile
    s = schedule
    run_days = ", ".join(WEEKDAY_LABELS[i][:3] for i in s.get("run_days", []))
    lrp_sessions = _load_lrp_sessions(s)
    lrp_rows = "".join(
        f"<div style='display:flex;gap:8px;margin-bottom:4px'>"
        f"<span style='background:#EFF6FF;color:#1D4ED8;padding:2px 8px;border-radius:99px;font-size:12px;font-weight:600'>"
        f"{WEEKDAY_LABELS[sess['day']][:3]}</span>"
        f"<span style='color:#374151;font-size:13px'>{sess['km']:.0f} km · {sess['type']}</span>"
        f"</div>"
        for sess in lrp_sessions if sess.get("day") is not None
    ) or "<span style='color:#9CA3AF;font-size:13px'>No club session configured</span>"

    strength_days = ", ".join(WEEKDAY_LABELS[i][:3] for i in s.get("strength_days", [])) or "—"
    cycling_days  = ", ".join(WEEKDAY_LABELS[i][:3] for i in s.get("cycling_days", []))  or "—"
    injury = p.get("injury_level", "none")
    injury_badge_color = {"none": "#10B981", "light": "#F5871F", "moderate": "#DC2626"}.get(injury, "#6B7280")

    vdot = zones_data.get("vdot", "—") if zones_data else "—"

    def row(label, value):
        return (f"<div style='display:flex;justify-content:space-between;padding:8px 0;"
                f"border-bottom:1px solid #F3F4F6'>"
                f"<span style='color:#6B7280;font-size:13px'>{label}</span>"
                f"<span style='color:#111827;font-size:13px;font-weight:500'>{value}</span>"
                f"</div>")

    inj_badge = (
        "<span style='background:" + injury_badge_color
        + ";color:white;padding:1px 8px;border-radius:99px;font-size:11px'>"
        + injury + "</span>"
    )
    return (
        "<div style='font-family:-apple-system,sans-serif;padding:0 4px'>"
        + row("Goal race",   p.get("goal_race", "—"))
        + row("Race date",   _iso_to_dmy(p.get("marathon_date", "")) or "—")
        + row("Target time", _fmt_duration(p.get("goal_time_s", 0)))
        + row("VDOT",        str(vdot))
        + row("Injury level", inj_badge)
        + row("Running days", run_days or "—")
        + row("Strength",    strength_days)
        + row("Cycling",     cycling_days)
        + "<div style='padding:8px 0'>"
          "<div style='color:#6B7280;font-size:13px;margin-bottom:6px'>LRP club sessions</div>"
        + lrp_rows
        + "</div></div>"
    )


def compute_and_generate(
    name, goal_race, marathon_date_str,
    goal_h, goal_m, goal_s,
    b1_dist, b1_h, b1_m, b1_s, b1_date_str,
    b2_dist, b2_h, b2_m, b2_s, b2_date_str,
    injury_level, injury_notes,
    run_days_labels,
    runs_per_week,
    allow_volume_increase,
    lrp1_day, lrp1_km, lrp1_type,
    lrp2_day, lrp2_km, lrp2_type, lrp2_visible,
    lrp3_day, lrp3_km, lrp3_type, lrp3_visible,
    lrp4_day, lrp4_km, lrp4_type, lrp4_visible,
    strength_labels, cycling_labels,
):
    goal_total = _parse_time(goal_h, goal_m, goal_s)
    if not goal_total:
        return {"error": "Invalid goal time"}, None, "Fix errors above.", ""

    try:
        marathon_date = datetime.strptime(marathon_date_str.strip(), "%d-%m-%Y").date()
    except Exception:
        return {"error": "Invalid date — use DD-MM-YYYY"}, None, "Fix errors above.", ""

    b1_time  = _parse_time(b1_h, b1_m, b1_s)
    b1_m_val = DISTANCES.get(b1_dist)
    if not b1_time or not b1_m_val:
        return {"error": "Invalid benchmark 1"}, None, "Fix errors above.", ""

    b1_date_iso = _dmy_to_iso(b1_date_str) if b1_date_str else ""
    b2_date_iso = _dmy_to_iso(b2_date_str) if b2_date_str else ""
    recency1 = vdot_recency_factor(b1_date_iso) if b1_date_iso else 1.0
    vdot     = vdot_from_race(b1_m_val, b1_time) * recency1
    b2_time  = _parse_time(b2_h, b2_m, b2_s)
    b2_m_val = DISTANCES.get(b2_dist)

    if b2_time and b2_m_val and b2_m_val != b1_m_val:
        recency2  = vdot_recency_factor(b2_date_iso) if b2_date_iso else 1.0
        vdot2     = vdot_from_race(b2_m_val, b2_time) * recency2
        vdot      = (vdot + vdot2) / 2
        efforts   = sorted([(b1_m_val, b1_time), (b2_m_val, b2_time)])
        cv        = cv_from_two_efforts(efforts[0][0], efforts[0][1], efforts[1][0], efforts[1][1])
        cv_source = "exact (Monod-Billat)"
    else:
        cv        = cv_from_vdot(vdot)
        cv_source = "estimated from VDOT"

    zones        = build_zones(vdot, cv)
    run_days     = sorted([WEEKDAY_MAP[d] for d in (run_days_labels or [])])
    lrp_sessions = _parse_lrp_sessions(
        lrp1_day, lrp1_km, lrp1_type,
        lrp2_day, lrp2_km, lrp2_type, lrp2_visible,
        lrp3_day, lrp3_km, lrp3_type, lrp3_visible,
        lrp4_day, lrp4_km, lrp4_type, lrp4_visible,
    )
    strength     = [WEEKDAY_MAP[d] for d in (strength_labels or [])]
    cycling      = [WEEKDAY_MAP[d] for d in (cycling_labels or [])]

    plan = generate_plan(
        marathon_date=marathon_date,
        goal_time_s=goal_total,
        zones=zones,
        run_days=run_days,
        lrp_sessions=lrp_sessions,
        strength_days=strength,
        cycling_days=cycling,
        injury=injury_level,
        runs_per_week=int(runs_per_week or 0),
        allow_volume_increase=bool(allow_volume_increase),
    )

    new_profile = {
        "name": name, "goal_race": goal_race,
        "marathon_date": str(marathon_date),
        "goal_time_s": goal_total,
        "injury_level": injury_level,
        "injury_notes": injury_notes,
        "b1_dist": b1_dist, "b1_time_s": b1_time, "b1_date": b1_date_iso or "",
        "b2_dist": b2_dist, "b2_time_s": b2_time or 0, "b2_date": b2_date_iso or "",
    }
    new_schedule = {
        "run_days": run_days,
        "runs_per_week": int(runs_per_week or 0),
        "allow_volume_increase": bool(allow_volume_increase),
        "lrp_sessions": lrp_sessions,
        "strength_days": strength,
        "cycling_days": cycling,
    }

    existing = state_mod.load()
    existing.update({
        "profile": new_profile,
        "zones": zones.__dict__,
        "schedule": new_schedule,
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
    recency_note = ""
    if b1_date_str and recency1 < 1.0:
        recency_note = f"  |  Benchmark discounted {round((1-recency1)*100):.0f}% (age)"
    msg = (f"Plan saved — {len(plan)} weeks to {goal_race}  |  "
           f"VDOT {zones.vdot}  |  SVC {zones.cv_mps * 3.6:.1f} km/h  |  "
           f"Target {_fmt_duration(goal_total)}{recency_note}")
    summary_html = _profile_summary_html(new_profile, new_schedule, zones.__dict__)
    return zs, _plan_to_html(existing["plan"]), msg, summary_html


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
    return _format_history_df(pd.DataFrame(merged)), msg


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
    _best_quality_dev = float("inf")
    hr_list, drift_list = [], []

    if files:
        for f in files:
            s = fit_mod.summarize(f.name)
            if not s or "error" in s:
                continue
            actual_km += s.get("distance_km", 0)
            if s.get("avg_hr"):
                hr_list.append(s["avg_hr"])
            if s.get("hr_drift_pct") is not None:
                drift_list.append(s["hr_drift_pct"])
            if quality_target_s and s.get("avg_pace_s") and s.get("distance_km", 0) <= 16:
                dev = abs(s["avg_pace_s"] - quality_target_s)
                if dev <= 60 and dev < _best_quality_dev:
                    quality_pace_s = s["avg_pace_s"]
                    _best_quality_dev = dev

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

    updated_zones = infer_vdot_adjustment(Zones(**zones_data), state.get("history", []))
    zones_update_note = ""
    if updated_zones is not None:
        delta = round(updated_zones.vdot - zones_data.get("vdot", 0), 1)
        direction = "↑" if delta > 0 else "↓"
        zones_update_note = f"VDOT {direction}{abs(delta):.1f} → {updated_zones.vdot} (auto-updated from recent runs)"
        state["zones"] = updated_zones.__dict__
        zones_data = updated_zones.__dict__

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
    if zones_update_note:
        assessment["Zones updated"] = zones_update_note

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


# ── Garmin Connect UI helpers ─────────────────────────────────────────────

def _garmin_status_html(connected: bool, email: str | None) -> str:
    if connected:
        return (
            "<div style='display:inline-flex;align-items:center;gap:8px;"
            "background:#DCFCE7;border:1px solid #86EFAC;border-radius:8px;"
            "padding:8px 14px;font-size:13px;font-family:-apple-system,sans-serif'>"
            "<span style='color:#16A34A;font-size:14px'>●</span>"
            f"<span style='color:#166534;font-weight:500'>Connected as {email}</span>"
            "</div>"
        )
    return (
        "<div style='display:inline-flex;align-items:center;gap:8px;"
        "background:#FEF2F2;border:1px solid #FCA5A5;border-radius:8px;"
        "padding:8px 14px;font-size:13px;font-family:-apple-system,sans-serif'>"
        "<span style='color:#DC2626;font-size:14px'>○</span>"
        "<span style='color:#991B1B;font-weight:500'>Not connected</span>"
        "</div>"
    )


def load_garmin_ui():
    status = garmin_mod.connection_status()
    connected = status != "not_connected"
    email = status.replace("connected:", "") if connected else None
    s = state_mod.load()
    auto_sync = s.get("garmin_auto_sync", False)
    next_run = sched_mod.next_run_str() if auto_sync else ""
    return (
        _garmin_status_html(connected, email),
        gr.update(visible=not connected),
        gr.update(visible=False),
        gr.update(visible=connected),
        auto_sync,
        f"<span style='font-size:12px;color:#6B7280'>Next sync: {next_run}</span>" if next_run else "",
    )


def _garmin_ui_outputs(connected, email, auto_sync=False, next_run="", msg=""):
    return (
        _garmin_status_html(connected, email),
        gr.update(visible=not connected),
        gr.update(visible=False),
        gr.update(visible=connected),
        auto_sync,
        f"<span style='font-size:12px;color:#6B7280'>Next sync: {next_run}</span>" if next_run else "",
        msg,
    )


def garmin_connect_ui(email, password):
    if not email or not password:
        return _garmin_ui_outputs(False, None, msg="Enter email and password.")
    ok, msg = garmin_mod.connect(email, password)
    if ok:
        s = state_mod.load()
        auto_sync = s.get("garmin_auto_sync", False)
        return _garmin_ui_outputs(True, email, auto_sync=auto_sync, msg=msg)
    if msg == "MFA_REQUIRED":
        return (
            _garmin_status_html(False, None),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
            False,
            "",
            "Check your email — enter the verification code below.",
        )
    return _garmin_ui_outputs(False, None, msg=msg)


def garmin_mfa_ui(email, password, mfa_code):
    ok, msg = garmin_mod.submit_mfa(email, password, mfa_code)
    if ok:
        return _garmin_ui_outputs(True, email, msg=msg)
    return (
        _garmin_status_html(False, None),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        False,
        "",
        msg,
    )


def garmin_disconnect_ui():
    garmin_mod.clear_auth()
    sched_mod.set_enabled(False)
    s = state_mod.load()
    s["garmin_auto_sync"] = False
    state_mod.save(s)
    return _garmin_ui_outputs(False, None, msg="Disconnected.")


def _merge_into_history(records: list) -> tuple[pd.DataFrame, str]:
    if not records:
        return load_history_on_start(), ""
    existing = state_mod.load().get("history", [])
    existing_dates = {r["date"] for r in existing}
    added = [r for r in records if r["date"] not in existing_dates]
    merged = sorted(existing + added, key=lambda r: r["date"], reverse=True)
    state_mod.update("history", merged)
    return _format_history_df(pd.DataFrame(merged)), f"{len(added)} new run{'s' if len(added) != 1 else ''} added ({len(merged)} total)"


def garmin_import_ui():
    records, sync_msg = garmin_mod.sync_activities(days=30)
    df, merge_msg = _merge_into_history(records)
    return df, f"{sync_msg}{' — ' + merge_msg if merge_msg else ''}"


def garmin_sync_new_ui():
    records, sync_msg = garmin_mod.sync_activities(days=3)
    df, merge_msg = _merge_into_history(records)
    return df, f"{sync_msg}{' — ' + merge_msg if merge_msg else ''}"


def garmin_autosync_toggle(enabled):
    def _job():
        garmin_sync_new_ui()

    msg = sched_mod.set_enabled(enabled, _job if enabled else None)
    s = state_mod.load()
    s["garmin_auto_sync"] = enabled
    state_mod.save(s)
    next_run = sched_mod.next_run_str() if enabled else ""
    return (
        f"<span style='font-size:12px;color:#6B7280'>Next sync: {next_run}</span>" if next_run else "",
        msg,
    )


# ── Weekly check-in from synced history ────────────────────────────────────

def checkin_from_history(week_num, feeling, prev_hr_input):
    s = state_mod.load()
    plan = s.get("plan", [])
    profile = s.get("profile", {})
    zones_data = s.get("zones", {})
    history = s.get("history", [])

    if not plan:
        return None, "No plan found — go to Setup & Plan first.", pd.DataFrame()

    wk_idx = int(week_num) - 1
    if wk_idx < 0 or wk_idx >= len(plan):
        return None, f"Week {week_num} not in plan ({len(plan)} weeks total).", pd.DataFrame()

    week = plan[wk_idx]
    week_dates = {d["date"] for d in week["days"]}
    week_records = [r for r in history if r.get("date") in week_dates]

    if not week_records:
        return (
            None,
            f"No synced runs found for week {week_num}. Upload FIT files manually or sync from Garmin first.",
            pd.DataFrame(),
        )

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
    _best_quality_dev = float("inf")
    hr_list, drift_list = [], []

    for rec in week_records:
        actual_km += rec.get("distance_km", 0)
        if rec.get("avg_hr"):
            hr_list.append(rec["avg_hr"])
        if rec.get("hr_drift_pct") is not None:
            drift_list.append(rec["hr_drift_pct"])
        if quality_target_s and rec.get("avg_pace_s") and rec.get("distance_km", 0) <= 16:
            dev = abs(rec["avg_pace_s"] - quality_target_s)
            if dev <= 60 and dev < _best_quality_dev:
                quality_pace_s = rec["avg_pace_s"]
                _best_quality_dev = dev

    metrics = WeekMetrics(
        actual_km=actual_km, target_km=target_km,
        quality_avg_pace_s=quality_pace_s, quality_target_pace_s=quality_target_s,
        easy_avg_hr=sum(hr_list) / len(hr_list) if hr_list else None,
        prev_easy_hr=float(prev_hr_input) if prev_hr_input else None,
        avg_hr_drift=sum(drift_list) / len(drift_list) if drift_list else None,
        avg_feeling=float(feeling),
    )
    result = score_week(metrics)

    s["plan"] = _apply_checkin_adaptation(plan, wk_idx, result, zones_data)

    updated_zones = infer_vdot_adjustment(Zones(**zones_data), history)
    zones_update_note = ""
    if updated_zones is not None:
        delta = round(updated_zones.vdot - zones_data.get("vdot", 0), 1)
        direction = "↑" if delta > 0 else "↓"
        zones_update_note = f"VDOT {direction}{abs(delta):.1f} → {updated_zones.vdot} (auto-updated)"
        s["zones"] = updated_zones.__dict__
        zones_data = updated_zones.__dict__

    state_mod.save(s)

    assessment = {
        "Performance score": f"{result.score} / 100",
        "Decision": result.decision,
        "Volume": f"{actual_km:.1f} km  (target {target_km:.1f} km)",
        "Quality session pace": (f"{fmt_pace(quality_pace_s)}  (target {fmt_pace(quality_target_s)})"
                                  if quality_pace_s else "not detected"),
        "Source": f"{len(week_records)} synced run{'s' if len(week_records) != 1 else ''}",
        "Signals": result.flags,
        "Next week volume": f"×{result.volume_adj:.2f}",
        "Quality session": "→ replaced with easy run" if result.drop_quality else "kept as planned",
        "Extra recovery": "yes" if result.add_recovery else "no",
    }
    if zones_update_note:
        assessment["Zones updated"] = zones_update_note

    z = Zones(**zones_data)
    weeks_left = len(plan) - wk_idx
    next_focus = plan[wk_idx + 1]["focus"] if wk_idx + 1 < len(plan) else "Race week — stay calm"

    note = llm_mod.coaching_note(llm_mod.build_context(
        name=profile.get("name", "Athlete"),
        goal_race=profile.get("goal_race", "your marathon"),
        goal_time=_fmt_duration(profile.get("goal_time_s", 0)),
        weeks_left=weeks_left, phase=week["phase"],
        zones_dict=zones_summary(z),
        metrics=metrics, result=result, next_focus=next_focus,
    ))

    return assessment, note, _plan_to_df(s["plan"])


# ── CSS ────────────────────────────────────────────────────────────────────

CSS = """
:root {
    --lrp-navy:   #1B2874;
    --lrp-blue:   #3B82F6;
    --lrp-orange: #F5871F;
    --lrp-bg:     #F0F2F8;
    --lrp-white:  #FFFFFF;
    --lrp-text:   #111827;

    /* Fix: primary_hue bleeds into block label tabs — neutralise to grey */
    --block-label-background-fill: #F3F4F6;
    --block-label-text-color: #374151;
    --block-title-background-fill: #F3F4F6;
    --block-title-text-color: #374151;
    --block-info-text-color: #6B7280;

    /* Fix: Soft theme input focus defaults to indigo secondary */
    --input-background-fill-focus: #EFF6FF;
    --input-border-color-focus: #3B82F6;

    /* Make ALL group/form containers white — eliminates nested grey cards */
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #ffffff;
    --panel-background-fill: #ffffff;

    /* Add subtle border so form fields are visible on white background */
    --block-border-width: 1px;
    --block-border-color: #E5E7EB;
    --block-shadow: 0 1px 3px rgba(0,0,0,0.04);

    /* Clean JSON / code block backgrounds */
    --code-background-fill: #F8FAFC;

    /* Override dark-mode body-text-color that wins in theme cascade */
    --body-text-color: #111827;
    --body-text-color-subdued: #6B7280;
}

footer { display: none !important; }

body, .gradio-container {
    background: var(--lrp-bg) !important;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif !important;
}

/* ── App-level row: sidebar + content side by side ── */
#app-layout {
    gap: 0 !important;
    align-items: stretch !important;
    padding: 0 !important;
    min-height: 100vh;
    flex-wrap: nowrap !important;
}

/* ── Sidebar column ──────────────────────────────── */
#sidebar {
    background: var(--lrp-navy) !important;
    min-width: 220px !important;
    max-width: 220px !important;
    flex-shrink: 0 !important;
    padding: 0 !important;
    gap: 0 !important;
    border-radius: 0 !important;
    overflow: hidden;
}

#sidebar > .form,
#sidebar > div,
#sidebar > .gap {
    padding: 0 !important;
    gap: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Sidebar logo block */
#sidebar-logo {
    padding: 20px 16px 16px;
    display: flex !important;
    align-items: center;
    gap: 11px;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}

#sidebar-logo img {
    width: 40px;
    height: 40px;
    border-radius: 9px;
    flex-shrink: 0;
    object-fit: cover;
}

.brand-name {
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.01em;
    line-height: 1.2;
}

.brand-sub {
    color: rgba(255,255,255,0.48);
    font-size: 10px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 1px;
}

/* ── Nav buttons ─────────────────────────────────── */
.nav-btn {
    display: block !important;
    width: calc(100% - 16px) !important;
    margin: 2px 8px !important;
    text-align: left !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    border: none !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    box-shadow: none !important;
    min-height: unset !important;
}

.nav-btn.secondary,
.nav-btn.secondary:focus {
    background: transparent !important;
    color: rgba(255,255,255,0.70) !important;
}

.nav-btn.secondary:hover {
    background: rgba(255,255,255,0.09) !important;
    color: #ffffff !important;
}

.nav-btn.primary,
.nav-btn.primary:focus {
    background: var(--lrp-orange) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-color: transparent !important;
}

.nav-btn.primary:hover {
    background: #df7318 !important;
}

/* First nav button: add top margin for breathing room */
#nav-plan { margin-top: 12px !important; }

/* ── Strip Group / Row container wrappers — no grey cards ───────── */
#panel-setup .form, #panel-setup .gap,
#panel-history .form, #panel-history .gap,
#panel-adj .form, #panel-adj .gap,
#panel-checkin .form, #panel-checkin .gap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Form-field blocks — subtle white card (non-table panels) */
#panel-setup .block,
#panel-history .block:not(.gradio-dataframe),
#panel-adj .block:not(.gradio-dataframe),
#panel-checkin .block:not(.gradio-dataframe) {
    background: white !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    border-radius: var(--block-radius) !important;
}

/* Dataframe outer block — let the table-wrap carry the border */
#panel-history .gradio-dataframe,
#panel-adj .gradio-dataframe,
#panel-checkin .gradio-dataframe {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── ALL gr.Dataframe tables: orange header, white rows ──────────── */
#panel-history .table-wrap,
#panel-adj .table-wrap,
#panel-checkin .table-wrap {
    border-radius: 12px;
    border: 1px solid #F5871F !important;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

#panel-history table,
#panel-adj table,
#panel-checkin table {
    background: white !important;
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
}

/* Header: orange background, white text */
#panel-history thead th,
#panel-adj thead th,
#panel-checkin thead th {
    background: #F5871F !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 12px 10px !important;
    border: none !important;
    text-align: left !important;
}

/* Odd rows: white */
#panel-history tbody tr:nth-child(odd),
#panel-adj tbody tr:nth-child(odd),
#panel-checkin tbody tr:nth-child(odd) {
    background: white !important;
}

/* Even rows: very light grey */
#panel-history tbody tr:nth-child(even),
#panel-adj tbody tr:nth-child(even),
#panel-checkin tbody tr:nth-child(even) {
    background: #FAFAFA !important;
}

/* Row separator */
#panel-history tbody tr,
#panel-adj tbody tr,
#panel-checkin tbody tr {
    border-bottom: 1px solid #F3F4F6 !important;
}

/* Cells: dark readable text, no extra background */
#panel-history tbody td,
#panel-adj tbody td,
#panel-checkin tbody td {
    color: #374151 !important;
    padding: 9px 10px !important;
    font-size: 13px !important;
    background: transparent !important;
    border: none !important;
}

/* ── File upload drop zone — ALL text LRP blue ───────────────────── */
/* Targets "Drop File(s) Here", "-or-", and "Click to Upload" */
#content-area .file-preview-holder,
#content-area .file-preview-holder *,
#content-area .upload-container,
#content-area .upload-container *,
#content-area .empty p,
#content-area .empty span,
#content-area .grey {
    color: var(--lrp-blue) !important;
}

/* Upload button inside the drop zone */
#content-area .empty button,
#content-area .upload-container button {
    color: var(--lrp-blue) !important;
    background: transparent !important;
    border: 1px solid var(--lrp-blue) !important;
}

/* ── Content area ────────────────────────────────── */
#content-area {
    flex: 1 1 0% !important;
    min-width: 0 !important;
    padding: 28px 32px !important;
    background: var(--lrp-bg) !important;
    border-radius: 0 !important;
    overflow-y: auto;
}

#content-area > .form,
#content-area > div,
#content-area > .gap {
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Status bar ──────────────────────────────────── */
#status-bar { margin-bottom: 8px !important; }

/* Strip the gr.HTML block wrapper so the navy div sits flush */
#status-bar .block {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* ── Panel containers — white cards ─────────────── */
#panel-plan, #panel-setup, #panel-history,
#panel-adj, #panel-checkin {
    background: #ffffff !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04) !important;
    border: 1px solid rgba(0,0,0,0.05) !important;
    padding: 0 0 28px 0 !important;
    gap: 0 !important;
    overflow: hidden;
}

/* Inner wrappers */
#panel-plan > div, #panel-setup > div,
#panel-history > div, #panel-adj > div, #panel-checkin > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 28px !important;
    gap: 12px !important;
}

/* ── Page headers — clean modern style ───────────── */
.page-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 22px 0 18px 0;
    border-bottom: 1px solid #F3F4F6;
    margin-bottom: 24px;
    position: relative;
}

.page-header::before {
    content: '';
    display: block;
    width: 4px;
    min-height: 42px;
    background: linear-gradient(180deg, #1B2874 0%, #3B82F6 100%);
    border-radius: 2px;
    flex-shrink: 0;
}

.page-header-icon {
    font-size: 22px;
    line-height: 1;
}

.page-header-title {
    font-size: 18px;
    font-weight: 700;
    color: #1B2874;
    line-height: 1.2;
}

.page-header-sub {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 3px;
    font-weight: 400;
}

/* ── Orange-accent section dividers ─────────────── */
.section-label {
    border-left: 3px solid var(--lrp-orange);
    padding: 5px 0 5px 12px;
    margin: 22px 0 10px;
}

.section-label-text {
    font-size: 11px;
    font-weight: 700;
    color: var(--lrp-navy);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.section-label-sub {
    font-size: 12px;
    color: #9CA3AF;
    margin-top: 2px;
}

/* Block label text (catches the "FIT files" tab label) */
#content-area .block > span:first-child,
#content-area .block .block-info {
    color: #374151 !important;
}

/* ── Prose inside panels — keep it readable ──────── */
#content-area .prose p {
    color: #374151 !important;
    font-size: 13px !important;
    margin: 0 0 8px 0 !important;
}

/* ── Form element labels — force dark text, not subdued grey ─ */
#content-area .block label,
#content-area .block label span,
#content-area .block .wrap span,
#content-area .block .head span {
    color: #374151 !important;
}

/* ── Primary action buttons (in content) ─────────── */
#content-area button.lg.primary,
#content-area button.primary {
    background: var(--lrp-orange) !important;
    border-color: var(--lrp-orange) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}

#content-area button.lg.primary:hover,
#content-area button.primary:hover {
    background: #df7318 !important;
    border-color: #df7318 !important;
}

#content-area button.stop {
    background: #DC2626 !important;
    border-color: #DC2626 !important;
    color: #ffffff !important;
}
"""

# ── UI ─────────────────────────────────────────────────────────────────────

_NAV_COUNT = 5

def _nav_handler(active_i):
    """Return a click handler that shows panel[active_i] and highlights its button."""
    def handler():
        panels  = [gr.update(visible=(i == active_i)) for i in range(_NAV_COUNT)]
        buttons = [gr.update(variant=("primary" if i == active_i else "secondary"))
                   for i in range(_NAV_COUNT)]
        return panels + buttons
    return handler


_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#FFF7ED", c100="#FFEDD5", c200="#FED7AA",
        c300="#FDBA74", c400="#FB923C", c500="#F5871F",
        c600="#EA7615", c700="#C2620F", c800="#9A4D0B",
        c900="#7C3D08", c950="#431D03",
    ),
    neutral_hue=gr.themes.Color(
        c50="#F8FAFC", c100="#F1F5F9", c200="#E2E8F0",
        c300="#CBD5E1", c400="#94A3B8", c500="#64748B",
        c600="#475569", c700="#334155", c800="#1E293B",
        c900="#0F172A", c950="#020617",
    ),
)

def _daily_sync_job():
    garmin_sync_new_ui()

_startup = state_mod.load()
if _startup.get("garmin_auto_sync", False) and garmin_mod.is_authenticated():
    sched_mod.start(_daily_sync_job)

with gr.Blocks(title="LRP Coach", css=CSS, theme=_theme) as demo:

    with gr.Row(elem_id="app-layout", equal_height=True):

        # ── Sidebar ───────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
            gr.HTML("""
                <div id="sidebar-logo">
                    <img src="/file=scripts/lrp_icon.png" alt="LRP">
                    <div>
                        <div class="brand-name">LRP Coach</div>
                        <div class="brand-sub">Marathon Training</div>
                    </div>
                </div>
            """)
            btn_plan    = gr.Button("📋  My Plan",          variant="primary",   elem_classes="nav-btn", elem_id="nav-plan")
            btn_setup   = gr.Button("⚙️  Setup & Plan",     variant="secondary", elem_classes="nav-btn")
            btn_hist    = gr.Button("🏃  Run History",      variant="secondary", elem_classes="nav-btn")
            btn_adj     = gr.Button("✏️  Adjustments",      variant="secondary", elem_classes="nav-btn")
            btn_checkin = gr.Button("✅  Weekly Check-in",  variant="secondary", elem_classes="nav-btn")

        # ── Content area ──────────────────────────────────────────────────
        with gr.Column(scale=5, elem_id="content-area"):
            status_bar = gr.HTML("", elem_id="status-bar")

            # Panel 0 — My Plan (default)
            with gr.Group(visible=True, elem_id="panel-plan") as panel_plan:
                gr.HTML("""
                    <div class="page-header">
                        <span class="page-header-icon">📋</span>
                        <div>
                            <div class="page-header-title">My Plan</div>
                            <div class="page-header-sub">Loads automatically · updates after every check-in or adjustment</div>
                        </div>
                    </div>
                """)
                plan_df = gr.HTML()

            # Panel 1 — Setup & Plan
            with gr.Group(visible=False, elem_id="panel-setup") as panel_setup:
                gr.HTML("""
                    <div class="page-header">
                        <span class="page-header-icon">⚙️</span>
                        <div>
                            <div class="page-header-title">Setup & Plan</div>
                            <div class="page-header-sub">Your profile · edit to update and regenerate</div>
                        </div>
                    </div>
                """)

                # ── Static summary view (shown when plan exists) ──────────
                _init_summary = _profile_summary_html(
                    _prof, _sched, _saved.get("zones", {})
                ) if _has_plan else ""

                with gr.Group(visible=_has_plan, elem_id="setup-summary") as setup_summary:
                    profile_summary_html = gr.HTML(_init_summary)
                    edit_btn = gr.Button("✏️  Edit profile & regenerate plan",
                                         variant="secondary", size="sm")

                # ── Edit form (shown when no plan, or after Edit clicked) ──
                with gr.Group(visible=not _has_plan, elem_id="setup-form") as setup_form:

                    gr.HTML('<div class="section-label"><div class="section-label-text">Profile & Goals</div></div>')
                    with gr.Row():
                        name_in          = gr.Textbox(label="Your name", value=_prof.get("name", "Cecilia"))
                        goal_race_in     = gr.Textbox(label="Target race", value=_prof.get("goal_race", ""),
                                                      placeholder="Paris Marathon 2027")
                        marathon_date_in = gr.Textbox(label="Race date (DD-MM-YYYY)",
                                                      value=_iso_to_dmy(_prof.get("marathon_date", "")),
                                                      placeholder="11-04-2027")

                    gr.HTML('<div class="section-label"><div class="section-label-text">Target finish time</div></div>')
                    with gr.Row():
                        goal_h = gr.Number(label="Hours",   value=_g_h, precision=0, minimum=2, maximum=7)
                        goal_m = gr.Number(label="Minutes", value=_g_m, precision=0, minimum=0, maximum=59)
                        goal_s = gr.Number(label="Seconds", value=_g_s, precision=0, minimum=0, maximum=59)

                    gr.HTML('<div class="section-label"><div class="section-label-text">Benchmark 1 — required</div><div class="section-label-sub">Recent race or time trial · older results are slightly discounted</div></div>')
                    with gr.Row():
                        b1_dist    = gr.Dropdown(DIST_KEYS, label="Distance", value=_b1_dist_saved)
                        b1_h       = gr.Number(label="h",   value=_b1_h, precision=0, minimum=0, maximum=5)
                        b1_m       = gr.Number(label="min", value=_b1_m, precision=0, minimum=0, maximum=59)
                        b1_s       = gr.Number(label="sec", value=_b1_s, precision=0, minimum=0, maximum=59)
                        b1_date_in = gr.Textbox(label="Date (DD-MM-YYYY)", value=_b1_date_saved,
                                                placeholder="15-03-2025", scale=2)

                    gr.HTML('<div class="section-label"><div class="section-label-text">Benchmark 2 — optional</div><div class="section-label-sub">Different distance → exact SVC via Monod-Billat. Leave zero if only one result.</div></div>')
                    with gr.Row():
                        b2_dist    = gr.Dropdown(DIST_KEYS, label="Distance", value=_b2_dist_saved)
                        b2_h       = gr.Number(label="h",   value=_b2_h, precision=0, minimum=0, maximum=5)
                        b2_m       = gr.Number(label="min", value=_b2_m, precision=0, minimum=0, maximum=59)
                        b2_s       = gr.Number(label="sec", value=_b2_s, precision=0, minimum=0, maximum=59)
                        b2_date_in = gr.Textbox(label="Date (DD-MM-YYYY)", value=_b2_date_saved,
                                                placeholder="10-01-2025", scale=2)

                    gr.HTML('<div class="section-label"><div class="section-label-text">Physical status</div></div>')
                    with gr.Row():
                        injury_in       = gr.Radio(
                            ["none", "light", "moderate"], label="Rehab / injury level",
                            value=_prof.get("injury_level", "none"),
                            info="none = full training  ·  light = reduced intensity  ·  moderate = significant restriction",
                        )
                        injury_notes_in = gr.Textbox(label="Notes", value=_prof.get("injury_notes", ""), lines=2)

                    gr.HTML('<div class="section-label"><div class="section-label-text">Weekly running days</div><div class="section-label-sub">Tick the days you are available to run</div></div>')
                    run_days_in = gr.CheckboxGroup(
                        WEEKDAY_LABELS, label="",
                        value=_saved_run_days,
                    )
                    with gr.Row():
                        runs_per_week_in = gr.Number(
                            label="Runs per week (0 = use all available days)",
                            value=_sched.get("runs_per_week", 0),
                            precision=0, minimum=0, maximum=7,
                            info="Limit actual sessions if you want fewer runs than available days",
                        )
                        allow_volume_increase_in = gr.Checkbox(
                            label="Increase weekly volume during Build & Peak phases",
                            value=_sched.get("allow_volume_increase", True),
                        )

                    gr.HTML('<div class="section-label"><div class="section-label-text">Club / group run sessions</div><div class="section-label-sub">Locked into the plan · up to 4 sessions · not all need to be LRP</div></div>')

                    # Session 1 (always shown)
                    with gr.Row():
                        lrp1_day_in  = gr.Dropdown(["None"] + WEEKDAY_LABELS, label="Session 1 — day",
                                                    value=_lrp1_day)
                        lrp1_km_in   = gr.Number(label="km", value=_lrp1_km, minimum=3, maximum=35)
                        lrp1_type_in = gr.Radio(["easy", "tempo", "long"], label="Type",
                                                 value=_lrp1_type)

                    # Sessions 2–4 (chained toggles — 3 and 4 live inside 2's group)
                    lrp2_visible_state = gr.State(value=_has_lrp2)
                    lrp3_visible_state = gr.State(value=_has_lrp3)
                    lrp4_visible_state = gr.State(value=_has_lrp4)

                    with gr.Group(visible=_has_lrp2, elem_id="lrp-session2") as lrp2_group:
                        with gr.Row():
                            lrp2_day_in  = gr.Dropdown(["None"] + WEEKDAY_LABELS, label="Session 2 — day",
                                                        value=_lrp2_day)
                            lrp2_km_in   = gr.Number(label="km", value=_lrp2_km, minimum=3, maximum=35)
                            lrp2_type_in = gr.Radio(["easy", "tempo", "long"], label="Type",
                                                     value=_lrp2_type)
                        remove_lrp2_btn = gr.Button("✕ Remove session 2", size="sm", variant="secondary")

                        with gr.Group(visible=_has_lrp3, elem_id="lrp-session3") as lrp3_group:
                            with gr.Row():
                                lrp3_day_in  = gr.Dropdown(["None"] + WEEKDAY_LABELS, label="Session 3 — day",
                                                            value=_lrp3_day)
                                lrp3_km_in   = gr.Number(label="km", value=_lrp3_km, minimum=3, maximum=35)
                                lrp3_type_in = gr.Radio(["easy", "tempo", "long"], label="Type",
                                                         value=_lrp3_type)
                            remove_lrp3_btn = gr.Button("✕ Remove session 3", size="sm", variant="secondary")

                            with gr.Group(visible=_has_lrp4, elem_id="lrp-session4") as lrp4_group:
                                with gr.Row():
                                    lrp4_day_in  = gr.Dropdown(["None"] + WEEKDAY_LABELS, label="Session 4 — day",
                                                                value=_lrp4_day)
                                    lrp4_km_in   = gr.Number(label="km", value=_lrp4_km, minimum=3, maximum=35)
                                    lrp4_type_in = gr.Radio(["easy", "tempo", "long"], label="Type",
                                                             value=_lrp4_type)
                                remove_lrp4_btn = gr.Button("✕ Remove session 4", size="sm", variant="secondary")

                            add_lrp4_btn = gr.Button("+ Add 4th session", size="sm",
                                                      variant="secondary", visible=not _has_lrp4)

                        add_lrp3_btn = gr.Button("+ Add 3rd session", size="sm",
                                                  variant="secondary", visible=not _has_lrp3)

                    add_lrp2_btn = gr.Button("+ Add 2nd session", size="sm",
                                              variant="secondary", visible=not _has_lrp2,
                                              elem_id="add-lrp2-btn")

                    gr.HTML('<div class="section-label"><div class="section-label-text">Cross-training</div></div>')
                    with gr.Row():
                        strength_in = gr.CheckboxGroup(WEEKDAY_LABELS, label="Strength days",
                                                        value=_saved_strength)
                        cycling_in  = gr.CheckboxGroup(WEEKDAY_LABELS, label="Cycling / Zwift days",
                                                        value=_saved_cycling)

                    with gr.Row():
                        cancel_edit_btn = gr.Button("Cancel",
                                                     variant="secondary", visible=_has_plan)
                        gen_btn = gr.Button("Save & Generate Plan", variant="primary", size="lg")
                    gen_msg   = gr.Textbox(label="Status", interactive=False)
                    zones_out = gr.JSON(label="Training Zones")

            # Panel 2 — Run History
            with gr.Group(visible=False, elem_id="panel-history") as panel_history:
                gr.HTML("""
                    <div class="page-header">
                        <span class="page-header-icon">🏃</span>
                        <div>
                            <div class="page-header-title">Run History</div>
                            <div class="page-header-sub">Upload .fit files · duplicates ignored · metrics: pace, HR, drift, cadence, elevation</div>
                        </div>
                    </div>
                """)
                with gr.Row(equal_height=False):
                    # Left column — manual FIT upload
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-label"><div class="section-label-text">Manual upload</div></div>')
                        hist_files = gr.File(file_count="multiple", file_types=[".fit"], label="FIT files")
                        with gr.Row():
                            hist_btn       = gr.Button("Add to history", variant="primary")
                            hist_clear_btn = gr.Button("Clear all history", variant="stop")
                        hist_msg = gr.Textbox(label="Status", interactive=False)

                    # Right column — Garmin Connect
                    with gr.Column(scale=1):
                        gr.HTML('<div class="section-label"><div class="section-label-text">Garmin Connect</div><div class="section-label-sub">Sync automatically · password never stored</div></div>')
                        garmin_status = gr.HTML(_garmin_status_html(False, None))
                        with gr.Group(visible=True, elem_id="garmin-connect-form") as garmin_form:
                            garmin_email_in = gr.Textbox(label="Garmin email", placeholder="you@example.com")
                            garmin_pass_in  = gr.Textbox(label="Password", type="password")
                            garmin_conn_btn = gr.Button("Connect", variant="primary")
                        with gr.Group(visible=False, elem_id="garmin-mfa-row") as garmin_mfa_group:
                            garmin_mfa_in  = gr.Textbox(label="Verification code", placeholder="6-digit code from your email")
                            garmin_mfa_btn = gr.Button("Submit code", variant="primary")
                        with gr.Group(visible=False, elem_id="garmin-synced") as garmin_synced_group:
                            with gr.Row():
                                garmin_import_btn     = gr.Button("Import last 30 days", variant="primary")
                                garmin_sync_btn       = gr.Button("Sync new (3 days)", variant="secondary")
                            garmin_autosync_in  = gr.Checkbox(label="Auto-sync daily at 07:00", value=False)
                            garmin_next_run_out = gr.HTML("")
                            garmin_disconnect_btn = gr.Button("Disconnect", variant="stop", size="sm")
                        garmin_msg = gr.Textbox(label="Status", interactive=False)

                gr.HTML('<div class="section-label"><div class="section-label-text">Run log</div></div>')
                hist_df  = gr.Dataframe(label="", wrap=True)

                _garmin_outputs = [garmin_status, garmin_form, garmin_mfa_group,
                                   garmin_synced_group, garmin_autosync_in, garmin_next_run_out]

                hist_btn.click(process_history, inputs=hist_files, outputs=[hist_df, hist_msg])
                hist_clear_btn.click(clear_history, outputs=[hist_df, hist_msg])
                garmin_conn_btn.click(
                    garmin_connect_ui,
                    inputs=[garmin_email_in, garmin_pass_in],
                    outputs=_garmin_outputs + [garmin_msg],
                )
                garmin_mfa_btn.click(
                    garmin_mfa_ui,
                    inputs=[garmin_email_in, garmin_pass_in, garmin_mfa_in],
                    outputs=_garmin_outputs + [garmin_msg],
                )
                garmin_disconnect_btn.click(
                    garmin_disconnect_ui,
                    outputs=_garmin_outputs + [garmin_msg],
                )
                garmin_import_btn.click(garmin_import_ui, outputs=[hist_df, garmin_msg])
                garmin_sync_btn.click(garmin_sync_new_ui, outputs=[hist_df, garmin_msg])
                garmin_autosync_in.change(
                    garmin_autosync_toggle,
                    inputs=[garmin_autosync_in],
                    outputs=[garmin_next_run_out, garmin_msg],
                )

            # Panel 3 — Adjustments
            with gr.Group(visible=False, elem_id="panel-adj") as panel_adj:
                gr.HTML("""
                    <div class="page-header">
                        <span class="page-header-icon">✏️</span>
                        <div>
                            <div class="page-header-title">Adjustments</div>
                            <div class="page-header-sub">Physio, travel, illness — describe it and the plan updates immediately</div>
                        </div>
                    </div>
                """)
                gr.HTML('<div class="section-label"><div class="section-label-text">What\'s going on?</div></div>')
                adj_message_in = gr.Textbox(
                    label="",
                    placeholder="e.g. Still doing physio for my knee — skipping LRP for 3 weeks, easy runs only.",
                    lines=3,
                )
                gr.HTML('<div class="section-label"><div class="section-label-text">Scope</div></div>')
                with gr.Row():
                    adj_from_in  = gr.Number(label="Starting from week #", value=1, precision=0, minimum=1)
                    adj_weeks_in = gr.Number(label="For how many weeks  (0 = rest of plan)",
                                              value=3, precision=0, minimum=0)

                gr.HTML('<div class="section-label"><div class="section-label-text">What to change</div></div>')
                with gr.Row():
                    adj_no_lrp_in    = gr.Checkbox(label="Skip LRP club runs → replace with easy")
                    adj_easy_only_in = gr.Checkbox(label="Easy runs only → remove all quality sessions")
                adj_volume_in = gr.Slider(60, 110, value=100, step=5, label="Volume (% of planned km)")

                adj_btn = gr.Button("Apply to plan & get coaching note", variant="primary")

                gr.HTML('<div class="section-label"><div class="section-label-text">Coach response</div></div>')
                adj_note_out    = gr.Textbox(label="", lines=10, interactive=False)
                gr.HTML('<div class="section-label"><div class="section-label-text">Changes applied</div></div>')
                adj_changes_out = gr.JSON(label="")
                gr.HTML('<div class="section-label"><div class="section-label-text">Updated plan</div></div>')
                adj_plan_out    = gr.Dataframe(label="", wrap=True)

                adj_btn.click(
                    apply_adjustments_ui,
                    inputs=[adj_message_in, adj_from_in, adj_weeks_in,
                            adj_no_lrp_in, adj_easy_only_in, adj_volume_in],
                    outputs=[adj_note_out, adj_changes_out, adj_plan_out],
                )

            # Panel 4 — Weekly Check-in
            with gr.Group(visible=False, elem_id="panel-checkin") as panel_checkin:
                gr.HTML("""
                    <div class="page-header">
                        <span class="page-header-icon">✅</span>
                        <div>
                            <div class="page-header-title">Weekly Check-in</div>
                            <div class="page-header-sub">Score pace · HR · volume · feeling → adapt next week · get your coaching note</div>
                        </div>
                    </div>
                """)
                gr.HTML('<div class="section-label"><div class="section-label-text">This week</div></div>')
                with gr.Row():
                    checkin_week = gr.Number(label="Plan week number", value=1, precision=0, minimum=1)
                    feeling_in   = gr.Slider(1, 5, value=3, step=0.5,
                                             label="Overall feeling  (1 = rough · 5 = excellent)")
                    prev_hr_in   = gr.Number(label="Last week avg easy HR (bpm, optional)", value=None)

                checkin_files = gr.File(file_count="multiple", file_types=[".fit"],
                                        label="FIT files for this week")
                checkin_btn = gr.Button("Analyse week & get coaching note", variant="primary")
                gr.HTML("<div style='text-align:center;color:#9CA3AF;font-size:12px;margin:2px 0'>— or —</div>")
                checkin_history_btn = gr.Button("Use synced Garmin history for this week", variant="secondary")

                gr.HTML('<div class="section-label"><div class="section-label-text">Performance assessment</div></div>')
                checkin_json = gr.JSON(label="")
                gr.HTML('<div class="section-label"><div class="section-label-text">Your coaching note</div></div>')
                coaching_out = gr.Textbox(label="", lines=12, interactive=False)
                gr.HTML('<div class="section-label"><div class="section-label-text">Updated plan</div></div>')
                checkin_plan_out = gr.Dataframe(label="", wrap=True)

                checkin_btn.click(
                    checkin,
                    inputs=[checkin_week, checkin_files, feeling_in, prev_hr_in],
                    outputs=[checkin_json, coaching_out, checkin_plan_out],
                )
                checkin_history_btn.click(
                    checkin_from_history,
                    inputs=[checkin_week, feeling_in, prev_hr_in],
                    outputs=[checkin_json, coaching_out, checkin_plan_out],
                )

    # ── Nav button wiring ──────────────────────────────────────────────────
    _panels = [panel_plan, panel_setup, panel_history, panel_adj, panel_checkin]
    _btns   = [btn_plan, btn_setup, btn_hist, btn_adj, btn_checkin]

    for _i, _btn in enumerate(_btns):
        _btn.click(_nav_handler(_i), outputs=_panels + _btns)

    # ── Setup: edit / cancel toggles ──────────────────────────────────────
    edit_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[setup_summary, setup_form],
    )
    cancel_edit_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[setup_summary, setup_form],
    )

    # ── Club session add / remove wiring ─────────────────────────────────────
    add_lrp2_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), True),
        outputs=[lrp2_group, add_lrp2_btn, lrp2_visible_state],
    )
    remove_lrp2_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), False,
                 gr.update(visible=False), gr.update(visible=True), False,
                 gr.update(visible=False), gr.update(visible=True), False),
        outputs=[lrp2_group, add_lrp2_btn, lrp2_visible_state,
                 lrp3_group, add_lrp3_btn, lrp3_visible_state,
                 lrp4_group, add_lrp4_btn, lrp4_visible_state],
    )
    add_lrp3_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), True),
        outputs=[lrp3_group, add_lrp3_btn, lrp3_visible_state],
    )
    remove_lrp3_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), False,
                 gr.update(visible=False), gr.update(visible=True), False),
        outputs=[lrp3_group, add_lrp3_btn, lrp3_visible_state,
                 lrp4_group, add_lrp4_btn, lrp4_visible_state],
    )
    add_lrp4_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), True),
        outputs=[lrp4_group, add_lrp4_btn, lrp4_visible_state],
    )
    remove_lrp4_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), False),
        outputs=[lrp4_group, add_lrp4_btn, lrp4_visible_state],
    )

    # ── Generate button ────────────────────────────────────────────────────
    def _generate_and_show_summary(*args):
        result = compute_and_generate(*args)
        zones_out_val, plan_html, msg, summary_html = result
        return (zones_out_val, plan_html, msg, summary_html,
                gr.update(visible=True), gr.update(visible=False),
                load_status_on_start())

    gen_btn.click(
        _generate_and_show_summary,
        inputs=[
            name_in, goal_race_in, marathon_date_in,
            goal_h, goal_m, goal_s,
            b1_dist, b1_h, b1_m, b1_s, b1_date_in,
            b2_dist, b2_h, b2_m, b2_s, b2_date_in,
            injury_in, injury_notes_in,
            run_days_in,
            runs_per_week_in, allow_volume_increase_in,
            lrp1_day_in, lrp1_km_in, lrp1_type_in,
            lrp2_day_in, lrp2_km_in, lrp2_type_in, lrp2_visible_state,
            lrp3_day_in, lrp3_km_in, lrp3_type_in, lrp3_visible_state,
            lrp4_day_in, lrp4_km_in, lrp4_type_in, lrp4_visible_state,
            strength_in, cycling_in,
        ],
        outputs=[zones_out, plan_df, gen_msg, profile_summary_html,
                 setup_summary, setup_form, status_bar],
    )

    # ── Load saved data on page open ──────────────────────────────────────
    demo.load(load_status_on_start,  outputs=status_bar)
    demo.load(load_plan_on_start,    outputs=plan_df)
    demo.load(load_history_on_start, outputs=hist_df)
    demo.load(load_garmin_ui,        outputs=_garmin_outputs)


if __name__ == "__main__":
    demo.queue().launch(allowed_paths=["scripts"])
