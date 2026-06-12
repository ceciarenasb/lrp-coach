"""LRP Coach — Marathon Training Assistant"""

from __future__ import annotations

VERSION = "1.0"

from datetime import date, datetime, timedelta

import gradio as gr
import pandas as pd

from coach import debrief as debrief_mod
from coach import editplan as editplan_mod
from coach import fit as fit_mod
from coach import garmin as garmin_mod
from coach import llm as llm_mod
from coach import scheduler as sched_mod
from coach import state as state_mod
from coach.adapt import WeekMetrics, score_week
from coach.decide import propose_week
from coach.load import compute as compute_load
from coach.plan import generate_plan
from coach.readiness import label as readiness_label
from coach.state import active_cycle, new_cycle, set_active_cycle
from coach.state_model import (
    apply_history, from_dict as sm_from_dict, to_dict as sm_to_dict,
)
from coach.zones import (
    Zones, build_zones, fmt_pace, hr_zones, infer_vdot_adjustment,
    marathon_time_from_vdot, pace_zones_extended, vdot_from_race,
    zones_summary,
)

# ── Constants ──────────────────────────────────────────────────────────────

DISTANCES = {"5 km": 5_000, "10 km": 10_000, "Half-marathon": 21_097, "Marathon": 42_195}
DIST_KEYS = list(DISTANCES.keys())
WEEKDAY_LABELS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_MAP    = {v: i for i, v in enumerate(WEEKDAY_LABELS)}

_SESSION_COLOR = {
    "Easy": "#10B981", "Recovery": "#34D399", "Long Run": "#2563EB",
    "Medium-Long": "#3B82F6", "Tempo": "#F5871F", "SVC Intervals": "#DC2626",
    "Marathon Pace": "#7C3AED", "Progression Run": "#8B5CF6",
    "Rest": "#9CA3AF", "Strength": "#D97706", "Cycling / Zwift": "#0891B2",
    "Club Run (LRP)": "#0D9488",
}
_PHASE_COLORS = {
    "Base": "#059669", "Build": "#2563EB", "Peak": "#EA580C", "Taper": "#64748B",
}
_RPE_COLOR = {
    "1–2": "#94A3B8", "3–4": "#10B981", "4–5": "#10B981",
    "6": "#3B82F6", "7": "#F59E0B", "7–8": "#F59E0B",
    "8": "#F97316", "8–9": "#F97316", "9": "#EF4444",
    "9–10": "#EF4444", "10": "#DC2626",
}
_HIST_PER_PAGE = 15

# ── Pre-load saved state for form defaults ─────────────────────────────────

_saved    = state_mod.load()
_prof     = _saved.get("profile", {})
_sched    = _saved.get("schedule", {})
_has_plan = bool(_saved.get("plan"))

def _time_parts(total_s):
    h, r = divmod(int(total_s or 0), 3600)
    m, s = divmod(r, 60)
    return h, m, s

def _fmt_duration(total_s):
    h, r = divmod(int(total_s or 0), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}"

def _iso_to_dmy(iso):
    try:
        return datetime.strptime(iso, "%Y-%m-%d").strftime("%d-%m-%Y")
    except Exception:
        return iso or ""

def _dmy_to_iso(dmy):
    try:
        return datetime.strptime(str(dmy).strip(), "%d-%m-%Y").strftime("%Y-%m-%d")
    except Exception:
        return str(dmy or "")

_g_h, _g_m, _g_s = _time_parts(_prof.get("goal_time_s", 13500))
_DIST_M_TO_LABEL = {v: k for k, v in DISTANCES.items()}
def _dist_label(raw):
    if not raw:
        return None
    try:
        m = round(float(raw) / 100) * 100
        return _DIST_M_TO_LABEL.get(int(float(raw))) or _DIST_M_TO_LABEL.get(m) or str(raw)
    except (TypeError, ValueError):
        return str(raw) if str(raw) in DISTANCES else None

_b1_dist_saved = _dist_label(_prof.get("b1_dist")) or "Half-marathon"
if _b1_dist_saved not in DIST_KEYS:
    _b1_dist_saved = "Half-marathon"
_b1_h, _b1_m, _b1_s = _time_parts(_prof.get("b1_time_s", 6300))
_b1_date_saved = _iso_to_dmy(_prof.get("b1_date", ""))
_b2_dist_saved = _dist_label(_prof.get("b2_dist")) or "10 km"
if _b2_dist_saved not in DIST_KEYS:
    _b2_dist_saved = "10 km"
_b2_h, _b2_m, _b2_s = _time_parts(_prof.get("b2_time_s", 2850))
_b2_date_saved = _iso_to_dmy(_prof.get("b2_date", ""))
_saved_run_days = [WEEKDAY_LABELS[i] for i in _sched.get("run_days", [1, 3, 4, 5, 6])]
_saved_strength = [WEEKDAY_LABELS[i] for i in _sched.get("strength_days", [])]
_saved_cycling  = [WEEKDAY_LABELS[i] for i in _sched.get("cycling_days", [])]

def _week_num_today(plan: list) -> int:
    today = date.today()
    for w in plan:
        days = w.get("days", [])
        if not days:
            continue
        start = date.fromisoformat(days[0]["date"])
        end   = date.fromisoformat(days[-1]["date"])
        if start <= today <= end:
            return w["week_num"]
        if start > today:
            return w["week_num"]
    return plan[-1]["week_num"] if plan else 1

_cyc_init = active_cycle(_saved)
_init_plan = _cyc_init.get("plan", []) if _cyc_init else _saved.get("plan", [])
_current_week_num = _week_num_today(_init_plan)

def _load_lrp_sessions(sched):
    if sched.get("lrp_sessions"):
        return sched["lrp_sessions"]
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
_lrp1_day = _lrp_day_label(_lrp1); _lrp1_km = _lrp1.get("km", 12.0); _lrp1_type = _lrp1.get("type", "easy")
_lrp2_day = _lrp_day_label(_lrp2); _lrp2_km = _lrp2.get("km", 10.0); _lrp2_type = _lrp2.get("type", "easy")
_lrp3_day = _lrp_day_label(_lrp3); _lrp3_km = _lrp3.get("km", 10.0); _lrp3_type = _lrp3.get("type", "easy")
_lrp4_day = _lrp_day_label(_lrp4); _lrp4_km = _lrp4.get("km", 10.0); _lrp4_type = _lrp4.get("type", "easy")
_has_lrp2 = bool(_lrp2.get("day") is not None)
_has_lrp3 = bool(_lrp3.get("day") is not None)
_has_lrp4 = bool(_lrp4.get("day") is not None)

# ── History helpers ────────────────────────────────────────────────────────

def _format_history_df(df: pd.DataFrame) -> pd.DataFrame:
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
    if "date" in d.columns:
        def _fmt_date(v):
            parts = str(v).split("-")
            if len(parts) == 3 and len(parts[0]) == 4:
                return f"{parts[2]}-{parts[1]}-{parts[0]}"
            return v
        d["date"] = d["date"].apply(_fmt_date)
    # Activity type: pace-based safety net catches misclassified cycling (< 3:20/km impossible on foot)
    if "activity_type" not in d.columns:
        d["activity_type"] = "Other"
    if "avg_pace_s" in df.columns:
        very_fast = df["avg_pace_s"].fillna(9999) < 200
        d.loc[very_fast, "activity_type"] = "Indoor Cycling"
    d["activity_type"] = d["activity_type"].fillna("Other").replace("", "Other")
    d = d.rename(columns={
        "date": "Date", "activity_type": "Type",
        "distance_km": "Distance (km)", "duration_s": "Duration",
        "avg_pace_s": "Avg Pace", "avg_hr": "Avg HR",
        "elevation_gain_m": "Elevation (m)",
        "avg_cadence_spm": "Cadence", "max_hr": "Max HR",
    })
    # hr_drift_pct kept in raw data for load calculations but not shown
    order = ["Date", "Type", "Distance (km)", "Duration", "Avg Pace",
             "Avg HR", "Elevation (m)", "Cadence", "Max HR"]
    cols = [c for c in order if c in d.columns] + \
           [c for c in d.columns if c not in order
            and c not in ("training_load", "hr_drift_pct", "HR Drift")]
    return d[cols]


def _hist_page_html(df, page: int):
    if df is None or not hasattr(df, "iloc") or len(df) == 0:
        return (
            "<div style='text-align:center;padding:48px;color:#9CA3AF;"
            "font-family:-apple-system,sans-serif'>"
            "<div style='font-size:36px'>🏃</div>"
            "<div style='font-size:14px;font-weight:600;color:#374151;margin-top:8px'>No activities yet</div>"
            "<div style='font-size:12px;margin-top:4px'>Upload FIT files or sync from Garmin below.</div>"
            "</div>",
            0,
            "<div style='text-align:center;padding:4px;font-size:12px;color:#9CA3AF'>0 activities</div>",
        )
    total  = len(df)
    pages  = max(1, (total + _HIST_PER_PAGE - 1) // _HIST_PER_PAGE)
    page   = max(0, min(page, pages - 1))
    subset = df.iloc[page * _HIST_PER_PAGE:(page + 1) * _HIST_PER_PAGE]
    cols   = list(df.columns)
    _th = (
        "padding:9px 12px;text-align:left;font-size:10px;font-weight:700;"
        "color:#ffffff;text-transform:uppercase;letter-spacing:.06em;"
        "background:#1B2874;white-space:nowrap;border-right:1px solid rgba(255,255,255,0.1)"
    )
    _th_last = (
        "padding:9px 12px;text-align:left;font-size:10px;font-weight:700;"
        "color:#ffffff;text-transform:uppercase;letter-spacing:.06em;"
        "background:#1B2874;white-space:nowrap"
    )
    _td = (
        "padding:8px 12px;font-size:12px;color:#374151;white-space:nowrap;"
        "border-bottom:1px solid #F0F2F8;border-right:1px solid #F0F2F8"
    )
    _td_last = (
        "padding:8px 12px;font-size:12px;color:#374151;white-space:nowrap;"
        "border-bottom:1px solid #F0F2F8"
    )
    header = "".join(
        f"<th style='{_th_last if i == len(cols)-1 else _th}'>{c}</th>"
        for i, c in enumerate(cols)
    )
    body = ""
    ncols = len(cols)
    for i, (_, row) in enumerate(subset.iterrows()):
        bg          = "#ffffff" if i % 2 == 0 else "#F8FAFE"
        left_border = "border-left:3px solid #F5871F;" if i % 2 == 0 else "border-left:3px solid #E5E7EB;"
        cells = ""
        for j, c in enumerate(cols):
            v      = row[c]
            vstr   = v if pd.notna(v) else "—"
            style  = _td_last if j == ncols - 1 else _td
            lbord  = left_border if j == 0 else ""
            cells += f"<td style='{style}{lbord}'>{vstr}</td>"
        body += f"<tr style='background:{bg}'>{cells}</tr>"
    html = (
        "<div style='overflow-x:auto;border-radius:10px;border:1px solid #E5E7EB;"
        "box-shadow:0 2px 8px rgba(27,40,116,0.06);font-family:-apple-system,sans-serif'>"
        "<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr style='background:#1B2874'>{header}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )
    info = (
        f"<div style='text-align:center;padding:8px 0 2px;font-size:12px;color:#6B7280'>"
        f"Page <b>{page + 1}</b> of {pages} &nbsp;·&nbsp; {total} activities</div>"
    )
    return html, page, info

def _hist_reset_page(df):  return _hist_page_html(df, 0)
def _hist_prev_page(df, p): return _hist_page_html(df, p - 1)
def _hist_next_page(df, p): return _hist_page_html(df, p + 1)

# ── Zones rendering ────────────────────────────────────────────────────────

def _pace_zones_html(zones_data: dict) -> str:
    if not zones_data:
        return "<p style='color:#9CA3AF;padding:24px;text-align:center'>No zones yet — complete Setup &amp; Plan first.</p>"
    z    = Zones(**{k: zones_data[k] for k in Zones.__dataclass_fields__ if k in zones_data})
    rows = pace_zones_extended(z)
    th   = ("padding:9px 12px;text-align:left;font-size:11px;font-weight:600;"
            "color:#6B7280;text-transform:uppercase;letter-spacing:.04em;"
            "background:#F9FAFB;border-bottom:1px solid #E5E7EB")
    body = ""
    for i, r in enumerate(rows):
        bg  = "#fff" if i % 2 == 0 else "#F5F7FA"
        col = _RPE_COLOR.get(r["rpe"], "#6B7280")
        body += (
            f"<tr style='background:{bg}'>"
            f"<td style='padding:8px 12px;font-weight:600;color:#111827;font-size:13px'>{r['workout']}</td>"
            f"<td style='padding:8px 12px;font-weight:700;color:#1B2874;font-size:13px;"
            f"font-variant-numeric:tabular-nums'>{r['pace']}</td>"
            f"<td style='padding:8px 12px;text-align:center'>"
            f"<span style='background:{col};color:#fff;padding:2px 7px;"
            f"border-radius:99px;font-size:11px;font-weight:700'>RPE&nbsp;{r['rpe']}</span></td>"
            f"<td style='padding:8px 12px;color:#6B7280;font-size:12px'>{r['notes']}</td></tr>"
        )
    return (
        f"<table style='width:100%;border-collapse:collapse;border:1px solid #E5E7EB;"
        f"border-radius:8px;overflow:hidden;font-family:-apple-system,sans-serif'>"
        f"<thead><tr>"
        f"<th style='{th}'>Workout</th><th style='{th}'>Pace</th>"
        f"<th style='{th}'>Effort</th><th style='{th}'>Notes</th>"
        f"</tr></thead><tbody>{body}</tbody></table>"
    )


def _hr_zones_html(hr_max: int, hr_rest: int = 50) -> str:
    if not hr_max or hr_max < 100:
        return "<p style='color:#9CA3AF;padding:24px;text-align:center'>Enter your max heart rate above to see HR zones.</p>"
    body = ""
    for z in hr_zones(int(hr_max), int(hr_rest)):
        bar_w = min(100, max(20, round((z["hi"] - z["lo"]) * 5)))
        body += (
            f"<tr>"
            f"<td style='padding:10px 14px'>"
            f"<div style='display:flex;align-items:center;gap:10px'>"
            f"<span style='width:30px;height:30px;border-radius:50%;background:{z['color']};"
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"font-weight:700;color:#fff;font-size:13px;flex-shrink:0'>Z{z['zone']}</span>"
            f"<div><div style='font-weight:600;font-size:13px;color:#111827'>{z['name']}</div>"
            f"<div style='font-size:11px;font-weight:600;color:#374151'>{z['desc']}</div></div></div></td>"
            f"<td style='padding:10px 14px;font-weight:700;color:#1B2874;"
            f"font-variant-numeric:tabular-nums;white-space:nowrap;font-size:13px'>{z['lo']}–{z['hi']} bpm</td>"
            f"<td style='padding:10px 14px;width:120px'>"
            f"<div style='background:#F3F4F6;border-radius:4px;height:8px'>"
            f"<div style='background:{z['color']};height:8px;border-radius:4px;width:{bar_w}%'></div>"
            f"</div></td></tr>"
        )
    return (
        f"<div style='background:#F0FBF4;border-radius:10px;padding:10px 4px'>"
        f"<table style='width:100%;border-collapse:collapse;font-family:-apple-system,sans-serif'>"
        f"<tbody>{body}</tbody></table></div>"
    )


# ── Plan rendering ─────────────────────────────────────────────────────────

def _session_hr_text(session_type: str, desc: str, zones: list) -> str:
    if not zones or session_type in ("Rest", "Strength", "Cycling / Zwift"):
        return "—"
    def z(n):
        i = zones[n - 1]
        return f"{i['name']} ({i['lo']}–{i['hi']})"
    if session_type == "Recovery":
        return f"Throughout: {z(1)}"
    if session_type == "Easy":
        return f"Throughout: {z(1)} – {z(2)}"
    if session_type in ("Long Run", "Medium-Long"):
        return (f"Easy: {z(1)}–{z(2)} · M-pace: {z(3)}"
                if "M-pace" in desc else f"Throughout: {z(1)} – {z(2)}")
    if session_type == "Tempo":
        return f"WU: {z(1)} · Effort: {z(4)} · CD: {z(1)}"
    if session_type == "SVC Intervals":
        return f"WU: {z(1)}–{z(2)} · Intervals: {z(5)} · Récup: {z(1)} · CD: {z(2)}"
    if session_type == "Marathon Pace":
        return f"WU: {z(1)} · M-pace: {z(3)} · CD: {z(1)}"
    if session_type == "Progression Run":
        return f"Easy: {z(1)} · M-pace: {z(3)} · T-pace: {z(4)}"
    if session_type == "Club Run (LRP)":
        return (f"WU: {z(1)} · Effort: {z(4)}–{z(5)} · CD: {z(1)}"
                if "tempo" in desc.lower() else f"Throughout: {z(1)} – {z(2)}")
    return f"{z(1)} – {z(2)}"


def _plan_week_html(plan: list, week_idx: int, hr_max: int = 177, hr_rest: int = 50) -> str:
    _EMPTY = (
        "<div style='text-align:center;padding:56px;color:#9CA3AF;"
        "font-family:-apple-system,sans-serif'>"
        "<div style='font-size:44px'>🏃</div>"
        "<div style='font-size:15px;font-weight:600;color:#374151;margin-top:8px'>No plan yet</div>"
        "<div style='font-size:13px;margin-top:4px'>Go to Setup &amp; Plan to generate your training plan.</div>"
        "</div>"
    )
    if not plan:
        return _EMPTY
    week_idx = max(0, min(week_idx, len(plan) - 1))
    w        = plan[week_idx]
    phase    = w.get("phase", "")
    accent   = _PHASE_COLORS.get(phase, "#64748B")
    focus    = w.get("focus", "")
    wk_num   = w.get("week_num", week_idx + 1)
    total_km = w.get("target_km", 0)
    zones_hr = hr_zones(hr_max, hr_rest)
    header   = (
        f"<div style='background:{accent};border-radius:10px 10px 0 0;"
        f"padding:10px 16px;color:#fff;font-family:-apple-system,sans-serif'>"
        f"<span style='font-size:14px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:.06em'>{phase} · Week {wk_num}</span>"
        f"<span style='font-size:12px;opacity:.85;margin-left:12px'>{focus}</span>"
        f"<span style='float:right;font-size:11px;opacity:.75'>"
        f"Target {total_km:.0f} km &nbsp;·&nbsp; {week_idx+1} / {len(plan)}</span>"
        f"</div>"
    )
    _th = ("padding:10px;text-align:left;font-weight:600;font-size:11px;"
           "letter-spacing:.05em;text-transform:uppercase;color:#fff;background:#1B2874")
    rows = ""
    today = date.today()
    for i, day in enumerate(w["days"]):
        d      = date.fromisoformat(day["date"])
        sess   = day["session_type"]
        desc   = day["description"]
        km     = f"{day['distance_km']:.0f} km" if day.get("distance_km") else "—"
        _PACE_COLOR = {
            "M-pace": "#1B2874", "T-pace": "#F5871F", "I-pace": "#DC2626",
            "Easy section": "#16A34A", "Finish at": "#1B2874",
            "WU/CD": "#6B7280", "recovery": "#6B7280",
        }
        def _chip(k, v):
            c = _PACE_COLOR.get(k, "#6B7280")
            return (f"<span style='display:inline-block;margin:3px 4px 0 0;padding:2px 7px;"
                    f"border-radius:99px;font-size:10px;font-weight:700;color:#fff;"
                    f"background:{c}'>{k}: {v}</span>")
        chips = "".join(_chip(k, v) for k, v in day.get("targets", {}).items())
        detail = f"{desc}<br>{chips}" if chips else desc
        bg     = "#fff" if i % 2 == 0 else "#FAFAFA"
        badge  = _SESSION_COLOR.get(sess, "#6B7280")
        hr_txt = _session_hr_text(sess, desc, zones_hr)
        left_border = "border-left:4px solid #F59E0B;" if d == today else f"border-left:3px solid {accent};"
        rows += (
            f"<tr style='background:{bg}'>"
            f"<td style='padding:7px 10px;color:#9CA3AF;font-size:12px;"
            f"white-space:nowrap;{left_border}'>{d.day} {d.strftime('%b')}</td>"
            f"<td style='padding:7px 10px;font-weight:600;color:#374151;"
            f"white-space:nowrap'>{d.strftime('%a')}</td>"
            f"<td style='padding:7px 10px;white-space:nowrap'>"
            f"<span style='background:{badge};color:#fff;padding:2px 9px;"
            f"border-radius:99px;font-size:11px;font-weight:700'>{sess}</span></td>"
            f"<td style='padding:7px 10px;color:#374151;font-size:12px'>{detail}</td>"
            f"<td style='padding:7px 10px;color:#374151;font-size:11px;line-height:1.5'>{hr_txt}</td>"
            f"<td style='padding:7px 10px;text-align:center;font-weight:700;"
            f"color:#1B2874;white-space:nowrap;font-size:12px'>{km}</td></tr>"
        )
    return (
        header
        + "<div style='overflow-x:auto;border-radius:0 0 10px 10px;"
          "border:1px solid #E5E7EB;border-top:none;"
          "font-family:-apple-system,BlinkMacSystemFont,sans-serif'>"
          "<table style='width:100%;border-collapse:collapse;font-size:13px'>"
          "<thead><tr>"
          f"<th style='{_th}'>Date</th><th style='{_th}'>Day</th>"
          f"<th style='{_th}'>Session</th><th style='{_th}'>Details &amp; Targets</th>"
          f"<th style='{_th}'>Target HR</th><th style='{_th};text-align:center'>Km</th>"
          "</tr></thead>"
          f"<tbody>{rows}</tbody></table></div>"
    )


def _current_week_idx(plan: list) -> int:
    today = date.today()
    for i, w in enumerate(plan):
        days = w.get("days", [])
        if not days:
            continue
        start = date.fromisoformat(days[0]["date"])
        end   = date.fromisoformat(days[-1]["date"])
        if start <= today <= end:
            return i
        if start > today:
            return i
    return max(0, len(plan) - 1)


def _calendar_week_html(plan: list, week_idx: int) -> str:
    _EMPTY = (
        "<div style='text-align:center;padding:56px;color:#9CA3AF;"
        "font-family:-apple-system,sans-serif'>"
        "<div style='font-size:44px'>🏃</div>"
        "<div style='font-size:15px;font-weight:600;color:#374151;margin-top:8px'>No plan yet</div>"
        "<div style='font-size:13px;margin-top:4px'>Go to Setup &amp; Plan to generate your training plan.</div>"
        "</div>"
    )
    if not plan:
        return _EMPTY
    week_idx = max(0, min(week_idx, len(plan) - 1))
    w      = plan[week_idx]
    phase  = w.get("phase", "")
    accent = _PHASE_COLORS.get(phase, "#64748B")
    wk_num = w.get("week_num", week_idx + 1)
    total_km = w.get("target_km", 0)
    today  = date.today()

    cols = ""
    for day in w["days"]:
        d    = date.fromisoformat(day["date"])
        sess = day["session_type"]
        desc = day.get("description", "")
        km   = day.get("distance_km", 0)
        col_color = _SESSION_COLOR.get(sess, "#9CA3AF")
        is_today  = d == today
        border    = f"2px solid #F59E0B" if is_today else f"1px solid #E5E7EB"
        day_hdr   = (
            f"<div style='text-align:center;font-size:10px;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.05em;"
            f"color:{'#1B2874' if is_today else '#6B7280'};margin-bottom:6px'>"
            f"{d.strftime('%a')}<br><span style='font-size:12px'>{d.day}</span></div>"
        )
        if sess in ("Rest",):
            card = "<div style='text-align:center;color:#D1D5DB;font-size:11px;padding:6px'>Rest</div>"
        else:
            km_txt  = f"{km:.0f} km" if km else ""
            desc_sh = (desc[:38] + "…") if len(desc) > 38 else desc
            # chips inside card
            targets = day.get("targets", {})
            chip_html = ""
            if targets:
                _PACE_COLOR = {
                    "M-pace": "#1B2874", "T-pace": "#F5871F", "I-pace": "#DC2626",
                    "Easy section": "#16A34A", "Finish at": "#1B2874",
                    "WU/CD": "#6B7280", "recovery": "#6B7280",
                    "pace": "#10B981",
                }
                for k, v in targets.items():
                    tc = _PACE_COLOR.get(k, "#6B7280")
                    chip_html += (
                        f"<span style='display:inline-block;margin:2px 2px 0 0;"
                        f"padding:1px 5px;border-radius:99px;font-size:9px;"
                        f"font-weight:700;color:#fff;background:{tc}'>{k}: {v}</span>"
                    )
            card = (
                f"<div draggable='true' data-cal-date='{day['date']}' "
                f"style='background:{col_color};color:#fff;border-radius:6px;"
                f"padding:6px 8px;cursor:grab;user-select:none;font-family:-apple-system,sans-serif'>"
                f"<div style='font-size:11px;font-weight:700'>{sess}</div>"
                f"<div style='font-size:10px;opacity:.85;margin-top:2px;line-height:1.3'>{desc_sh}</div>"
                + (f"<div style='font-size:9px;opacity:.75;margin-top:4px'>{km_txt}</div>" if km_txt else "")
                + (f"<div style='margin-top:4px'>{chip_html}</div>" if chip_html else "")
                + "</div>"
            )
        cols += (
            f"<div data-cal-drop='{day['date']}' "
            f"style='min-height:140px;border-radius:8px;border:{border};"
            f"padding:8px;background:#fff;box-sizing:border-box'>"
            f"{day_hdr}{card}</div>"
        )

    header = (
        f"<div style='background:{accent};border-radius:10px 10px 0 0;"
        f"padding:10px 16px;color:#fff;font-family:-apple-system,sans-serif'>"
        f"<span style='font-size:14px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:.06em'>{phase} · Week {wk_num}</span>"
        f"<span style='float:right;font-size:11px;opacity:.75'>"
        f"Target {total_km:.0f} km &nbsp;·&nbsp; {week_idx+1} / {len(plan)}</span>"
        f"</div>"
    )
    return (
        header
        + f"<div style='display:grid;grid-template-columns:repeat(7,1fr);gap:8px;"
          f"padding:16px;background:#F9FAFB;border-radius:0 0 10px 10px'>{cols}</div>"
    )


def _calendar_month_html(plan: list) -> str:
    _EMPTY = (
        "<div style='text-align:center;padding:56px;color:#9CA3AF;"
        "font-family:-apple-system,sans-serif'>"
        "<div style='font-size:15px;font-weight:600;color:#374151'>No plan yet</div>"
        "</div>"
    )
    if not plan:
        return _EMPTY
    today = date.today()
    hdr_cells = "".join(
        f"<th style='padding:4px 6px;font-size:10px;font-weight:600;"
        f"color:#6B7280;text-align:left'>{d}</th>"
        for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    )
    rows = ""
    for wi, w in enumerate(plan):
        phase  = w.get("phase", "")
        accent = _PHASE_COLORS.get(phase, "#64748B")
        wk_num = w.get("week_num", wi + 1)
        day_map = {day["weekday"]: day for day in w["days"]}
        cells = ""
        for wd in range(7):
            day = day_map.get(wd)
            if day is None:
                cells += "<td style='padding:3px'></td>"
                continue
            d    = date.fromisoformat(day["date"])
            sess = day["session_type"]
            col  = _SESSION_COLOR.get(sess, "#9CA3AF")
            is_today = d == today
            bg     = "#FFFBEB" if is_today else "#fff"
            border = "2px solid #F59E0B" if is_today else "1px solid #E5E7EB"
            if sess == "Rest":
                badge = "<span style='color:#D1D5DB;font-size:9px'>—</span>"
            else:
                badge = (
                    f"<span style='background:{col};color:#fff;border-radius:99px;"
                    f"font-size:9px;font-weight:700;padding:1px 5px'>{sess[:4]}</span>"
                )
            cells += (
                f"<td style='padding:3px;vertical-align:top'>"
                f"<div style='background:{bg};border:{border};border-radius:4px;padding:3px 4px'>"
                f"<div style='font-size:9px;color:#9CA3AF'>{d.day} {d.strftime('%b')}</div>"
                f"{badge}</div></td>"
            )
        phase_cell = (
            f"<td style='padding:3px;vertical-align:middle'>"
            f"<div style='background:{accent}22;border-radius:4px;padding:3px 6px'>"
            f"<div style='font-size:9px;font-weight:700;color:{accent}'>{phase}</div>"
            f"<div style='font-size:9px;color:#6B7280'>W{wk_num}</div></div></td>"
        )
        rows += f"<tr>{phase_cell}{cells}</tr>"

    return (
        "<div style='overflow-x:auto;border-radius:10px;border:1px solid #E5E7EB;"
        "font-family:-apple-system,sans-serif'>"
        "<table style='width:100%;border-collapse:separate;border-spacing:2px;"
        "font-size:11px;padding:8px'>"
        f"<thead><tr><th></th>{hdr_cells}</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def _render_plan(week_idx: int, view: str) -> str:
    s = state_mod.load()
    plan = s.get("plan", [])
    if view == "Week":
        return _calendar_week_html(plan, int(week_idx or 0))
    if view == "Month":
        return _calendar_month_html(plan)
    return _plan_week_html(plan, int(week_idx or 0), s.get("hr_max", 177), s.get("hr_rest", 50))


# ── Data loaders ───────────────────────────────────────────────────────────

def load_plan_on_start():
    s    = state_mod.load()
    plan = s.get("plan", [])
    hr_max  = s.get("hr_max", 177)
    hr_rest = s.get("hr_rest", 50)
    idx = _current_week_idx(plan) if plan else 0
    return _plan_week_html(plan, idx, hr_max, hr_rest), idx


def render_plan_week(week_idx: int) -> str:
    s    = state_mod.load()
    plan = s.get("plan", [])
    return _plan_week_html(plan, int(week_idx or 0), s.get("hr_max", 177), s.get("hr_rest", 50))


_RATING_COLOR = {
    debrief_mod.RATING_SUCCESSFUL: "#16A34A",
    debrief_mod.RATING_MODERATE:   "#F59E0B",
    debrief_mod.RATING_PARTIAL:    "#F97316",
    debrief_mod.RATING_MISSED:     "#DC2626",
    debrief_mod.RATING_EXTRA:      "#7C3AED",
    debrief_mod.RATING_REST:       "#9CA3AF",
}
_WEEKDAY_SHORT = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _run_debrief(week_days: list, history: list, zones_d: dict,
                 rpe_map: dict, comment_map: dict) -> list:
    """Run match + rate for all sessions; rpe_map/comment_map keyed by date ISO."""
    matched  = debrief_mod.match_week(week_days, history)
    debriefs = []
    for planned, actual in matched:
        date_str = (planned or actual or {}).get("date", "")
        rpe      = rpe_map.get(date_str)
        comment  = comment_map.get(date_str, "")
        debriefs.append(debrief_mod.rate_session(planned, actual, zones_d, rpe=rpe, comment=comment))
    return debriefs


def _render_session_cards(debriefs: list) -> str:
    """Render per-session debrief cards as HTML."""
    if not debriefs:
        return ""
    parts = []
    for d in debriefs:
        if d.rating == debrief_mod.RATING_REST:
            continue
        color = _RATING_COLOR.get(d.rating, "#6B7280")
        badge_color = _SESSION_COLOR.get(d.session_type, "#6B7280")

        # Plan vs actual line
        actual_line = ""
        if d.actual:
            km   = d.actual.get("distance_km") or 0
            pace = d.actual.get("avg_pace_s")
            hr   = d.actual.get("avg_hr")
            drift = d.actual.get("hr_drift_pct")
            parts_a = [f"{km:.1f} km"]
            if pace:
                m, s = divmod(int(pace), 60)
                parts_a.append(f"{m}:{s:02d}/km")
            if hr:
                parts_a.append(f"HR {hr:.0f}")
            if drift is not None:
                parts_a.append(f"drift {drift:.1f}%")
            if d.rpe:
                parts_a.append(f"RPE {int(d.rpe)}")
            actual_line = " · ".join(parts_a)

        strengths_html = "".join(
            f"<div style='color:#16A34A;font-size:12px;margin:2px 0'>✓ {s}</div>"
            for s in d.strengths
        )
        weaknesses_html = "".join(
            f"<div style='color:#B45309;font-size:12px;margin:2px 0'>▲ {w}</div>"
            for w in d.weaknesses
        )
        advice_html = (
            f"<div style='color:#1B2874;font-size:12px;margin:4px 0'>→ {d.advice}</div>"
            if d.advice else ""
        )
        comment_html = (
            f"<div style='color:#9CA3AF;font-size:11px;font-style:italic;margin:4px 0'>"
            f"&ldquo;{d.comment}&rdquo;</div>"
            if d.comment else ""
        )

        def _chip(label, col):
            return (f"<span style='display:inline-block;padding:1px 7px;border-radius:99px;"
                    f"font-size:10px;font-weight:700;color:#fff;background:{col};margin-right:4px'>"
                    f"{label}</span>")

        card = (
            f"<div style='border-left:4px solid {color};padding:10px 14px;margin:6px 0;"
            f"background:#FAFAFA;border-radius:0 8px 8px 0;font-family:-apple-system,sans-serif'>"
            f"<div style='margin-bottom:6px'>"
            f"<span style='font-size:11px;color:#9CA3AF'>{d.date}</span>&nbsp;"
            + _chip(d.session_type, badge_color)
            + _chip(d.rating, color)
            + f"</div>"
            f"<div style='font-size:12px;color:#374151;margin-bottom:4px'>{d.description}</div>"
        )
        if d.planned_km:
            plan_str = f"Plan: {d.planned_km:.0f} km"
            card += (
                f"<div style='font-size:12px;color:#6B7280;margin-bottom:4px'>"
                f"{plan_str}"
                + (f" &nbsp;→&nbsp; Actual: {actual_line}" if actual_line else " &nbsp;→&nbsp; Not completed")
                + "</div>"
            )
        card += strengths_html + weaknesses_html + advice_html + comment_html + "</div>"
        parts.append(card)
    return "".join(parts)


def _render_compliance_card(comp: dict, prior_weeks: list = None) -> str:
    """Render the plan-completion compliance card as HTML."""
    pct = comp["pct_sessions"]
    bar_color = "#16A34A" if pct >= 85 else ("#F59E0B" if pct >= 60 else "#DC2626")
    bar_w = max(4, min(100, pct))

    def _chip(label, color="#6B7280"):
        return (f"<span style='display:inline-block;padding:2px 8px;border-radius:99px;"
                f"font-size:11px;font-weight:600;color:#fff;background:{color};margin:2px'>"
                f"{label}</span>")

    chips = (
        _chip(f"✓ {comp['successful']} done", "#16A34A")
        + (_chip(f"~ {comp['moderate']} moderate", "#F59E0B") if comp["moderate"] else "")
        + (_chip(f"↓ {comp['partial']} partial", "#F97316") if comp["partial"] else "")
        + (_chip(f"✗ {comp['missed']} missed", "#DC2626") if comp["missed"] else "")
        + (_chip(f"+ {comp['extra']} extra", "#7C3AED") if comp["extra"] else "")
    )

    km_line = (
        f"{comp['actual_km']:.1f} / {comp['planned_km']:.1f} km "
        f"({comp['pct_km']}%)"
    )
    key_line = (
        f"Key sessions (quality + long): {comp['key_done']} / {comp['key_planned']}"
        if comp["key_planned"] else ""
    )

    prior_html = ""
    if prior_weeks:
        n   = len(prior_weeks)
        avg = round(sum(w.get("pct_sessions", 0) for w in prior_weeks) / n)
        tot_planned = sum(w.get("planned_km", 0) for w in prior_weeks)
        tot_actual  = sum(w.get("actual_km", 0) for w in prior_weeks)
        prior_html = (
            f"<div style='margin-top:8px;font-size:11px;color:#6B7280;border-top:1px solid #E5E7EB;"
            f"padding-top:6px'>Across {n} debriefed week{'s' if n>1 else ''}: "
            f"{avg}% of sessions · {tot_actual:.0f} of {tot_planned:.0f} km</div>"
        )

    return (
        f"<div style='background:#F0F9FF;border:1px solid #BAE6FD;border-radius:10px;"
        f"padding:14px 18px;font-family:-apple-system,sans-serif;margin:8px 0'>"
        f"<div style='font-size:11px;font-weight:700;color:#1B2874;text-transform:uppercase;"
        f"letter-spacing:.05em;margin-bottom:8px'>Plan completion"
        f"<span style='font-weight:400;font-size:10px;color:#9CA3AF;margin-left:8px'>"
        f"measured against the plan as it stands today</span></div>"
        f"<div style='display:flex;align-items:center;gap:14px;margin-bottom:10px'>"
        f"<div style='font-size:36px;font-weight:800;color:{bar_color}'>{pct}%</div>"
        f"<div style='flex:1'>"
        f"<div style='background:#E5E7EB;border-radius:99px;height:10px'>"
        f"<div style='background:{bar_color};width:{bar_w}%;height:10px;border-radius:99px'></div>"
        f"</div>"
        f"<div style='font-size:11px;color:#6B7280;margin-top:4px'>{km_line}</div>"
        f"</div></div>"
        f"<div style='margin-bottom:6px'>{chips}</div>"
        + (f"<div style='font-size:11px;color:#374151'>{key_line}</div>" if key_line else "")
        + prior_html
        + "</div>"
    )


def _week_iso_for(week_days: list) -> str:
    if not week_days:
        return ""
    try:
        d0 = date.fromisoformat(week_days[0]["date"])
        return f"{d0.isocalendar()[0]}-W{d0.isocalendar()[1]:02d}"
    except Exception:
        return ""


def _session_choices(week_days: list, hist: list) -> list:
    """Build (label, date_str) choice list for the session dropdown."""
    matched = debrief_mod.match_week(week_days, hist)
    choices = []
    for planned, actual in matched:
        if planned and planned.get("session_type") == "Rest":
            continue
        if planned is None:
            continue
        d_str = planned.get("date", "")
        wd    = planned.get("weekday", -1)
        day   = _WEEKDAY_SHORT[wd] if 0 <= wd <= 6 else "—"
        s_type = planned.get("session_type", "")
        km     = planned.get("distance_km") or 0
        km_str = f" · {km:.0f} km" if km else ""
        try:
            dt = date.fromisoformat(d_str)
            label = f"{day} {dt.day} {dt.strftime('%b')} — {s_type}{km_str}"
        except Exception:
            label = f"{d_str} — {s_type}{km_str}"
        choices.append((label, d_str))
    return choices


def _saved_sessions_map(cyc: dict, week_iso: str) -> dict:
    return {
        sd["date"]: sd
        for sd in cyc.get("session_debriefs", {}).get(week_iso, {}).get("sessions", [])
    }


def load_checkin_panel():
    """Return (week_html, injury, session_dd_update, cards_html, comp_html)."""
    s   = state_mod.load()
    cyc = active_cycle(s)
    injury = cyc.get("config", {}).get("injury_level", "none") if cyc else "none"
    no_plan = "<p style='color:#9CA3AF;padding:16px;text-align:center'>No active plan — complete Setup &amp; Plan first.</p>"
    empty_dd = gr.update(choices=[], value=None)
    if not cyc or not cyc.get("plan"):
        return no_plan, injury, empty_dd, "", ""

    plan    = cyc["plan"]
    hr_max  = s.get("hr_max", 177)
    hr_rest = s.get("hr_rest", 50)
    idx     = _current_week_idx(plan)
    week_html = _plan_week_html(plan, idx, hr_max, hr_rest)

    week = plan[idx] if idx < len(plan) else None
    if not week:
        return week_html, injury, empty_dd, "", ""

    week_days = week.get("days", [])
    hist      = s.get("history", [])
    zones_d   = cyc.get("zones", {})
    week_iso  = _week_iso_for(week_days)
    saved     = _saved_sessions_map(cyc, week_iso)

    choices   = _session_choices(week_days, hist)
    dd_update = gr.update(
        choices=[c[0] for c in choices],
        value=choices[0][0] if choices else None,
    )

    # Show already-debriefed session cards
    rpe_map     = {d: sd["rpe"]     for d, sd in saved.items() if sd.get("rpe")}
    comment_map = {d: sd["comment"] for d, sd in saved.items() if sd.get("comment")}
    debriefed   = [
        debrief_mod.rate_session(p, a, zones_d,
                                 rpe=rpe_map.get((p or a or {}).get("date", "")),
                                 comment=comment_map.get((p or a or {}).get("date", ""), ""))
        for p, a in debrief_mod.match_week(week_days, hist)
        if (p or a or {}).get("date", "") in saved
    ]
    cards_html = _render_session_cards(debriefed) if debriefed else ""

    prior = [v.get("compliance", {}) for k, v in cyc.get("session_debriefs", {}).items()
             if k != week_iso and v.get("compliance")]
    comp  = debrief_mod.compliance(debriefed) if debriefed else {"planned": 0, "successful": 0,
            "moderate": 0, "partial": 0, "missed": 0, "extra": 0,
            "planned_km": 0, "actual_km": 0, "pct_sessions": 0, "pct_km": 0,
            "key_planned": 0, "key_done": 0}
    comp_html = _render_compliance_card(comp, prior or None) if debriefed else ""

    return week_html, injury, dd_update, cards_html, comp_html


def session_info_handler(label: str):
    """Return (rpe, comment, info_html) for the selected session label."""
    s   = state_mod.load()
    cyc = active_cycle(s)
    if not cyc or not cyc.get("plan"):
        return 5, "", ""

    plan      = cyc.get("plan", [])
    idx       = _current_week_idx(plan)
    week      = plan[idx] if idx < len(plan) else None
    if not week:
        return 5, "", ""

    week_days = week.get("days", [])
    hist      = s.get("history", [])
    zones_d   = cyc.get("zones", {})
    week_iso  = _week_iso_for(week_days)
    saved     = _saved_sessions_map(cyc, week_iso)

    # Resolve label → date_str
    choices = _session_choices(week_days, hist)
    date_str = next((v for lbl, v in choices if lbl == label), None)
    if not date_str:
        return 5, "", ""

    # Find matching pair
    matched   = debrief_mod.match_week(week_days, hist)
    pair      = next(((p, a) for p, a in matched
                      if (p or {}).get("date") == date_str), (None, None))
    planned, actual = pair

    sv        = saved.get(date_str, {})
    rpe_val   = sv.get("rpe") or 5
    comment   = sv.get("comment", "")

    # Build info card (plan + auto-matched actual)
    if planned:
        s_type  = planned.get("session_type", "")
        plan_km = planned.get("distance_km") or 0
        badge_c = _SESSION_COLOR.get(s_type, "#6B7280")
        def _chip(lbl, col):
            return (f"<span style='padding:2px 8px;border-radius:99px;font-size:11px;"
                    f"font-weight:700;color:#fff;background:{col};margin-right:4px'>{lbl}</span>")
        desc = planned.get("description", "")
        actual_line = ""
        if actual:
            km   = actual.get("distance_km") or 0
            pace = actual.get("avg_pace_s")
            hr   = actual.get("avg_hr")
            parts_a = [f"{km:.1f} km"]
            if pace:
                m, sc = divmod(int(pace), 60)
                parts_a.append(f"{m}:{sc:02d}/km")
            if hr:
                parts_a.append(f"HR {hr:.0f}")
            actual_line = (
                f"<div style='font-size:12px;color:#374151;margin-top:4px'>"
                f"<b>Actual:</b> {' · '.join(parts_a)}</div>"
            )
        targets = planned.get("targets", {})
        def _tgt_chip(k, v):
            tc = {"M-pace": "#1B2874", "T-pace": "#F5871F", "I-pace": "#DC2626"}.get(k, "#6B7280")
            return (f"<span style='display:inline-block;padding:1px 6px;border-radius:99px;"
                    f"font-size:10px;font-weight:700;color:#fff;background:{tc};margin:2px 2px 0 0'>"
                    f"{k}: {v}</span>")
        tgt_chips = "".join(_tgt_chip(k, v) for k, v in targets.items())
        already = "✓ Already debriefed" if date_str in saved else ""
        already_html = (f"<div style='font-size:11px;color:#16A34A;margin-top:4px'>{already}</div>"
                        if already else "")
        info_html = (
            f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;"
            f"padding:10px 14px;font-family:-apple-system,sans-serif;margin:8px 0'>"
            + _chip(s_type, badge_c)
            + f"<span style='font-size:12px;color:#374151;margin-left:4px'>"
            f"Plan: {plan_km:.0f} km</span>"
            f"<div style='font-size:12px;color:#6B7280;margin:6px 0'>{desc}</div>"
            + (f"<div style='margin-bottom:4px'>{tgt_chips}</div>" if tgt_chips else "")
            + (actual_line if actual else
               "<div style='font-size:12px;color:#9CA3AF;margin-top:4px'>No matching activity found in history</div>")
            + already_html
            + "</div>"
        )
    else:
        info_html = ""

    return float(rpe_val), comment, info_html


def debrief_session_handler(label: str, rpe, comment: str):
    """Save debrief for one session; return (cards_html, comp_html)."""
    s   = state_mod.load()
    cyc = active_cycle(s)
    if not cyc or not cyc.get("plan"):
        return "", ""

    plan      = cyc.get("plan", [])
    idx       = _current_week_idx(plan)
    week      = plan[idx] if idx < len(plan) else None
    if not week:
        return "", ""

    week_days = week.get("days", [])
    hist      = s.get("history", [])
    zones_d   = cyc.get("zones", {})
    week_iso  = _week_iso_for(week_days)

    choices  = _session_choices(week_days, hist)
    date_str = next((v for lbl, v in choices if lbl == label), None)
    if not date_str:
        return "", ""

    rpe_val = None
    try:
        r = float(str(rpe).replace(",", "."))
        if 1 <= r <= 10:
            rpe_val = r
    except (TypeError, ValueError):
        pass

    # Load saved sessions, update this one
    existing = cyc.get("session_debriefs", {}).get(week_iso, {})
    saved_list = existing.get("sessions", [])
    saved_map  = {sd["date"]: sd for sd in saved_list}

    # Rate the selected session
    matched  = debrief_mod.match_week(week_days, hist)
    pair     = next(((p, a) for p, a in matched if (p or {}).get("date") == date_str), (None, None))
    new_d    = debrief_mod.rate_session(pair[0], pair[1], zones_d,
                                         rpe=rpe_val, comment=(comment or "").strip())

    from dataclasses import asdict
    saved_map[date_str] = asdict(new_d)

    # Rebuild full debriefs for compliance (only already-saved sessions)
    all_debriefs = []
    for p, a in matched:
        ds = (p or a or {}).get("date", "")
        if ds in saved_map:
            sd = saved_map[ds]
            all_debriefs.append(debrief_mod.rate_session(
                p, a, zones_d,
                rpe=sd.get("rpe"), comment=sd.get("comment", ""),
            ))

    comp = debrief_mod.compliance(all_debriefs)

    cyc.setdefault("session_debriefs", {})[week_iso] = {
        "week_iso":   week_iso,
        "saved_at":   date.today().isoformat(),
        "plan_week":  week_days,
        "compliance": comp,
        "sessions":   list(saved_map.values()),
    }
    s = set_active_cycle(s, cyc)
    state_mod.save(s)

    cards_html = _render_session_cards(all_debriefs)
    prior = [v.get("compliance", {}) for k, v in cyc.get("session_debriefs", {}).items()
             if k != week_iso and v.get("compliance")]
    comp_html = _render_compliance_card(comp, prior or None)
    return cards_html, comp_html


def load_debrief_history() -> str:
    """Render all saved session debriefs (newest first) as HTML for the history tab."""
    s   = state_mod.load()
    cyc = active_cycle(s)
    if not cyc:
        return "<p style='color:#9CA3AF;padding:16px'>No active cycle.</p>"
    all_debriefs = cyc.get("session_debriefs", {})
    if not all_debriefs:
        return "<p style='color:#9CA3AF;padding:16px'>No debriefs saved yet — debrief sessions in the Weekly Check-in tab.</p>"

    zones_d = cyc.get("zones", {})
    html_parts = []
    for week_iso in sorted(all_debriefs.keys(), reverse=True):
        entry = all_debriefs[week_iso]
        comp  = entry.get("compliance", {})
        sessions = entry.get("sessions", [])
        pct   = comp.get("pct_sessions", 0)
        bar_c = "#16A34A" if pct >= 85 else ("#F59E0B" if pct >= 60 else "#DC2626")
        header = (
            f"<div style='background:#F0F4FF;border-left:4px solid #1B2874;border-radius:0 8px 8px 0;"
            f"padding:8px 14px;margin:16px 0 6px;font-family:-apple-system,sans-serif'>"
            f"<span style='font-weight:700;color:#1B2874'>{week_iso}</span>"
            f"<span style='margin-left:12px;font-size:12px;color:{bar_c};font-weight:700'>{pct}% compliance</span>"
            f"<span style='margin-left:12px;font-size:11px;color:#6B7280'>"
            f"{comp.get('actual_km', 0):.0f}/{comp.get('planned_km', 0):.0f} km"
            f"</span></div>"
        )
        cards = []
        for sd in sessions:
            d = debrief_mod.SessionDebrief(**{k: v for k, v in sd.items()
                                              if k in debrief_mod.SessionDebrief.__dataclass_fields__})
            if d.rating == debrief_mod.RATING_REST:
                continue
            color   = _RATING_COLOR.get(d.rating, "#6B7280")
            badge_c = _SESSION_COLOR.get(d.session_type, "#6B7280")
            def _bc(lbl, col):
                return (f"<span style='padding:1px 7px;border-radius:99px;font-size:10px;"
                        f"font-weight:700;color:#fff;background:{col};margin-right:4px'>{lbl}</span>")
            cards.append(
                f"<div style='border-left:3px solid {color};padding:6px 12px;margin:4px 0;"
                f"background:#FAFAFA;border-radius:0 6px 6px 0;font-family:-apple-system,sans-serif'>"
                f"<span style='font-size:11px;color:#9CA3AF'>{d.date}</span>&nbsp;"
                + _bc(d.session_type, badge_c) + _bc(d.rating, color)
                + (f"<span style='font-size:11px;color:#6B7280;margin-left:4px'>RPE {int(d.rpe)}</span>" if d.rpe else "")
                + (f"<div style='font-size:11px;color:#9CA3AF;font-style:italic'>&ldquo;{d.comment}&rdquo;</div>" if d.comment else "")
                + "</div>"
            )
        html_parts.append(header + "".join(cards))

    return "".join(html_parts) if html_parts else "<p style='color:#9CA3AF;padding:16px'>No debriefs yet.</p>"


def load_history_on_start():
    s    = state_mod.load()
    hist = s.get("history", [])
    return _format_history_df(pd.DataFrame(hist)) if hist else pd.DataFrame()


def load_status_on_start():
    s = state_mod.load()
    cyc = active_cycle(s)
    parts = []
    if cyc:
        r = cyc["race"]
        parts.append(
            f"<b>{r.get('name','')}</b> · {r.get('date','')} · "
            f"Target {_fmt_duration(r.get('goal_time_s',0))}"
        )
    hist_n = len(s.get("history", []))
    if hist_n:
        parts.append(f"{hist_n} activities")
    if s.get("plan"):
        parts.append(f"{len(s['plan'])} weeks")
    text = " &nbsp;·&nbsp; ".join(parts) if parts else "No saved data yet — fill in Setup &amp; Plan to get started."
    return (
        "<div style='background:#1B2874;border-radius:8px;padding:9px 16px;"
        "color:#fff;font-size:13px;font-family:-apple-system,sans-serif'>" + text + "</div>"
    )


def load_zones_tab():
    s    = state_mod.load()
    zd   = s.get("zones", {})
    if not zd:
        empty = "<p style='color:#9CA3AF;padding:32px;text-align:center'>No zones calculated yet — complete Setup &amp; Plan first.</p>"
        return empty, 177, 50, empty, empty
    hr_max  = int(s.get("hr_max", 177))
    hr_rest = int(s.get("hr_rest", 50))
    vdot    = zd.get("vdot", 0)
    cur_s   = marathon_time_from_vdot(vdot) if vdot else 0
    goal_s  = s.get("profile", {}).get("goal_time_s", 0)
    goal_str = f" · Goal: <b>{_fmt_duration(goal_s)}</b>" if goal_s else ""
    header = (
        "<div style='background:#1B2874;border-radius:8px;padding:10px 18px;margin-bottom:4px;"
        "color:#fff;font-family:-apple-system,sans-serif;font-size:13px'>"
        f"<b>VDOT {vdot:.1f}</b> · Marathon equivalent: <b>{_fmt_duration(cur_s)}</b>{goal_str}</div>"
    ) if vdot else ""
    return header, hr_max, hr_rest, _pace_zones_html(zd), _hr_zones_html(hr_max, hr_rest)


def update_zones_hr(hr_max, hr_rest):
    s = state_mod.load()
    cyc = active_cycle(s)
    if hr_max:
        s["athlete"] = {**s.get("athlete", {}), "hr_max": int(hr_max)}
        if cyc:
            cyc["athlete"] = {**cyc.get("athlete", {}), "hr_max": int(hr_max)}
            s = set_active_cycle(s, cyc)
    if hr_rest:
        s["athlete"] = {**s.get("athlete", {}), "hr_rest": int(hr_rest)}
        if cyc:
            cyc["athlete"] = {**cyc.get("athlete", {}), "hr_rest": int(hr_rest)}
            s = set_active_cycle(s, cyc)
    state_mod.save(s)
    return _hr_zones_html(int(hr_max or 177), int(hr_rest or 50))


# ── Generate plan handler ──────────────────────────────────────────────────

def _parse_lrp_sessions(
    lrp1_day, lrp1_km, lrp1_type,
    lrp2_day, lrp2_km, lrp2_type, lrp2_visible,
    lrp3_day, lrp3_km, lrp3_type, lrp3_visible,
    lrp4_day, lrp4_km, lrp4_type, lrp4_visible,
) -> list:
    sessions = []
    defaults = [12, 10, 10, 10]
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
            sessions.append({"day": d, "km": float(km_v or defaults[i]), "type": type_v or "easy"})
    return sessions


def compute_and_generate(
    name, goal_race, marathon_date_str,
    goal_h, goal_m, goal_s_val,
    b1_dist, b1_h, b1_m, b1_s_val, b1_date_str,
    b2_dist, b2_h, b2_m, b2_s_val, b2_date_str,
    injury, injury_notes,
    run_days_labels,
    min_runs_per_week, max_runs_per_week, allow_volume_increase,
    lrp1_day, lrp1_km, lrp1_type,
    lrp2_day, lrp2_km, lrp2_type, lrp2_visible,
    lrp3_day, lrp3_km, lrp3_type, lrp3_visible,
    lrp4_day, lrp4_km, lrp4_type, lrp4_visible,
    strength_labels, cycling_labels,
    hr_rest_val,
):
    try:
        goal_time_s = int(goal_h or 0) * 3600 + int(goal_m or 0) * 60 + int(goal_s_val or 0)
        if not goal_time_s:
            return {}, "", "⚠ Enter a goal time.", ""

        marathon_date = _dmy_to_iso(marathon_date_str)
        if not marathon_date:
            return {}, "", "⚠ Enter race date (DD-MM-YYYY).", ""
        race_dt = date.fromisoformat(marathon_date)

        b1_time_s = int(b1_h or 0) * 3600 + int(b1_m or 0) * 60 + int(b1_s_val or 0)
        b2_time_s = int(b2_h or 0) * 3600 + int(b2_m or 0) * 60 + int(b2_s_val or 0)
        b1_dist_m = DISTANCES.get(b1_dist, 0)
        b2_dist_m = DISTANCES.get(b2_dist, 0)

        if not b1_time_s or not b1_dist_m:
            return {}, "", "⚠ Benchmark 1 is required.", ""

        vdot = vdot_from_race(b1_dist_m, b1_time_s)
        cv   = 0.0
        if b2_time_s and b2_dist_m and b2_dist_m != b1_dist_m:
            from coach.zones import cv_from_two_efforts
            cv = cv_from_two_efforts(b1_dist_m, b1_time_s, b2_dist_m, b2_time_s)
        else:
            from coach.zones import cv_from_vdot
            cv = cv_from_vdot(vdot)

        zones = build_zones(vdot, cv)
        # M-pace = goal race pace, not VDOT-predicted pace (athletes train at their target)
        from dataclasses import replace as _dc_replace
        goal_mp_s = int(goal_time_s / 42.195)
        if goal_mp_s > zones.marathon:
            zones = _dc_replace(zones, marathon=goal_mp_s)

        run_days = sorted([WEEKDAY_MAP[d] for d in (run_days_labels or []) if d in WEEKDAY_MAP])
        strength_days = [WEEKDAY_MAP[d] for d in (strength_labels or []) if d in WEEKDAY_MAP]
        cycling_days  = [WEEKDAY_MAP[d] for d in (cycling_labels or []) if d in WEEKDAY_MAP]

        lrp_sessions = _parse_lrp_sessions(
            lrp1_day, lrp1_km, lrp1_type,
            lrp2_day, lrp2_km, lrp2_type, lrp2_visible,
            lrp3_day, lrp3_km, lrp3_type, lrp3_visible,
            lrp4_day, lrp4_km, lrp4_type, lrp4_visible,
        )

        plan_weeks = generate_plan(
            marathon_date        = race_dt,
            goal_time_s          = goal_time_s,
            zones                = zones,
            run_days             = run_days,
            lrp_sessions         = lrp_sessions,
            strength_days        = strength_days,
            cycling_days         = cycling_days,
            injury               = injury or "none",
            runs_per_week        = int(min_runs_per_week or 0),
            max_runs_per_week    = int(max_runs_per_week or 0),
            allow_volume_increase= bool(allow_volume_increase),
        )
        plan = [
            {
                "week_num": w.week_num, "phase": w.phase,
                "focus": w.focus, "target_km": w.target_km,
                "days": [
                    {"date": str(d.date), "weekday": d.weekday,
                     "session_type": d.session.type,
                     "description": d.session.description,
                     "distance_km": d.session.distance_km,
                     "targets": d.session.targets}
                    for d in w.days
                ],
            }
            for w in plan_weeks
        ]

        # Persist in new v2 schema
        s   = state_mod.load()
        cyc = active_cycle(s)
        hr_rest = int(hr_rest_val or 50)
        hr_max  = s.get("hr_max", 177)

        bmks = []
        if b1_time_s and b1_dist_m:
            bmks.append({"distance_m": b1_dist_m, "time_s": b1_time_s,
                         "date": _dmy_to_iso(b1_date_str)})
        if b2_time_s and b2_dist_m:
            bmks.append({"distance_m": b2_dist_m, "time_s": b2_time_s,
                         "date": _dmy_to_iso(b2_date_str)})

        club_runs = [
            {"id": f"lrp_{i}", "day": sess["day"], "type": sess["type"],
             "distance_km": sess["km"], "pinned_day": True,
             "pinned_distance": (sess.get("type") != "long"),
             "description": f"LRP — {WEEKDAY_LABELS[sess['day']]} {sess['km']:.0f} km"}
            for i, sess in enumerate(lrp_sessions)
        ]
        cfg = {
            "default_run_days":   run_days,
            "strength_days":      strength_days,
            "cycling_days":       cycling_days,
            "injury_level":       injury or "none",
            "injury_notes":       injury_notes or "",
            "min_runs_per_week":  int(min_runs_per_week or 0),
            "max_runs_per_week":  int(max_runs_per_week or 0),
            "allow_volume_increase": bool(allow_volume_increase),
            "club_runs":          club_runs,
        }

        cyc_id = cyc["id"] if cyc else f"marathon-{marathon_date}"
        cycle = {
            "id":     cyc_id,
            "status": "active",
            "race": {
                "name":           goal_race or f"Marathon {race_dt.year}",
                "date":           marathon_date,
                "distance_km":    42.195,
                "distance_label": "marathon",
                "goal_time_s":    goal_time_s,
            },
            "athlete":     {"name": name or "", "hr_max": hr_max, "hr_rest": hr_rest},
            "config":      cfg,
            "benchmarks":  bmks,
            "zones":       zones.__dict__,
            "plan":        plan,
            "state_model": sm_to_dict(apply_history(sm_from_dict({}), s.get("history", []))),
            "weekly_overrides":  {},
            "check_in_history": [],
        }

        s["athlete"] = cycle["athlete"]
        for c in s.get("cycles", []):
            if c.get("status") == "active":
                c["status"] = "archived"
        s = set_active_cycle(s, cycle)
        state_mod.save(s)

        idx = _current_week_idx(plan)
        plan_html = _plan_week_html(plan, idx, hr_max, hr_rest)
        msg = (f"✓ Plan saved — {len(plan)} weeks · VDOT {zones.vdot:.1f} · "
               f"M-pace {fmt_pace(zones.marathon)} · T-pace {fmt_pace(zones.threshold)}")
        summary_html = (
            f"<div style='background:#DCFCE7;border:1px solid #16A34A;border-radius:8px;"
            f"padding:10px 14px;font-size:13px;color:#166534'>{msg}</div>"
        )
        return zones.__dict__, plan_html, msg, summary_html

    except Exception as e:
        import traceback; traceback.print_exc()
        return {}, "", f"Error: {e}", ""


# ── Weekly check-in handler ────────────────────────────────────────────────

def checkin_handler(feeling, comments, avail_days, injury_status):
    s   = state_mod.load()
    cyc = active_cycle(s)
    if not cyc:
        return "⚠ No active cycle — complete Setup & Plan first.", "", "", ""

    hr_max  = s.get("hr_max", 177)
    hr_rest = s.get("hr_rest", 50)
    hist    = s.get("history", [])

    # Regenerate plan if injury level changed
    injury_status = injury_status or "none"
    old_injury = cyc.get("config", {}).get("injury_level", "none")
    if injury_status != old_injury and cyc.get("benchmarks"):
        from dataclasses import replace as _dc_replace
        from coach.plan import generate_plan as _gen_plan
        from coach.zones import build_zones as _bz, vdot_from_race as _vfr
        cfg  = cyc["config"]
        race = cyc["race"]
        bmk  = cyc["benchmarks"][0]
        _vdot  = _vfr(bmk["distance_m"], bmk["time_s"])
        _zones = _bz(_vdot, 0)
        _gmp   = int(race["goal_time_s"] / 42.195)
        if _gmp > _zones.marathon:
            _zones = _dc_replace(_zones, marathon=_gmp)
        _lrp = [{"day": cr["day"], "km": cr["distance_km"], "type": cr["type"]}
                for cr in cfg.get("club_runs", [])]
        _pw = _gen_plan(
            marathon_date     = date.fromisoformat(race["date"]),
            goal_time_s       = race["goal_time_s"],
            zones             = _zones,
            run_days          = cfg["default_run_days"],
            lrp_sessions      = _lrp,
            strength_days     = cfg.get("strength_days", []),
            cycling_days      = cfg.get("cycling_days", []),
            injury            = injury_status,
            runs_per_week     = cfg.get("min_runs_per_week", 0),
            max_runs_per_week = cfg.get("max_runs_per_week", 0),
            allow_volume_increase = cfg.get("allow_volume_increase", True),
        )
        cyc["plan"]  = [
            {"week_num": w.week_num, "phase": w.phase, "focus": w.focus,
             "target_km": w.target_km,
             "days": [{"date": str(d.date), "weekday": d.weekday,
                       "session_type": d.session.type, "description": d.session.description,
                       "distance_km": d.session.distance_km, "targets": d.session.targets}
                      for d in w.days]}
            for w in _pw
        ]
        cyc["zones"] = _zones.__dict__
        cyc["config"]["injury_level"] = injury_status

    # Build availability override — derive club-run skips from unticked days
    availability = None
    default_run_days = set(cyc.get("config", {}).get("default_run_days", []))
    club_run_days    = {cr["day"] for cr in cyc.get("config", {}).get("club_runs", [])}
    if avail_days is not None:
        avail   = [WEEKDAY_MAP.get(d, d) if isinstance(d, str) else d for d in (avail_days or [])]
        unavail = [d for d in default_run_days if d not in avail]
        club = {}
        for d in club_run_days:
            if d in unavail:
                if d == 0:
                    club["lrp_monday"] = "skip"
                elif d == 5:
                    club["lrp_saturday"] = "skip"
        availability = {"available_days": avail, "unavailable_days": unavail,
                        "club_run_decisions": club}

    # Pick up Saturday km stored by adjustment panel for this week
    from datetime import timedelta as _td
    next_monday = date.today() + _td(days=(7 - date.today().weekday()) % 7 or 7)
    week_iso = next_monday.strftime("%G-W%V")
    sat_km = cyc.get("weekly_overrides", {}).get(week_iso, {}).get("saturday_km")

    proposal = propose_week(cyc, hist, feeling=float(feeling or 3.0),
                            saturday_km=sat_km, availability=availability)

    # Overwrite the matching week in the stored plan so the Plan tab reflects adaptation
    plan = cyc.get("plan", [])
    for stored_week in plan:
        days = stored_week.get("days", [])
        if not days:
            continue
        try:
            first_d = date.fromisoformat(days[0]["date"])
            wiso = f"{first_d.isocalendar()[0]}-W{first_d.isocalendar()[1]:02d}"
            if wiso == proposal.week_iso:
                stored_week["days"] = [
                    {"date": str(dp.date), "weekday": dp.weekday,
                     "session_type": dp.session.type,
                     "description": dp.session.description,
                     "distance_km": dp.session.distance_km,
                     "targets": dp.session.targets}
                    for dp in proposal.sessions
                ]
                stored_week["phase"]     = proposal.phase
                stored_week["focus"]     = proposal.focus
                stored_week["target_km"] = proposal.target_km
                break
        except Exception:
            continue
    cyc["plan"] = plan

    cyc.setdefault("weekly_overrides", {})[proposal.week_iso] = availability or {}
    cyc.setdefault("check_in_history", []).append({
        "week_iso":        proposal.week_iso,
        "feeling":         float(feeling or 3.0),
        "comments":        comments or "",
        "ladder_score":    proposal.ladder_score,
        "ladder_decision": proposal.ladder_decision,
        "load_target_km":  proposal.target_km,
    })
    s = set_active_cycle(s, cyc)
    s["history"] = hist
    state_mod.save(s)

    r_label = readiness_label(proposal.readiness)
    metrics_html = (
        f"<div style='display:flex;gap:12px;flex-wrap:wrap;font-family:-apple-system,sans-serif;"
        f"margin-bottom:8px'>"
        + "".join(
            f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;"
            f"padding:10px 14px;min-width:100px'>"
            f"<div style='font-size:10px;color:#6B7280;text-transform:uppercase;"
            f"letter-spacing:.05em'>{label}</div>"
            f"<div style='font-size:17px;font-weight:700;color:#1B2874;margin-top:2px'>{val}</div>"
            f"</div>"
            for label, val in [
                ("Readiness",   r_label),
                ("Target km",   f"{proposal.target_km:.0f}"),
                ("Saturday",    f"{proposal.saturday_km:.0f} km"),
                ("ACWR",        f"{proposal.acwr:.2f}"),
                (proposal.ladder_decision, f"{proposal.ladder_score}/100"),
            ]
        )
        + "</div>"
    )
    if proposal.warnings:
        warnings_html = "".join(
            f"<div style='background:#FEF3C7;border:1px solid #F59E0B;border-radius:8px;"
            f"padding:8px 12px;font-size:12px;color:#92400E;margin-bottom:4px'>⚠ {w}</div>"
            for w in proposal.warnings
        )
        metrics_html += warnings_html

    # Coaching note
    zd = cyc.get("zones", {})
    z  = Zones(**{k: v for k, v in zd.items() if k in Zones.__dataclass_fields__}) if zd else build_zones(40.0, 0.0)
    ctx_data  = proposal.coaching_context
    r_data    = ctx_data.get("race", {}) if isinstance(ctx_data.get("race"), dict) else {}
    days_left = (date.fromisoformat(cyc["race"]["date"]) - date.today()).days
    context   = llm_mod.build_context(
        name       = s.get("athlete", {}).get("name", "Athlete"),
        goal_race  = cyc["race"].get("name", ""),
        goal_time  = _fmt_duration(cyc["race"].get("goal_time_s", 0)),
        weeks_left = days_left // 7,
        phase      = proposal.phase,
        zones_dict = zones_summary(z),
        metrics    = ctx_data.get("metrics"),
        result     = ctx_data.get("ladder"),
        next_focus = proposal.focus,
    ) if ctx_data.get("metrics") else ""
    note = llm_mod.coaching_note(context) if context else (
        f"Readiness: {r_label} · {proposal.ladder_decision} · "
        f"Target {proposal.target_km:.0f} km next week."
    )

    # Next week preview
    next_html = _plan_week_html([{
        "week_num": 99, "phase": proposal.phase, "focus": proposal.focus,
        "target_km": proposal.target_km,
        "days": [{"date": str(dp.date), "weekday": dp.weekday,
                  "session_type": dp.session.type, "description": dp.session.description,
                  "distance_km": dp.session.distance_km, "targets": dp.session.targets}
                 for dp in proposal.sessions],
    }], 0, hr_max, hr_rest)

    status = (
        f"<div style='background:#DCFCE7;border:1px solid #16A34A;border-radius:8px;"
        f"padding:10px 14px;font-size:13px;color:#166534'>"
        f"✓ Check-in saved · {proposal.ladder_decision} · Readiness: {r_label}</div>"
    )
    return status, metrics_html, note, next_html


# ── Garmin handlers ────────────────────────────────────────────────────────

def _garmin_status_html(auth: bool, email=None) -> str:
    if auth and email:
        return (f"<span style='color:#16A34A;font-size:13px'>● Connected as {email}</span>")
    return "<span style='color:#DC2626;font-size:13px'>● Not connected</span>"


def _merge_into_history(records: list):
    if not records:
        return load_history_on_start(), ""
    s = state_mod.load()
    hist = s.get("history", [])
    # Dedup by (date, rounded duration)
    existing = {(r.get("date"), round(r.get("duration_s", 0) / 60)) for r in hist}
    cyc = active_cycle(s)
    hr_max  = s.get("hr_max", 177)
    hr_rest = s.get("hr_rest", 50)
    z = None
    if cyc:
        zd = cyc.get("zones", {})
        if zd:
            z = Zones(**{k: v for k, v in zd.items() if k in Zones.__dataclass_fields__})
    added = []
    for r in records:
        key = (r.get("date"), round(r.get("duration_s", 0) / 60))
        if key not in existing:
            r["training_load"] = compute_load(r, hr_max, hr_rest, z)
            added.append(r)
    merged = sorted(hist + added, key=lambda r: r.get("date", ""), reverse=True)
    s["history"] = merged
    state_mod.save(s)
    df = _format_history_df(pd.DataFrame(merged)) if merged else pd.DataFrame()
    msg = f"{len(added)} new activit{'ies' if len(added) != 1 else 'y'} added ({len(merged)} total)"
    return df, msg


def garmin_import_ui():
    records, sync_msg = garmin_mod.sync_activities(days=30)
    df, merge_msg = _merge_into_history(records)
    return df, f"{sync_msg}{' — ' + merge_msg if merge_msg else ''}"


def garmin_sync_new_ui():
    records, sync_msg = garmin_mod.sync_activities(days=3)
    df, merge_msg = _merge_into_history(records)
    return df, f"{sync_msg}{' — ' + merge_msg if merge_msg else ''}"


def _g_updates(auth, email, msg=""):
    """Return the 5 Garmin component updates for one panel."""
    return (
        _garmin_status_html(auth, email),
        gr.update(visible=not auth),
        gr.update(visible=False),
        gr.update(visible=auth),
        msg,
    )

def _g_mfa_updates(email, msg=""):
    """Updates when MFA is pending — show both login and MFA groups."""
    return (
        _garmin_status_html(False),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=False),
        msg,
    )

def garmin_connect_ui(email, password):
    if not email or not password:
        u = _g_updates(False, None, "Enter email and password.")
        return u + u
    ok, msg = garmin_mod.connect(email, password)
    if ok:
        u = _g_updates(True, email, msg)
        return u + u
    if msg == "MFA_REQUIRED":
        u = _g_mfa_updates(email, "Check your email — enter the verification code below.")
        return u + u
    u = _g_updates(False, None, msg)
    return u + u


def garmin_mfa_ui(email, password, mfa_code):
    ok, msg = garmin_mod.submit_mfa(email, password, mfa_code)
    if ok:
        u = _g_updates(True, email, msg)
        return u + u
    u = _g_updates(False, None, msg)
    return u + u


def garmin_disconnect_ui():
    u = _g_updates(False, None, "Disconnected.")
    return u + u


def load_garmin_ui():
    auth  = garmin_mod.is_authenticated()
    email = garmin_mod.load_email() if auth else None
    u = _g_updates(auth, email)
    return u + u


# ── Adjustments handler ────────────────────────────────────────────────────

def apply_adjustment_handler(from_wk, num_wks, no_club, easy_only, volume_pct, sat_km_val, user_msg):
    s   = state_mod.load()
    cyc = active_cycle(s)
    if not cyc or not cyc.get("plan"):
        return "", "⚠ No plan to adjust."
    from coach import adjustments as adj_mod
    zd   = cyc.get("zones", {})
    plan = cyc["plan"]
    vol  = 1.0
    try:
        vol = float(str(volume_pct).replace("%", "").strip()) / 100
    except Exception:
        pass
    plan, log = adj_mod.apply(plan, int(from_wk or 1), int(num_wks or 0),
                              bool(no_club), bool(easy_only), vol, zd)
    cyc["plan"] = plan

    # Store Saturday km override in weekly overrides for next check-in to pick up
    if sat_km_val:
        try:
            from datetime import timedelta
            next_monday = (date.today() + timedelta(days=(7 - date.today().weekday()) % 7 or 7))
            week_iso = next_monday.strftime("%G-W%V")
            cyc.setdefault("weekly_overrides", {}).setdefault(week_iso, {})["saturday_km"] = float(sat_km_val)
        except Exception:
            pass

    s = set_active_cycle(s, cyc)
    state_mod.save(s)

    r = cyc["race"]
    days_left = max(0, (date.fromisoformat(r["date"]) - date.today()).days)
    ctx = adj_mod.build_adjustment_context(
        athlete_name = s.get("athlete", {}).get("name", "Athlete"),
        goal_race    = r.get("name", ""),
        weeks_left   = days_left // 7,
        phase        = plan[int(from_wk or 1) - 1].get("phase", "?") if plan else "?",
        user_message = user_msg or "",
        no_club_run  = bool(no_club),
        easy_only    = bool(easy_only),
        volume_pct   = vol,
        from_week    = int(from_wk or 1),
        num_weeks    = int(num_wks or 0),
        change_log   = log,
    )
    note = llm_mod.coaching_note(ctx) if ctx else "\n".join(log) or "No changes applied."
    return note, "✓ Plan adjusted."


# ── CSS ────────────────────────────────────────────────────────────────────

CSS = """
:root {
    --lrp-navy:   #1B2874;
    --lrp-blue:   #3B82F6;
    --lrp-orange: #F5871F;
    --lrp-bg:     #F0F2F8;
    --lrp-white:  #FFFFFF;
    --lrp-text:   #111827;
    --block-label-background-fill: #F3F4F6;
    --block-label-text-color: #374151;
    --block-title-background-fill: #F3F4F6;
    --block-title-text-color: #374151;
    --block-info-text-color: #6B7280;
    --input-background-fill-focus: #EFF6FF;
    --input-border-color-focus: #3B82F6;
    --background-fill-primary: #ffffff;
    --background-fill-secondary: #ffffff;
    --panel-background-fill: #ffffff;
    --block-border-width: 1px;
    --block-border-color: #E5E7EB;
    --block-shadow: 0 1px 3px rgba(0,0,0,0.04);
    --code-background-fill: #F8FAFC;
    --body-text-color: #111827;
    --body-text-color-subdued: #6B7280;
}
footer { display: none !important; }
body, .gradio-container {
    background: var(--lrp-bg) !important;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, sans-serif !important;
}
/* Gradio component overrides — match LRP palette everywhere */
.gradio-container .wrap, .gradio-container .block,
.gradio-container .form, .gradio-container fieldset {
    background: #ffffff !important;
}
/* CheckboxGroup: white bg, navy checked boxes, visible labels */
.gradio-container input[type="checkbox"] { accent-color: var(--lrp-navy) !important; }
.gradio-container .checkbox-group label span,
.gradio-container .checkbox-group label,
.gradio-container .check-label { color: #374151 !important; font-size: 13px !important; }
.gradio-container .checkbox-group { background: #ffffff !important; border: 1px solid #E5E7EB !important; border-radius: 8px !important; padding: 8px 12px !important; }
/* Slider: orange track */
.gradio-container input[type="range"] { accent-color: var(--lrp-orange) !important; }
/* Number inputs: white bg */
.gradio-container input[type="number"] { background: #ffffff !important; border: 1px solid #D1D5DB !important; border-radius: 6px !important; color: #111827 !important; }
.gradio-container input[type="number"]:focus { border-color: var(--lrp-navy) !important; outline: none !important; box-shadow: 0 0 0 2px rgba(27,40,116,0.1) !important; }
/* Dropdown select (native) */
.gradio-container select { background: #ffffff !important; border: 1px solid #D1D5DB !important; border-radius: 6px !important; color: #111827 !important; }
/* Dropdown input (selected value) — target by role and scoped classes */
[role="listbox"] { color: #111827 !important; }
input[role="combobox"], input.border-none { color: #111827 !important; }
/* Override subdued (no-selection placeholder) to still be readable */
.subdued, [class*="subdued"] { color: #6B7280 !important; }
/* Dropdown options popup — role="option" is not scoped */
[role="listbox"] ul, ul[class*="options"], [class*="options"] {
    background: #ffffff !important;
    color: #111827 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12) !important;
}
li[role="option"], [role="option"] {
    color: #111827 !important;
    background: #ffffff !important;
}
li[role="option"]:hover, [role="option"]:hover,
li[role="option"].active, [role="option"].active,
[class*="active"][role="option"] {
    background: #EEF2FF !important;
    color: #1B2874 !important;
}
li[role="option"].selected, [role="option"][aria-selected="true"] {
    background: #1B2874 !important;
    color: #ffffff !important;
}
/* Scoped Svelte dropdown classes for this Gradio version */
.svelte-y6qw75 { color: #111827 !important; background: #ffffff !important; }
.item.svelte-y6qw75 { color: #111827 !important; }
.item.svelte-y6qw75:hover { background: #EEF2FF !important; color: #1B2874 !important; }
input.svelte-1scun43, input.svelte-1sk0pyu { color: #111827 !important; }
/* Labels inside blocks */
.gradio-container .block label span, .gradio-container label.block { color: #374151 !important; }
/* File upload zones */
.gradio-container .upload-container { border: 2px dashed #CBD5E1 !important; border-radius: 10px !important; background: #F8FAFC !important; }
/* Textbox */
.gradio-container textarea, .gradio-container input[type="text"], .gradio-container input[type="password"] {
    background: #ffffff !important; border: 1px solid #D1D5DB !important;
    border-radius: 6px !important; color: #111827 !important;
}
/* Read-only / disabled textareas (e.g. coaching note output) */
.gradio-container textarea:disabled, .gradio-container textarea[readonly] {
    color: #111827 !important; opacity: 1 !important; -webkit-text-fill-color: #111827 !important;
}
/* Radio group */
.gradio-container input[type="radio"] { accent-color: var(--lrp-navy) !important; }
#app-layout {
    gap: 0 !important;
    align-items: stretch !important;
    padding: 0 !important;
    min-height: 100vh;
    flex-wrap: nowrap !important;
}
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
#sidebar > .form, #sidebar > div, #sidebar > .gap {
    padding: 0 !important; gap: 0 !important;
    background: transparent !important;
    border: none !important; box-shadow: none !important;
}
#sidebar-logo {
    padding: 20px 16px 16px;
    display: flex !important;
    align-items: center; gap: 11px;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}
#sidebar-logo img { width: 40px; height: 40px; border-radius: 9px; flex-shrink: 0; object-fit: cover; }
.brand-name { color: #ffffff; font-size: 15px; font-weight: 700; letter-spacing: 0.01em; line-height: 1.2; }
.brand-sub { color: rgba(255,255,255,0.48); font-size: 10px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 1px; }
.nav-btn {
    display: block !important; width: calc(100% - 16px) !important;
    margin: 2px 8px !important; text-align: left !important;
    padding: 10px 14px !important; border-radius: 8px !important;
    border: none !important; font-size: 13.5px !important;
    font-weight: 500 !important; cursor: pointer !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    box-shadow: none !important; min-height: unset !important;
}
.nav-btn.secondary, .nav-btn.secondary:focus {
    background: transparent !important; color: rgba(255,255,255,0.70) !important;
}
.nav-btn.secondary:hover { background: rgba(255,255,255,0.09) !important; color: #ffffff !important; }
.nav-btn.primary, .nav-btn.primary:focus {
    background: var(--lrp-orange) !important; color: #ffffff !important;
    font-weight: 600 !important; border-color: transparent !important;
}
.nav-btn.primary:hover { background: #df7318 !important; }
#nav-plan { margin-top: 12px !important; }
#content-area {
    flex: 1 1 0% !important; min-width: 0 !important;
    padding: 28px 32px !important; background: var(--lrp-bg) !important;
    border-radius: 0 !important; overflow-y: auto;
}
#content-area > .form, #content-area > div, #content-area > .gap {
    padding: 0 !important; background: transparent !important;
    border: none !important; box-shadow: none !important;
}
#status-bar { margin-bottom: 8px !important; }
#status-bar .block { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; }
#panel-plan, #panel-setup, #panel-history, #panel-adj, #panel-checkin, #panel-zones {
    background: #ffffff !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04) !important;
    border: 1px solid rgba(0,0,0,0.05) !important;
    padding: 0 0 28px 0 !important;
    gap: 0 !important; overflow: hidden;
}
#panel-plan > div, #panel-setup > div, #panel-history > div,
#panel-adj > div, #panel-checkin > div, #panel-zones > div {
    background: transparent !important; border: none !important;
    box-shadow: none !important; padding: 0 28px !important; gap: 12px !important;
}
/* Force all nested blocks inside panels to white */
#panel-adj .block, #panel-checkin .block, #panel-zones .block,
#panel-history .block, #panel-setup .block, #panel-plan .block {
    background: #ffffff !important; border-color: #E5E7EB !important;
}
/* Deeper nested containers */
#panel-adj .form, #panel-checkin .form, #panel-zones .form,
#panel-history .form, #panel-setup .form {
    background: transparent !important; border: none !important;
    box-shadow: none !important;
}
/* Override any gray/secondary fill */
#panel-adj *, #panel-checkin * {
    --block-background-fill: #ffffff;
    --background-fill-secondary: #F8FAFC;
    --input-background-fill: #ffffff;
    --checkbox-background-color: #ffffff;
    --checkbox-background-color-focus: #EFF6FF;
    --checkbox-background-color-hover: #F5F5FF;
    --checkbox-background-color-selected: #1B2874;
    --checkbox-border-color: #D1D5DB;
    --checkbox-border-color-selected: #1B2874;
    --checkbox-label-background-fill: #ffffff;
    --checkbox-label-background-fill-selected: #EEF2FF;
    --checkbox-label-text-color: #374151;
    --checkbox-label-text-color-selected: #1B2874;
}
.page-header {
    display: flex; align-items: center; gap: 14px;
    padding: 22px 0 18px 0;
    border-bottom: 1px solid #F3F4F6; margin-bottom: 24px; position: relative;
}
.page-header::before {
    content: ''; display: block; width: 4px; min-height: 42px;
    background: linear-gradient(180deg, #1B2874 0%, #3B82F6 100%);
    border-radius: 2px; flex-shrink: 0;
}
.page-header-icon { font-size: 22px; line-height: 1; }
.page-header-title { font-size: 18px; font-weight: 700; color: #1B2874; line-height: 1.2; }
.page-header-sub { font-size: 12px; color: #9CA3AF; margin-top: 3px; font-weight: 400; }
.section-label { border-left: 3px solid var(--lrp-orange); padding: 5px 0 5px 12px; margin: 22px 0 10px; }
.section-label-text { font-size: 11px; font-weight: 700; color: var(--lrp-navy); text-transform: uppercase; letter-spacing: 0.08em; }
.section-label-sub { font-size: 12px; color: #9CA3AF; margin-top: 2px; }
#content-area .block label, #content-area .block label span,
#content-area .block .wrap span, #content-area .block .head span { color: #374151 !important; }
#content-area button.lg.primary, #content-area button.primary {
    background: var(--lrp-orange) !important; border-color: var(--lrp-orange) !important;
    color: #ffffff !important; font-weight: 600 !important;
}
#content-area button.lg.primary:hover, #content-area button.primary:hover {
    background: #df7318 !important; border-color: #df7318 !important;
}
#app-footer {
    text-align: center;
    font-size: 11px;
    color: #9CA3AF;
    padding: 18px 0 8px;
    margin-top: 24px;
    border-top: 1px solid #E5E7EB;
    letter-spacing: 0.04em;
}
/* Drag-drop bridge components — must be in DOM but visually hidden */
#cal-move-bridge, #cal-trigger-btn {
    position: absolute !important;
    width: 1px !important; height: 1px !important;
    overflow: hidden !important; clip: rect(0,0,0,0) !important;
    opacity: 0 !important;
}
"""

# ── Theme ──────────────────────────────────────────────────────────────────

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

# ── Nav helper ─────────────────────────────────────────────────────────────

_NAV_COUNT = 6

def _nav_handler(active_i):
    def handler():
        panels  = [gr.update(visible=(i == active_i)) for i in range(_NAV_COUNT)]
        buttons = [gr.update(variant=("primary" if i == active_i else "secondary"))
                   for i in range(_NAV_COUNT)]
        return panels + buttons
    return handler


# ── Auto-sync ──────────────────────────────────────────────────────────────

def _daily_sync_job():
    garmin_sync_new_ui()

_startup = state_mod.load()
if _startup.get("garmin_auto_sync", False) and garmin_mod.is_authenticated():
    sched_mod.start(_daily_sync_job)

# ── Calendar drag-drop JS ──────────────────────────────────────────────────

_CAL_JS = """
(function() {
    'use strict';
    var _from = null;
    function _trigger(from, to) {
        var bridge = document.querySelector('#cal-move-bridge textarea, #cal-move-bridge input[type=text]');
        if (!bridge) return;
        bridge.value = from + '|' + to + '|' + Date.now();
        bridge.dispatchEvent(new Event('input', { bubbles: true }));
    }
    document.addEventListener('dragstart', function(e) {
        var card = e.target.closest('[data-cal-date]');
        if (card) { _from = card.getAttribute('data-cal-date'); e.dataTransfer.effectAllowed = 'move'; }
    });
    document.addEventListener('dragover', function(e) {
        var cell = e.target.closest('[data-cal-drop]');
        if (cell && _from) { e.preventDefault(); cell.style.outline = '2px dashed #F5871F'; }
    });
    document.addEventListener('dragleave', function(e) {
        var cell = e.target.closest('[data-cal-drop]');
        if (cell && !cell.contains(e.relatedTarget)) { cell.style.outline = ''; }
    });
    document.addEventListener('drop', function(e) {
        var cell = e.target.closest('[data-cal-drop]');
        if (!cell) return;
        e.preventDefault();
        cell.style.outline = '';
        var to = cell.getAttribute('data-cal-drop');
        if (_from && to && _from !== to) { _trigger(_from, to); }
        _from = null;
    });
    document.addEventListener('dragend', function() { _from = null; });
})();
"""

# ── App ────────────────────────────────────────────────────────────────────

_garmin_outputs = None

with gr.Blocks(title="LRP Coach", css=CSS, theme=_theme,
               head=f"<script>{_CAL_JS}</script>") as demo:

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
            btn_hist    = gr.Button("🏃  Activity Log",      variant="secondary", elem_classes="nav-btn")
            btn_adj     = gr.Button("✏️  Adjustments",      variant="secondary", elem_classes="nav-btn")
            btn_checkin = gr.Button("✅  Weekly Check-in",  variant="secondary", elem_classes="nav-btn")
            btn_zones   = gr.Button("📊  My Zones",         variant="secondary", elem_classes="nav-btn")

        # ── Content area ──────────────────────────────────────────────────
        with gr.Column(scale=5, elem_id="content-area"):
            status_bar = gr.HTML("", elem_id="status-bar")

            # ── Panel 0: My Plan ──────────────────────────────────────────
            with gr.Group(visible=True, elem_id="panel-plan") as panel_plan:
                gr.HTML("""<div class="page-header">
                    <span class="page-header-icon">📋</span>
                    <div>
                        <div class="page-header-title">My Plan</div>
                        <div class="page-header-sub">One week at a time · ← → navigate · drag sessions to reschedule</div>
                    </div></div>""")
                plan_week_state = gr.State(value=0)
                cal_undo_stack  = gr.State(value=[])
                with gr.Row():
                    plan_prev_btn  = gr.Button("← Prev week", size="sm", scale=1)
                    plan_this_btn  = gr.Button("This week",   size="sm", scale=1)
                    plan_next_btn  = gr.Button("Next week →", size="sm", scale=1)
                    cal_view_radio = gr.Radio(
                        choices=["List", "Week", "Month"], value="List",
                        label=None, scale=2, interactive=True
                    )
                    cal_undo_btn = gr.Button("↩ Undo", size="sm", scale=1, visible=False)
                plan_html    = gr.HTML()
                cal_warnings = gr.HTML()
                cal_move_bridge = gr.Textbox(
                    value="", label=None, show_label=False,
                    interactive=True, elem_id="cal-move-bridge"
                )

            # ── Panel 1: Setup & Plan ─────────────────────────────────────
            with gr.Group(visible=False, elem_id="panel-setup") as panel_setup:
                gr.HTML("""<div class="page-header">
                    <span class="page-header-icon">⚙️</span>
                    <div>
                        <div class="page-header-title">Setup &amp; Plan</div>
                        <div class="page-header-sub">Your profile · edit to update and regenerate</div>
                    </div></div>""")

                gr.HTML('<div class="section-label"><div class="section-label-text">Profile &amp; Goals</div></div>')
                with gr.Row():
                    name_in          = gr.Textbox(label="Your name", value=_prof.get("name", "Cecilia"))
                    goal_race_in     = gr.Textbox(label="Target race", value=_prof.get("goal_race", ""),
                                                  placeholder="Marathon de Colmar 2026")
                    marathon_date_in = gr.Textbox(label="Race date (DD-MM-YYYY)",
                                                  value=_iso_to_dmy(_prof.get("marathon_date", "")),
                                                  placeholder="27-09-2026")
                with gr.Row():
                    hr_rest_setup_in = gr.Number(label="Resting heart rate (bpm)",
                                                 precision=0, minimum=30, maximum=100,
                                                 value=_saved.get("hr_rest", 50),
                                                 info="Morning resting HR — used for Karvonen HR zones", scale=1)
                    with gr.Column(scale=3): pass

                gr.HTML('<div class="section-label"><div class="section-label-text">Target finish time</div></div>')
                with gr.Row():
                    goal_h = gr.Number(label="Hours",   value=_g_h, precision=0, minimum=2, maximum=7)
                    goal_m = gr.Number(label="Minutes", value=_g_m, precision=0, minimum=0, maximum=59)
                    goal_s = gr.Number(label="Seconds", value=_g_s, precision=0, minimum=0, maximum=59)

                gr.HTML('<div class="section-label"><div class="section-label-text">Benchmark 1 — required</div><div class="section-label-sub">Recent race or time trial</div></div>')
                with gr.Row():
                    b1_dist    = gr.Dropdown(DIST_KEYS, label="Distance", value=_b1_dist_saved, allow_custom_value=False)
                    b1_h       = gr.Number(label="h",   value=_b1_h, precision=0, minimum=0, maximum=5)
                    b1_m       = gr.Number(label="min", value=_b1_m, precision=0, minimum=0, maximum=59)
                    b1_s       = gr.Number(label="sec", value=_b1_s, precision=0, minimum=0, maximum=59)
                    b1_date_in = gr.Textbox(label="Date (DD-MM-YYYY)", value=_b1_date_saved,
                                            placeholder="15-03-2026", scale=2)

                gr.HTML('<div class="section-label"><div class="section-label-text">Benchmark 2 — optional</div><div class="section-label-sub">Different distance → exact SVC. Leave zero if only one result.</div></div>')
                with gr.Row():
                    b2_dist    = gr.Dropdown(DIST_KEYS, label="Distance", value=_b2_dist_saved, allow_custom_value=False)
                    b2_h       = gr.Number(label="h",   value=_b2_h, precision=0, minimum=0, maximum=5)
                    b2_m       = gr.Number(label="min", value=_b2_m, precision=0, minimum=0, maximum=59)
                    b2_s       = gr.Number(label="sec", value=_b2_s, precision=0, minimum=0, maximum=59)
                    b2_date_in = gr.Textbox(label="Date (DD-MM-YYYY)", value=_b2_date_saved,
                                            placeholder="10-01-2026", scale=2)

                gr.HTML('<div class="section-label"><div class="section-label-text">Physical status</div></div>')
                with gr.Row():
                    injury_in       = gr.Radio(["none","light","moderate"], label="Rehab / injury level",
                                               value=_prof.get("injury_level","none"))
                    injury_notes_in = gr.Textbox(label="Notes", value=_prof.get("injury_notes",""), lines=2)

                gr.HTML('<div class="section-label"><div class="section-label-text">Weekly running days</div></div>')
                run_days_in = gr.CheckboxGroup(WEEKDAY_LABELS, label="", value=_saved_run_days)
                with gr.Row():
                    min_runs_in = gr.Number(label="Min runs/week (Base phase)",
                                            value=_sched.get("runs_per_week", 0) or _sched.get("min_runs_per_week", 0),
                                            precision=0, minimum=0, maximum=7)
                    max_runs_in = gr.Number(label="Max runs/week (Build/Peak)",
                                            value=_sched.get("max_runs_per_week", 0),
                                            precision=0, minimum=0, maximum=7)
                    allow_volume_in = gr.Checkbox(label="Allow progressive volume increase",
                                                  value=_sched.get("allow_volume_increase", True))

                gr.HTML('<div class="section-label"><div class="section-label-text">Cross-training</div></div>')
                with gr.Row():
                    strength_in = gr.CheckboxGroup(WEEKDAY_LABELS, label="Strength days", value=_saved_strength)
                    cycling_in  = gr.CheckboxGroup(WEEKDAY_LABELS, label="Cycling / Zwift days", value=_saved_cycling)

                gr.HTML('<div class="section-label"><div class="section-label-text">LRP club sessions</div></div>')
                LRP_DAYS  = ["None"] + WEEKDAY_LABELS
                LRP_TYPES = ["easy","tempo","long","medium-long","intervals"]
                with gr.Row():
                    lrp1_day_in  = gr.Dropdown(LRP_DAYS, label="Session 1 day", value=_lrp1_day)
                    lrp1_km_in   = gr.Number(label="km", value=_lrp1_km, precision=1, minimum=1, maximum=45)
                    lrp1_type_in = gr.Dropdown(LRP_TYPES, label="type", value=_lrp1_type)

                lrp2_visible_state = gr.State(value=_has_lrp2)
                lrp3_visible_state = gr.State(value=_has_lrp3)
                lrp4_visible_state = gr.State(value=_has_lrp4)

                add_lrp2_btn = gr.Button("+ Add session", size="sm", variant="secondary",
                                         visible=not _has_lrp2)
                with gr.Group(visible=_has_lrp2) as lrp2_group:
                    with gr.Row():
                        lrp2_day_in  = gr.Dropdown(LRP_DAYS, label="Session 2 day", value=_lrp2_day)
                        lrp2_km_in   = gr.Number(label="km", value=_lrp2_km, precision=1, minimum=1, maximum=45)
                        lrp2_type_in = gr.Dropdown(LRP_TYPES, label="type", value=_lrp2_type)
                        remove_lrp2_btn = gr.Button("✕", size="sm", variant="secondary")

                add_lrp3_btn = gr.Button("+ Add session", size="sm", variant="secondary",
                                         visible=(_has_lrp2 and not _has_lrp3))
                with gr.Group(visible=_has_lrp3) as lrp3_group:
                    with gr.Row():
                        lrp3_day_in  = gr.Dropdown(LRP_DAYS, label="Session 3 day", value=_lrp3_day)
                        lrp3_km_in   = gr.Number(label="km", value=_lrp3_km, precision=1, minimum=1, maximum=45)
                        lrp3_type_in = gr.Dropdown(LRP_TYPES, label="type", value=_lrp3_type)
                        remove_lrp3_btn = gr.Button("✕", size="sm", variant="secondary")

                add_lrp4_btn = gr.Button("+ Add session", size="sm", variant="secondary",
                                         visible=(_has_lrp3 and not _has_lrp4))
                with gr.Group(visible=_has_lrp4) as lrp4_group:
                    with gr.Row():
                        lrp4_day_in  = gr.Dropdown(LRP_DAYS, label="Session 4 day", value=_lrp4_day)
                        lrp4_km_in   = gr.Number(label="km", value=_lrp4_km, precision=1, minimum=1, maximum=45)
                        lrp4_type_in = gr.Dropdown(LRP_TYPES, label="type", value=_lrp4_type)
                        remove_lrp4_btn = gr.Button("✕", size="sm", variant="secondary")

                gr.HTML('<div class="section-label"><div class="section-label-text">Garmin Connection</div></div>')
                garmin_status_s = gr.HTML()
                with gr.Group() as garmin_login_s:
                    with gr.Row():
                        garmin_email_s = gr.Textbox(
                            label="Garmin email",
                            value=garmin_mod.load_email() or "",
                            placeholder="you@example.com", scale=2)
                        garmin_pass_s = gr.Textbox(
                            label="Password", type="password", scale=2)
                        garmin_conn_s = gr.Button("Connect", size="sm", scale=1)
                with gr.Group(visible=False) as garmin_mfa_s:
                    with gr.Row():
                        garmin_mfa_code_s = gr.Textbox(
                            label="Verification code", placeholder="6-digit code", scale=2)
                        garmin_mfa_btn_s = gr.Button("Submit", size="sm", scale=1)
                with gr.Group(visible=False) as garmin_connected_s:
                    with gr.Row():
                        garmin_import_s    = gr.Button("📥  Import last 30 days", size="sm", variant="primary")
                        garmin_sync_s      = gr.Button("🔄  Sync last 3 days",   size="sm", variant="secondary")
                        garmin_disconnect_s = gr.Button("Disconnect", size="sm", variant="secondary")
                garmin_msg_s = gr.HTML()

                gen_btn = gr.Button("💾  Save & Generate Plan", variant="primary", size="lg")
                gen_msg = gr.HTML()
                zones_out = gr.JSON(label="Training Zones", visible=False)

            # ── Panel 2: Activity Log ─────────────────────────────────────
            with gr.Group(visible=False, elem_id="panel-history") as panel_history:
                gr.HTML("""<div class="page-header">
                    <span class="page-header-icon">🏃</span>
                    <div>
                        <div class="page-header-title">Activity Log</div>
                        <div class="page-header-sub">Sync from Garmin or upload FIT files · table below</div>
                    </div></div>""")

                hist_df_state   = gr.State(value=None)
                hist_page_state = gr.State(value=0)

                # ── Top row: Garmin left, FIT upload right ────────────────
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=280):
                        gr.HTML('<div class="section-label"><div class="section-label-text">Garmin Connect</div></div>')
                        garmin_status_h = gr.HTML()
                        with gr.Group() as garmin_login_h:
                            with gr.Row():
                                garmin_email_h = gr.Textbox(
                                    label="Garmin email",
                                    value=garmin_mod.load_email() or "",
                                    placeholder="you@example.com", scale=2)
                                garmin_pass_h = gr.Textbox(
                                    label="Password", type="password", scale=2)
                                garmin_conn_h = gr.Button("Connect", size="sm", scale=1)
                        with gr.Group(visible=False) as garmin_mfa_h:
                            with gr.Row():
                                garmin_mfa_code_h = gr.Textbox(
                                    label="Verification code", placeholder="6-digit code", scale=2)
                                garmin_mfa_btn_h = gr.Button("Submit", size="sm", scale=1)
                        with gr.Group(visible=False) as garmin_connected_h:
                            with gr.Row():
                                garmin_import_btn = gr.Button("📥  Import last 30 days", size="sm", variant="primary")
                                garmin_sync_btn   = gr.Button("🔄  Sync last 3 days",   size="sm", variant="secondary")
                                garmin_disconnect_h = gr.Button("Disconnect", size="sm", variant="secondary")
                        garmin_msg_h = gr.HTML()

                    with gr.Column(scale=2, min_width=200):
                        gr.HTML('<div class="section-label"><div class="section-label-text">Upload FIT file</div></div>')
                        fit_upload = gr.File(label="Drop .fit file here", file_types=[".fit"])
                        fit_msg    = gr.HTML()

                # ── Activity table + pagination ───────────────────────────
                gr.HTML('<div class="section-label" style="margin-top:16px"><div class="section-label-text">Activity history</div></div>')
                with gr.Row():
                    hist_prev_btn = gr.Button("← Prev", size="sm")
                    hist_page_info = gr.HTML(
                        "<div style='text-align:center;padding:6px;font-size:12px;color:#6B7280'>—</div>",
                        elem_id="hist-page-info",
                    )
                    hist_next_btn = gr.Button("Next →", size="sm")
                hist_html = gr.HTML(visible=True)

                def upload_fit(f):
                    if f is None:
                        return None, ""
                    path = f.name if hasattr(f, "name") else str(f)
                    rec  = fit_mod.summarize(path)
                    if not rec or "error" in rec:
                        return None, f"<span style='color:#DC2626'>Could not parse: {rec}</span>"
                    s    = state_mod.load()
                    cyc  = active_cycle(s)
                    hr_max  = s.get("hr_max", 177)
                    hr_rest = s.get("hr_rest", 50)
                    zd = cyc.get("zones", {}) if cyc else {}
                    z  = Zones(**{k: v for k, v in zd.items() if k in Zones.__dataclass_fields__}) if zd else None
                    rec["training_load"] = compute_load(rec, hr_max, hr_rest, z)
                    df, msg = _merge_into_history([rec])
                    return df, f"<span style='color:#16A34A'>{msg}</span>"

                fit_upload.upload(upload_fit, inputs=fit_upload,
                                  outputs=[hist_df_state, fit_msg]).then(
                    _hist_reset_page, inputs=hist_df_state,
                    outputs=[hist_html, hist_page_state, hist_page_info])

                gr.HTML('<div class="section-label" style="margin-top:24px">'
                        '<div class="section-label-text">Session debrief history</div>'
                        '<div class="section-label-sub">All debriefed sessions · newest week first</div></div>')
                debrief_history_html = gr.HTML()

            # ── Panel 3: Adjustments ──────────────────────────────────────
            with gr.Group(visible=False, elem_id="panel-adj") as panel_adj:
                gr.HTML("""<div class="page-header">
                    <span class="page-header-icon">✏️</span>
                    <div>
                        <div class="page-header-title">Adjustments</div>
                        <div class="page-header-sub">Physio / travel / illness overrides · plan ahead</div>
                    </div></div>""")
                with gr.Row():
                    adj_from_in  = gr.Number(label="Start at plan week",
                                             value=_current_week_num, precision=0,
                                             info="Which plan week to start from (1 = first week)")
                    adj_num_in   = gr.Number(label="Duration (weeks)",
                                             value=1, precision=0,
                                             info="How many weeks to affect — 0 = all remaining weeks")
                with gr.Row():
                    adj_no_lrp_in    = gr.Checkbox(label="Skip LRP club runs (replace with easy)")
                    adj_easy_only_in = gr.Checkbox(label="Easy runs only (remove quality sessions)")
                adj_volume_in   = gr.Slider(60, 110, value=100, step=5, label="Volume %",
                                            info="Reduce to 80% for a recovery week, 60% for illness/travel")
                adj_sat_km_in   = gr.Number(label="Saturday long run km override",
                                            value=None, precision=1,
                                            info="Pin the Saturday LRP distance — leave blank to use the plan suggestion")
                adj_msg_in      = gr.Textbox(label="Message to coach (optional)", lines=2,
                                             placeholder="e.g. I have a race next weekend / travelling / physio said rest")
                adj_btn         = gr.Button("Apply & get coaching note", variant="primary")
                adj_note_out    = gr.Markdown(label="Coaching note")
                adj_status_out  = gr.HTML()

                adj_btn.click(
                    apply_adjustment_handler,
                    inputs=[adj_from_in, adj_num_in, adj_no_lrp_in,
                            adj_easy_only_in, adj_volume_in, adj_sat_km_in, adj_msg_in],
                    outputs=[adj_note_out, adj_status_out],
                )

            # ── Panel 4: Weekly Check-in ──────────────────────────────────
            with gr.Group(visible=False, elem_id="panel-checkin") as panel_checkin:
                gr.HTML("""<div class="page-header">
                    <span class="page-header-icon">✅</span>
                    <div>
                        <div class="page-header-title">Weekly Check-in</div>
                        <div class="page-header-sub">How are you feeling · get next week's proposal · coaching note</div>
                    </div></div>""")

                gr.HTML('<div class="section-label"><div class="section-label-text">This week\'s plan</div></div>')
                checkin_week_html = gr.HTML()

                gr.HTML('<div class="section-label"><div class="section-label-text">Session debrief</div>'
                        '<div class="section-label-sub">Select a session · add how it felt · save daily</div></div>')
                debrief_session_dd = gr.Dropdown(choices=[], label="Select session", interactive=True)
                debrief_session_info = gr.HTML()
                with gr.Row():
                    debrief_rpe = gr.Slider(minimum=1, maximum=10, step=0.5, value=5,
                                            label="RPE (1–10)",
                                            info="1 = very easy · 5 = moderate · 10 = max effort")
                    debrief_comment = gr.Textbox(label="Comment (optional)", lines=2,
                                                 placeholder="How did it feel? Any aches, adjustments…")
                debrief_save_btn = gr.Button("Save debrief for this session", variant="secondary")
                gr.HTML('<div class="section-label"><div class="section-label-text">Debriefed sessions</div>'
                        '<div class="section-label-sub">Updates each time you save a session</div></div>')
                debrief_cards = gr.HTML()
                debrief_comp  = gr.HTML()

                debrief_session_dd.change(
                    session_info_handler,
                    inputs=[debrief_session_dd],
                    outputs=[debrief_rpe, debrief_comment, debrief_session_info],
                )
                debrief_save_btn.click(
                    debrief_session_handler,
                    inputs=[debrief_session_dd, debrief_rpe, debrief_comment],
                    outputs=[debrief_cards, debrief_comp],
                )

                gr.HTML('<div class="section-label"><div class="section-label-text">How did the week go?</div></div>')
                feeling_in = gr.Slider(minimum=1, maximum=5, step=0.5, value=3,
                                       label="Overall feeling",
                                       info="1 = exhausted / sick  ·  3 = normal  ·  5 = great, ready for more")
                comments_in = gr.Textbox(label="Debrief (optional)",
                                         lines=3, placeholder="Highlights, aches, travel, life stress… FR/EN/ES ok")

                gr.HTML('<div class="section-label"><div class="section-label-text">Next week availability</div><div class="section-label-sub">Untick any day you cannot run — LRP days (Mon/Sat) will be skipped automatically if unticked</div></div>')
                avail_in = gr.CheckboxGroup(
                    WEEKDAY_LABELS,
                    label="Days available",
                    value=_saved_run_days,
                )

                checkin_injury_in = gr.Dropdown(
                    ["none", "light", "moderate"],
                    value=_sched.get("injury_level", "none"),
                    label="Injury / rehab status",
                    info="Changing this regenerates the whole plan — none = full training · light = 85% volume · moderate = 70%, easy only",
                )

                checkin_btn     = gr.Button("Generate next week", variant="primary", size="lg")
                checkin_status  = gr.HTML()
                checkin_metrics = gr.HTML()
                gr.HTML('<div class="section-label"><div class="section-label-text">Coaching note</div></div>')
                coaching_out    = gr.Markdown(label="")
                gr.HTML('<div class="section-label"><div class="section-label-text">Next week proposal</div></div>')
                checkin_plan_out = gr.HTML()

                checkin_btn.click(
                    checkin_handler,
                    inputs=[feeling_in, comments_in, avail_in, checkin_injury_in],
                    outputs=[checkin_status, checkin_metrics, coaching_out, checkin_plan_out],
                ).then(load_status_on_start, outputs=status_bar)

            # ── Panel 5: My Zones ─────────────────────────────────────────
            with gr.Group(visible=False, elem_id="panel-zones") as panel_zones:
                gr.HTML("""<div class="page-header">
                    <span class="page-header-icon">📊</span>
                    <div>
                        <div class="page-header-title">My Zones</div>
                        <div class="page-header-sub">Pace targets for every workout type · Heart rate zones</div>
                    </div></div>""")
                zones_vdot_header = gr.HTML()
                with gr.Row():
                    hr_max_zones_in = gr.Number(label="Max heart rate (bpm)",
                                                precision=0, minimum=130, maximum=230, value=177,
                                                info="Highest HR reached in a max effort", scale=1)
                    hr_rest_zones_in = gr.Number(label="Resting heart rate (bpm)",
                                                 precision=0, minimum=30, maximum=100, value=50,
                                                 info="Morning resting HR · used for Karvonen zones", scale=1)
                gr.HTML('<div class="section-label"><div class="section-label-text">Pace zones</div></div>')
                zones_pace_out = gr.HTML()
                gr.HTML('<div class="section-label"><div class="section-label-text">Heart rate zones</div></div>')
                zones_hr_out = gr.HTML()

            gr.HTML(f'<div id="app-footer">LRP Coach · v{VERSION}</div>')

    # ── Nav wiring ─────────────────────────────────────────────────────────
    _panels = [panel_plan, panel_setup, panel_history, panel_adj, panel_checkin, panel_zones]
    _btns   = [btn_plan, btn_setup, btn_hist, btn_adj, btn_checkin, btn_zones]
    for _i, _btn in enumerate(_btns):
        ev = _btn.click(_nav_handler(_i), outputs=_panels + _btns)
        if _i == 2:  # activity log tab: load debrief history
            ev.then(load_debrief_history, outputs=[debrief_history_html])
        if _i == 4:  # check-in tab: load week plan, injury, session dropdown, cards
            ev.then(load_checkin_panel,
                    outputs=[checkin_week_html, checkin_injury_in,
                             debrief_session_dd, debrief_cards, debrief_comp])

    # ── Plan week navigation ────────────────────────────────────────────────
    _PLAN_LOOKAHEAD = 2   # weeks beyond plan end that show a projected stub

    def _plan_prev(idx, view):
        idx = int(idx or 0)
        if idx <= 0:
            return (
                "<div style='text-align:center;padding:56px;color:#9CA3AF;"
                "font-family:-apple-system,sans-serif'>"
                "<div style='font-size:13px;color:#374151'>← No earlier weeks</div>"
                "</div>",
                0,
            )
        new_idx = idx - 1
        return _render_plan(new_idx, view), new_idx

    def _plan_next(idx, view):
        s    = state_mod.load()
        plan = s.get("plan", [])
        idx  = int(idx or 0)
        max_idx = len(plan) - 1 + _PLAN_LOOKAHEAD
        new_idx = min(max_idx, idx + 1)
        if new_idx >= len(plan):
            weeks_ahead = new_idx - len(plan) + 1
            return (
                "<div style='background:#FFFBEB;border:1px solid #FCD34D;border-radius:10px;"
                "padding:24px 28px;font-family:-apple-system,sans-serif'>"
                "<div style='font-size:13px;font-weight:700;color:#92400E;margin-bottom:6px'>"
                f"Week {weeks_ahead} beyond current plan — projected</div>"
                "<div style='font-size:12px;color:#78350F'>"
                "This week will be finalised after your next Weekly Check-in. "
                "Structure follows the current phase; distances adapt to your fitness.</div>"
                "</div>",
                new_idx,
            )
        return _render_plan(new_idx, view), new_idx

    def _plan_this(view):
        s    = state_mod.load()
        plan = s.get("plan", [])
        idx  = _current_week_idx(plan)
        return _render_plan(idx, view), idx

    plan_prev_btn.click(_plan_prev, inputs=[plan_week_state, cal_view_radio], outputs=[plan_html, plan_week_state])
    plan_next_btn.click(_plan_next, inputs=[plan_week_state, cal_view_radio], outputs=[plan_html, plan_week_state])
    plan_this_btn.click(_plan_this, inputs=[cal_view_radio], outputs=[plan_html, plan_week_state])

    # ── View toggle ─────────────────────────────────────────────────────────
    cal_view_radio.change(
        lambda view, idx: _render_plan(int(idx or 0), view),
        inputs=[cal_view_radio, plan_week_state],
        outputs=[plan_html],
    )

    # ── Drag-drop move handler ───────────────────────────────────────────────
    def _move_handler(bridge_val: str, week_idx: int, view: str, undo_stack: list):
        if not bridge_val or "|" not in bridge_val:
            return gr.update(), undo_stack, "", gr.update()
        parts = bridge_val.split("|")
        if len(parts) < 2:
            return gr.update(), undo_stack, "", gr.update()
        from_date, to_date = parts[0], parts[1]   # parts[2] is the timestamp noise
        s   = state_mod.load()
        cyc = state_mod.active_cycle(s)
        if not cyc:
            return gr.update(), undo_stack, "", gr.update()
        old_plan  = cyc.get("plan", [])
        new_stack = (list(undo_stack) + [{"plan": old_plan}])[-10:]
        new_plan, warnings = editplan_mod.move_session(old_plan, from_date, to_date)
        cyc["plan"] = new_plan
        s = state_mod.set_active_cycle(s, cyc)
        state_mod.save(s)
        warn_html = ""
        if warnings:
            items = "".join(f"<li>{w}</li>" for w in warnings)
            warn_html = (
                f"<div style='background:#FFFBEB;border:1px solid #FCD34D;"
                f"border-radius:8px;padding:10px 14px;font-size:12px;color:#92400E;"
                f"margin-top:8px'><b>Schedule note:</b>"
                f"<ul style='margin:4px 0 0 16px;padding:0'>{items}</ul></div>"
            )
        html = _render_plan(int(week_idx or 0), view)
        undo_visible = gr.update(visible=bool(new_stack))
        return html, new_stack, warn_html, undo_visible

    cal_move_bridge.input(
        _move_handler,
        inputs=[cal_move_bridge, plan_week_state, cal_view_radio, cal_undo_stack],
        outputs=[plan_html, cal_undo_stack, cal_warnings, cal_undo_btn],
    )

    # ── Undo handler ────────────────────────────────────────────────────────
    def _undo_handler(undo_stack: list, week_idx: int, view: str):
        if not undo_stack:
            return gr.update(), undo_stack, "", gr.update()
        new_stack = list(undo_stack)
        prev = new_stack.pop()
        s   = state_mod.load()
        cyc = state_mod.active_cycle(s)
        if not cyc:
            return gr.update(), undo_stack, "", gr.update()
        cyc["plan"] = prev["plan"]
        s = state_mod.set_active_cycle(s, cyc)
        state_mod.save(s)
        html = _render_plan(int(week_idx or 0), view)
        undo_visible = gr.update(visible=bool(new_stack))
        return html, new_stack, "", undo_visible

    cal_undo_btn.click(
        _undo_handler,
        inputs=[cal_undo_stack, plan_week_state, cal_view_radio],
        outputs=[plan_html, cal_undo_stack, cal_warnings, cal_undo_btn],
    )

    # ── LRP session add/remove wiring ──────────────────────────────────────
    add_lrp2_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), True),
        outputs=[lrp2_group, add_lrp2_btn, lrp2_visible_state])
    remove_lrp2_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), False,
                 gr.update(visible=False), gr.update(visible=True), False,
                 gr.update(visible=False), gr.update(visible=True), False),
        outputs=[lrp2_group, add_lrp2_btn, lrp2_visible_state,
                 lrp3_group, add_lrp3_btn, lrp3_visible_state,
                 lrp4_group, add_lrp4_btn, lrp4_visible_state])
    add_lrp3_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), True),
        outputs=[lrp3_group, add_lrp3_btn, lrp3_visible_state])
    remove_lrp3_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), False,
                 gr.update(visible=False), gr.update(visible=True), False),
        outputs=[lrp3_group, add_lrp3_btn, lrp3_visible_state,
                 lrp4_group, add_lrp4_btn, lrp4_visible_state])
    add_lrp4_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False), True),
        outputs=[lrp4_group, add_lrp4_btn, lrp4_visible_state])
    remove_lrp4_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True), False),
        outputs=[lrp4_group, add_lrp4_btn, lrp4_visible_state])

    # ── Generate plan ───────────────────────────────────────────────────────
    def _gen_and_nav(*args):
        zd, plan_html_val, msg, summary = compute_and_generate(*args)
        s = state_mod.load()
        plan = s.get("plan", [])
        idx  = _current_week_idx(plan)
        html = _render_plan(idx, "List")
        ok = bool(plan)
        status_html = (
            f"<div style='background:#DCFCE7;border:1px solid #16A34A;border-radius:8px;"
            f"padding:10px 14px;font-size:13px;color:#166534'>{msg}</div>"
            if ok else
            f"<div style='background:#FEE2E2;border:1px solid #DC2626;border-radius:8px;"
            f"padding:10px 14px;font-size:13px;color:#7F1D1D'>{msg}</div>"
        )
        return status_html, html, idx, load_status_on_start()

    gen_btn.click(
        _gen_and_nav,
        inputs=[
            name_in, goal_race_in, marathon_date_in,
            goal_h, goal_m, goal_s,
            b1_dist, b1_h, b1_m, b1_s, b1_date_in,
            b2_dist, b2_h, b2_m, b2_s, b2_date_in,
            injury_in, injury_notes_in,
            run_days_in, min_runs_in, max_runs_in, allow_volume_in,
            lrp1_day_in, lrp1_km_in, lrp1_type_in,
            lrp2_day_in, lrp2_km_in, lrp2_type_in, lrp2_visible_state,
            lrp3_day_in, lrp3_km_in, lrp3_type_in, lrp3_visible_state,
            lrp4_day_in, lrp4_km_in, lrp4_type_in, lrp4_visible_state,
            strength_in, cycling_in, hr_rest_setup_in,
        ],
        outputs=[gen_msg, plan_html, plan_week_state, status_bar],
    )

    # ── My Zones ────────────────────────────────────────────────────────────
    btn_zones.click(load_zones_tab,
                    outputs=[zones_vdot_header, hr_max_zones_in, hr_rest_zones_in,
                             zones_pace_out, zones_hr_out])
    hr_max_zones_in.change(update_zones_hr,
                           inputs=[hr_max_zones_in, hr_rest_zones_in],
                           outputs=zones_hr_out)
    hr_rest_zones_in.change(update_zones_hr,
                            inputs=[hr_max_zones_in, hr_rest_zones_in],
                            outputs=zones_hr_out)

    # ── Activity log pagination ─────────────────────────────────────────────
    hist_prev_btn.click(_hist_prev_page, inputs=[hist_df_state, hist_page_state],
                        outputs=[hist_html, hist_page_state, hist_page_info])
    hist_next_btn.click(_hist_next_page, inputs=[hist_df_state, hist_page_state],
                        outputs=[hist_html, hist_page_state, hist_page_info])
    btn_hist.click(load_history_on_start,
                   outputs=[hist_df_state]).then(
        _hist_reset_page, inputs=hist_df_state,
        outputs=[hist_html, hist_page_state, hist_page_info])

    # ── Garmin wiring — both panels share same handlers ─────────────────────
    # outputs order: s_status, s_login, s_mfa, s_connected, s_msg,
    #                h_status, h_login, h_mfa, h_connected, h_msg
    _g_both = [garmin_status_s, garmin_login_s, garmin_mfa_s, garmin_connected_s, garmin_msg_s,
               garmin_status_h, garmin_login_h, garmin_mfa_h, garmin_connected_h, garmin_msg_h]

    garmin_conn_s.click(garmin_connect_ui,
                        inputs=[garmin_email_s, garmin_pass_s],
                        outputs=_g_both)
    garmin_conn_h.click(garmin_connect_ui,
                        inputs=[garmin_email_h, garmin_pass_h],
                        outputs=_g_both)

    garmin_mfa_btn_s.click(garmin_mfa_ui,
                           inputs=[garmin_email_s, garmin_pass_s, garmin_mfa_code_s],
                           outputs=_g_both)
    garmin_mfa_btn_h.click(garmin_mfa_ui,
                           inputs=[garmin_email_h, garmin_pass_h, garmin_mfa_code_h],
                           outputs=_g_both)

    garmin_disconnect_s.click(garmin_disconnect_ui, outputs=_g_both)
    garmin_disconnect_h.click(garmin_disconnect_ui, outputs=_g_both)

    def _garmin_import():
        df, msg = garmin_import_ui()
        html, pg, info = _hist_reset_page(df)
        status = f"<span style='color:#16A34A'>{msg}</span>"
        return df, html, pg, info, status, status

    def _garmin_sync():
        df, msg = garmin_sync_new_ui()
        html, pg, info = _hist_reset_page(df)
        status = f"<span style='color:#16A34A'>{msg}</span>"
        return df, html, pg, info, status, status

    garmin_import_btn.click(_garmin_import,
                            outputs=[hist_df_state, hist_html, hist_page_state,
                                     hist_page_info, garmin_msg_s, garmin_msg_h])
    garmin_sync_btn.click(_garmin_sync,
                          outputs=[hist_df_state, hist_html, hist_page_state,
                                   hist_page_info, garmin_msg_s, garmin_msg_h])
    garmin_import_s.click(_garmin_import,
                          outputs=[hist_df_state, hist_html, hist_page_state,
                                   hist_page_info, garmin_msg_s, garmin_msg_h])
    garmin_sync_s.click(_garmin_sync,
                        outputs=[hist_df_state, hist_html, hist_page_state,
                                 hist_page_info, garmin_msg_s, garmin_msg_h])

    # ── Page load ───────────────────────────────────────────────────────────
    demo.load(load_status_on_start, outputs=status_bar)
    demo.load(load_plan_on_start,   outputs=[plan_html, plan_week_state])
    demo.load(load_history_on_start, outputs=hist_df_state).then(
        _hist_reset_page, inputs=hist_df_state,
        outputs=[hist_html, hist_page_state, hist_page_info])
    demo.load(load_garmin_ui, outputs=_g_both)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860,
                        allowed_paths=["scripts"], show_error=True)
