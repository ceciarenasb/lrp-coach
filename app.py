"""
LRP Coach — Gradio Blocks UI.
Five tabs, explicit linear flow, state never silently lost.
Garmin auth/sync code is kept exactly as in the original.
"""

from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import gradio as gr
import pandas as pd

import coach.state as state_mod
from coach.decide import propose_week
from coach.fit import summarize as fit_summarize
from coach.garmin import (
    clear_auth, connect, connection_status,
    is_authenticated, load_email, submit_mfa, sync_activities,
)
from coach.load import compute as compute_load
from coach.llm import build_context, coaching_note
from coach.plan import generate_plan
from coach.readiness import label as readiness_label
from coach.state import active_cycle, new_cycle, set_active_cycle
from coach.state_model import apply_history, from_dict as sm_from_dict, to_dict as sm_to_dict
from coach.zones import (
    Zones,
    build_zones, hr_zones, marathon_time_from_vdot, pace_zones_extended,
    vdot_from_race, zones_summary,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKDAY_MAP    = {l: i for i, l in enumerate(WEEKDAY_LABELS)}
DISTANCE_OPTS  = ["marathon (42.195 km)", "half (21.098 km)", "10 K (10 km)", "custom"]
DISTANCE_KM    = {
    "marathon (42.195 km)": 42.195,
    "half (21.098 km)":     21.0975,
    "10 K (10 km)":         10.0,
}
DISTANCE_LABEL = {
    "marathon (42.195 km)": "marathon",
    "half (21.098 km)":     "half",
    "10 K (10 km)":         "10K",
    "custom":               "custom",
}

_HIST_PER_PAGE = 15
_PHASE_COLORS  = {
    "Base":  "#059669", "Build": "#2563EB",
    "Peak":  "#EA580C", "Taper": "#64748B",
}
_SESSION_COLOR = {
    "Easy": "#10B981", "Recovery": "#34D399", "Long Run": "#2563EB",
    "Medium-Long": "#3B82F6", "Tempo": "#F5871F", "SVC Intervals": "#DC2626",
    "Marathon Pace": "#7C3AED", "Progression Run": "#8B5CF6",
    "Rest": "#9CA3AF", "Strength": "#D97706", "Cycling / Zwift": "#0891B2",
    "Club Run (LRP)": "#0D9488",
}


def _fmt_duration(s: int) -> str:
    if not s:
        return "—"
    h, r = divmod(int(s), 3600)
    m, _ = divmod(r, 60)
    return f"{h}:{m:02d}:00"


def _fmt_pace(sec: int | None) -> str:
    if not sec or sec <= 0:
        return "n/a"
    m, s = divmod(int(sec), 60)
    return f"{m}:{s:02d} /km"


def _iso_to_dmy(iso: str) -> str:
    try:
        d = date.fromisoformat(iso)
        return d.strftime("%d %b %Y")
    except Exception:
        return iso or "—"


def _check_gradio_client_patch():
    try:
        import gradio_client.utils as _u
        src = Path(_u.__file__).read_text()
        if "isinstance(schema, bool)" not in src:
            for fn in ("get_type", "_json_schema_to_python_type"):
                src = src.replace(
                    f"def {fn}(schema",
                    f"def {fn}(schema",
                )
        # Patch already applied by earlier session or not needed
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Plan HTML rendering
# ---------------------------------------------------------------------------

def _hr_text(sess_type: str, desc: str, zones: list) -> str:
    if not zones or sess_type in ("Rest", "Strength", "Cycling / Zwift"):
        return "—"
    def z(n):
        i = zones[n - 1]
        return f"{i['name']} ({i['lo']}–{i['hi']})"
    if sess_type == "Recovery":
        return f"Throughout: {z(1)}"
    if sess_type == "Easy":
        return f"Throughout: {z(1)} – {z(2)}"
    if sess_type == "Long Run":
        return (f"Main: {z(1)}–{z(2)} · M-pace finish: {z(3)}"
                if "M-pace" in desc else f"Throughout: {z(1)} – {z(2)}")
    if sess_type == "Medium-Long":
        return (f"Easy: {z(1)}–{z(2)} · M-pace: {z(3)}"
                if "M-pace" in desc else f"Throughout: {z(2)} – {z(3)}")
    if sess_type == "Tempo":
        return f"WU: {z(1)} · Effort: {z(4)} · CD: {z(1)}"
    if sess_type == "SVC Intervals":
        return f"WU: {z(1)}–{z(2)} · Intervals: {z(5)} · Récup: {z(1)} · CD: {z(2)}"
    if sess_type == "Marathon Pace":
        return f"WU: {z(1)} · M-pace: {z(3)} · CD: {z(1)}"
    if sess_type == "Progression Run":
        return f"Easy: {z(1)} · M-pace: {z(3)} · T-pace: {z(4)}"
    if sess_type == "Club Run (LRP)":
        return (f"WU: {z(1)} · Effort: {z(4)}–{z(5)} · CD: {z(1)}"
                if "tempo" in desc.lower() else f"Throughout: {z(1)} – {z(2)}")
    return f"{z(1)} – {z(2)}"


def _plan_phase_html(plan: list, phase: str, hr_max: int = 177, hr_rest: int = 50) -> str:
    _EMPTY = (
        "<div style='text-align:center;padding:56px;color:#9CA3AF;"
        "font-family:-apple-system,sans-serif'>"
        "<div style='font-size:44px'>🏃</div>"
        "<div style='font-size:15px;font-weight:600;color:#374151;margin-top:8px'>No plan yet</div>"
        "<div style='font-size:13px;margin-top:4px'>Complete Setup to generate your plan.</div>"
        "</div>"
    )
    if not plan:
        return _EMPTY
    zones_hr = hr_zones(hr_max, hr_rest)
    weeks    = [w for w in plan if w.get("phase") == phase]
    if not weeks:
        avail = sorted({w["phase"] for w in plan},
                       key=lambda p: ["Base","Build","Peak","Taper"].index(p)
                       if p in ["Base","Build","Peak","Taper"] else 99)
        return (f"<p style='color:#9CA3AF;padding:24px'>No {phase} weeks. "
                f"Available: {', '.join(avail)}</p>")

    accent = _PHASE_COLORS.get(phase, "#64748B")
    focus  = weeks[0].get("focus", "")
    header = (
        f"<div style='background:{accent};border-radius:10px 10px 0 0;"
        f"padding:10px 16px;color:#fff;font-family:-apple-system,sans-serif'>"
        f"<span style='font-size:14px;font-weight:700;text-transform:uppercase;"
        f"letter-spacing:.06em'>{phase}</span>"
        f"<span style='font-size:12px;opacity:.85;margin-left:12px'>{focus}</span>"
        f"<span style='float:right;font-size:11px;opacity:.75'>"
        f"Weeks {weeks[0]['week_num']}–{weeks[-1]['week_num']}</span>"
        f"</div>"
    )
    _th = ("padding:10px;text-align:left;font-weight:600;font-size:11px;"
           "letter-spacing:.05em;text-transform:uppercase;color:#fff;background:#1B2874")
    rows = ""
    prev_wk = None
    for idx, w in enumerate(weeks):
        for day in w["days"]:
            d    = date.fromisoformat(day["date"])
            sess = day["session_type"]
            desc = day["description"]
            km   = f"{day['distance_km']:.0f} km" if day.get("distance_km") else "—"
            tgts = " · ".join(f"{k}: {v}" for k, v in day.get("targets", {}).items())
            detail = f"{desc}<br><span style='color:#9CA3AF;font-size:11px'>{tgts}</span>" if tgts else desc
            sep  = "border-top:2px solid #E5E7EB;" if w["week_num"] != prev_wk else ""
            bg   = "#fff" if idx % 2 == 0 else "#FAFAFA"
            prev_wk = w["week_num"]
            badge   = _SESSION_COLOR.get(sess, "#6B7280")
            hr_txt  = _hr_text(sess, desc, zones_hr)
            rows += (
                f"<tr style='background:{bg};{sep}'>"
                f"<td style='padding:7px 10px;color:#9CA3AF;font-size:12px;"
                f"white-space:nowrap;border-left:3px solid {accent}'>"
                f"{d.day} {d.strftime('%b')}</td>"
                f"<td style='padding:7px 10px;font-weight:600;color:#374151;"
                f"white-space:nowrap'>{d.strftime('%a')}</td>"
                f"<td style='padding:7px 10px;white-space:nowrap'>"
                f"<span style='background:{badge};color:#fff;padding:2px 9px;"
                f"border-radius:99px;font-size:11px;font-weight:700'>{sess}</span></td>"
                f"<td style='padding:7px 10px;color:#374151;font-size:12px'>{detail}</td>"
                f"<td style='padding:7px 10px;color:#374151;font-size:11px;"
                f"line-height:1.5'>{hr_txt}</td>"
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


# ---------------------------------------------------------------------------
# History HTML helpers
# ---------------------------------------------------------------------------

def _format_history_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    def _fmt_date(v):
        parts = str(v).split("-")
        if len(parts) == 3 and len(parts[0]) == 4:
            return f"{parts[2]}-{parts[1]}-{parts[0]}"
        return v
    if "date" in d.columns:
        d["date"] = d["date"].apply(_fmt_date)
    if "activity_type" in d.columns:
        d["activity_type"] = d["activity_type"].fillna("Run").replace("", "Run")
    cols = ["date", "activity_type", "distance_km", "avg_pace_s", "avg_hr", "training_load"]
    d = d[[c for c in cols if c in d.columns]]
    if "avg_pace_s" in d.columns:
        d["pace"] = d["avg_pace_s"].apply(
            lambda v: _fmt_pace(v) if pd.notna(v) and v else "—"
        )
        d = d.drop(columns=["avg_pace_s"])
    return d.rename(columns={
        "date": "Date", "activity_type": "Type",
        "distance_km": "Km", "avg_hr": "Avg HR",
        "training_load": "Load", "pace": "Pace",
    })


def _hist_page_html(df: pd.DataFrame, page: int) -> tuple:
    if df is None or df.empty:
        return "<p style='color:#9CA3AF;padding:16px'>No activity history yet.</p>", 0, ""
    total = len(df)
    n_pages = max(1, (total + _HIST_PER_PAGE - 1) // _HIST_PER_PAGE)
    page = max(0, min(page, n_pages - 1))
    chunk = df.iloc[page * _HIST_PER_PAGE:(page + 1) * _HIST_PER_PAGE]
    th = ("padding:9px 12px;text-align:left;font-size:11px;font-weight:600;"
          "color:#6B7280;text-transform:uppercase;letter-spacing:.04em;"
          "background:#F9FAFB;border-bottom:1px solid #E5E7EB")
    rows = ""
    for i, (_, row) in enumerate(chunk.iterrows()):
        bg = "#fff" if i % 2 == 0 else "#F5F7FA"
        cells = "".join(
            f"<td style='padding:8px 12px;color:#374151;font-size:13px'>{v}</td>"
            for v in row.values
        )
        rows += f"<tr style='background:{bg}'>{cells}</tr>"
    headers = "".join(f"<th style='{th}'>{c}</th>" for c in chunk.columns)
    html = (
        "<div style='overflow-x:auto;border-radius:8px;border:1px solid #E5E7EB;"
        "font-family:-apple-system,sans-serif'>"
        f"<table style='width:100%;border-collapse:collapse'>"
        f"<thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>"
    )
    info = f"<span style='font-size:12px;color:#6B7280'>Page {page+1} / {n_pages} · {total} activities</span>"
    return html, page, info


# ---------------------------------------------------------------------------
# Startup loaders
# ---------------------------------------------------------------------------

def _startup_state():
    return state_mod.load()


def load_plan_on_start():
    s    = state_mod.load()
    cyc  = active_cycle(s)
    plan = cyc.get("plan", []) if cyc else []
    hr_max  = s.get("athlete", {}).get("hr_max", 177)
    hr_rest = s.get("athlete", {}).get("hr_rest", 50)
    phases  = [w["phase"] for w in plan]
    first   = next((p for p in ["Base","Build","Peak","Taper"] if p in phases), "Base")
    return _plan_phase_html(plan, first, hr_max, hr_rest)


def load_history_on_start():
    s = state_mod.load()
    hist = s.get("history", [])
    return _format_history_df(pd.DataFrame(hist)) if hist else pd.DataFrame()


def load_status_bar():
    s   = state_mod.load()
    cyc = active_cycle(s)
    parts = []
    if cyc:
        r = cyc["race"]
        parts.append(
            f"<b>{r.get('name','')}</b> · {_iso_to_dmy(r.get('date',''))} · "
            f"target {_fmt_duration(r.get('goal_time_s',0))}"
        )
        if cyc.get("plan"):
            parts.append(f"{len(cyc['plan'])} weeks planned")
    hist_n = len(s.get("history", []))
    if hist_n:
        parts.append(f"{hist_n} runs in history")
    text = " &nbsp;·&nbsp; ".join(parts) if parts else "No data yet — complete Setup to get started."
    return (
        "<div style='background:#1B2874;border-radius:8px;padding:9px 16px;"
        "color:#fff;font-size:13px;font-family:-apple-system,sans-serif'>"
        + text + "</div>"
    )


# ---------------------------------------------------------------------------
# Tab 1 — Setup handlers
# ---------------------------------------------------------------------------

def _parse_goal_time(hh, mm, ss):
    return int(hh or 0) * 3600 + int(mm or 0) * 60 + int(ss or 0)


def generate_plan_handler(
    name, hr_max, hr_rest,
    distance_sel, custom_km,
    race_name, race_date_str,
    goal_hh, goal_mm, goal_ss,
    b1_dist, b1_time_h, b1_time_m, b1_date,
    b2_dist, b2_time_h, b2_time_m, b2_date,
    run_days_labels, strength_labels, cycling_days_labels,
    sat_default_km, volume_cap, max_runs, injury,
    allow_increase,
):
    try:
        hr_max  = int(hr_max or 177)
        hr_rest = int(hr_rest or 50)
        goal_s  = _parse_goal_time(goal_hh, goal_mm, goal_ss)
        if not goal_s:
            return "⚠ Please enter a goal time.", gr.update(), gr.update()

        try:
            race_dt = date.fromisoformat(str(race_date_str).strip())
        except ValueError:
            return "⚠ Race date format must be YYYY-MM-DD.", gr.update(), gr.update()

        dist_km = (
            float(custom_km or 42.195) if distance_sel == "custom"
            else DISTANCE_KM.get(distance_sel, 42.195)
        )
        d_label = DISTANCE_LABEL.get(distance_sel, "custom")

        run_days  = sorted([WEEKDAY_MAP[d] for d in (run_days_labels or [])])
        strength  = [WEEKDAY_MAP[d] for d in (strength_labels or [])]
        cycling   = [WEEKDAY_MAP[d] for d in (cycling_days_labels or [])]

        # Saturday LRP club-run slot
        lrp_sessions = [
            {"day": 0, "km": 10.0, "type": "easy"},
            {"day": 5, "km": float(sat_default_km or 20.0), "type": "long"},
        ]

        # VDOT from benchmarks
        vdot = 40.0
        cv   = 0.0
        benchmarks = []
        for dist_m, th, tm, bdate in [
            (b1_dist, b1_time_h, b1_time_m, b1_date),
            (b2_dist, b2_time_h, b2_time_m, b2_date),
        ]:
            if dist_m and th is not None:
                t_s = int(th or 0)*3600 + int(tm or 0)*60
                if t_s > 0:
                    vdot = max(vdot, vdot_from_race(float(dist_m), t_s))
                    benchmarks.append({"distance_m": float(dist_m), "time_s": t_s,
                                       "date": str(bdate or "")})

        zones = build_zones(vdot, cv)

        plan = generate_plan(
            marathon_date        = race_dt,
            goal_time_s          = goal_s,
            zones                = zones,
            run_days             = run_days,
            lrp_sessions         = lrp_sessions,
            strength_days        = strength,
            cycling_days         = cycling,
            injury               = injury or "none",
            allow_volume_increase= bool(allow_increase),
        )

        cfg = {
            "volume_cap_km":     float(volume_cap or 75.0),
            "max_runs_per_week": int(max_runs or 5),
            "default_run_days":  run_days,
            "strength_days":     strength,
            "cycling_days":      cycling,
            "injury_level":      injury or "none",
            "club_runs": [
                {"id": "lrp_monday",   "day": 0, "type": "easy", "distance_km": 10.0,
                 "pinned_day": True, "pinned_distance": True, "no_target_pace": True,
                 "description": "LRP Mon from Nation — group pace"},
                {"id": "lrp_saturday", "day": 5, "type": "long",
                 "default_suggested_km": float(sat_default_km or 20.0),
                 "pinned_day": True, "pinned_distance": False,
                 "description": "LRP Sat long run from Jardin du Luxembourg"},
            ],
        }

        cyc_id = f"{d_label}-{race_dt.isoformat()}"
        s = state_mod.load()

        cyc = new_cycle(
            name          = race_name or f"{d_label.title()} {race_dt.year}",
            race_date     = race_dt.isoformat(),
            distance_km   = dist_km,
            distance_label= d_label,
            goal_time_s   = goal_s,
            config        = cfg,
        )
        cyc["id"]         = cyc_id
        cyc["athlete"]    = {"name": name or "", "hr_max": hr_max, "hr_rest": hr_rest}
        cyc["zones"]      = zones.__dict__
        cyc["benchmarks"] = benchmarks
        cyc["plan"]       = [
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
            for w in plan
        ]

        # Rebuild state_model from history
        hist = s.get("history", [])
        sm   = sm_from_dict({})
        sm   = apply_history(sm, hist)
        cyc["state_model"] = sm_to_dict(sm)

        s["athlete"] = cyc["athlete"]
        for existing in s.get("cycles", []):
            if existing.get("status") == "active":
                existing["status"] = "archived"
        s = set_active_cycle(s, cyc)
        state_mod.save(s)

        msg = (f"✓ Plan saved — {len(plan)} weeks · VDOT {zones.vdot} · "
               f"M-pace {_fmt_pace(zones.marathon)} · "
               f"T-pace {_fmt_pace(zones.threshold)}")
        phases = [w["phase"] for w in cyc["plan"]]
        first  = next((p for p in ["Base","Build","Peak","Taper"] if p in phases), "Base")
        return (
            f"<div style='background:#DCFCE7;border:1px solid #16A34A;"
            f"border-radius:8px;padding:10px 14px;font-size:13px;color:#166534'>{msg}</div>",
            gr.update(value=first),
            _plan_phase_html(cyc["plan"], first, hr_max, hr_rest),
        )
    except Exception as e:
        return (
            f"<div style='background:#FEE2E2;border:1px solid #DC2626;"
            f"border-radius:8px;padding:10px 14px;font-size:13px;color:#7F1D1D'>"
            f"Error: {e}</div>",
            gr.update(), gr.update(),
        )


def render_plan_phase(phase):
    s   = state_mod.load()
    cyc = active_cycle(s)
    plan = cyc.get("plan", []) if cyc else []
    hr_max  = s.get("athlete", {}).get("hr_max", 177)
    hr_rest = s.get("athlete", {}).get("hr_rest", 50)
    return _plan_phase_html(plan, phase, hr_max, hr_rest)


# ---------------------------------------------------------------------------
# Tab 3 — Weekly check-in handlers
# ---------------------------------------------------------------------------

def _parse_availability(
    available_days_labels, unavail_days_labels,
    mon_decision, sat_decision, sat_sub_day,
):
    if not any([available_days_labels, unavail_days_labels,
                mon_decision != "keep", sat_decision != "keep"]):
        return None
    avail   = [WEEKDAY_MAP[d] for d in (available_days_labels or [])]
    unavail = [WEEKDAY_MAP[d] for d in (unavail_days_labels or [])]
    club    = {}
    if mon_decision != "keep":
        club["lrp_monday"] = (
            {"type": "substitute_to_day", "day": WEEKDAY_MAP.get(sat_sub_day or "Tue", 1)}
            if mon_decision == "substitute" else mon_decision
        )
    if sat_decision != "keep":
        club["lrp_saturday"] = (
            {"type": "substitute_to_day", "day": WEEKDAY_MAP.get(sat_sub_day or "Fri", 4)}
            if sat_decision == "substitute" else sat_decision
        )
    return {
        "available_days":     avail,
        "unavailable_days":   unavail,
        "club_run_decisions": club,
    }


def process_checkin(
    fit_files, feeling,
    saturday_km_val,
    avail_days, unavail_days,
    mon_decision, sat_decision, sat_sub_day,
):
    s   = state_mod.load()
    cyc = active_cycle(s)
    if not cyc:
        return ("⚠ No active training cycle — complete Setup first.",
                "", "", gr.update())

    hr_max  = s.get("athlete", {}).get("hr_max", 177)
    hr_rest = s.get("athlete", {}).get("hr_rest", 50)
    hist    = s.get("history", [])

    # --- Parse FIT files ---
    new_runs = []
    if fit_files:
        for f in fit_files:
            path = f.name if hasattr(f, "name") else str(f)
            rec  = fit_summarize(path)
            if rec and "error" not in rec:
                zd  = cyc.get("zones", {})
                z = Zones(**{k: v for k, v in zd.items() if k in Zones.__dataclass_fields__}) if zd else None
                rec["training_load"] = compute_load(rec, hr_max, hr_rest, z)
                new_runs.append(rec)

    # Merge new runs into history (deduplicate by date)
    existing_dates = {r["date"] for r in hist}
    added = [r for r in new_runs if r["date"] not in existing_dates]
    hist.extend(added)
    hist.sort(key=lambda r: r.get("date", ""), reverse=True)

    # --- Update state model ---
    sm = sm_from_dict(cyc.get("state_model", {}))
    sm = apply_history(sm, added)
    cyc["state_model"] = sm_to_dict(sm)

    # --- Build availability override ---
    availability = _parse_availability(
        avail_days, unavail_days, mon_decision, sat_decision, sat_sub_day
    )

    sat_km = float(saturday_km_val) if saturday_km_val else None
    proposal = propose_week(cyc, hist, feeling=float(feeling or 3.0),
                            saturday_km=sat_km, availability=availability)

    # --- Save check-in to history log ---
    week_iso = proposal.week_iso
    cyc.setdefault("weekly_overrides", {})[week_iso] = availability or {}
    cyc.setdefault("check_in_history", []).append({
        "week_iso":        week_iso,
        "feeling":         float(feeling or 3.0),
        "ladder_score":    proposal.ladder_score,
        "ladder_decision": proposal.ladder_decision,
        "load_target":     proposal.load_target,
        "load_target_km":  proposal.target_km,
        "form_end":        sm_from_dict(cyc["state_model"]).form,
    })

    s = set_active_cycle(s, cyc)
    s["history"] = hist
    state_mod.save(s)

    # --- Metrics summary ---
    r_label = readiness_label(proposal.readiness)
    metrics_html = (
        f"<div style='display:flex;gap:12px;flex-wrap:wrap;font-family:-apple-system,sans-serif'>"
        + "".join(
            f"<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;"
            f"padding:10px 14px;min-width:110px'>"
            f"<div style='font-size:11px;color:#6B7280;text-transform:uppercase;"
            f"letter-spacing:.05em'>{label}</div>"
            f"<div style='font-size:18px;font-weight:700;color:#1B2874;margin-top:2px'>{val}</div>"
            f"</div>"
            for label, val in [
                ("Readiness",  f"{r_label}"),
                ("Target km",  f"{proposal.target_km:.0f}"),
                ("Sat long",   f"{proposal.saturday_km:.0f} km"),
                ("Load",       f"{proposal.load_target:.0f} TRIMP"),
                ("ACWR",       f"{proposal.acwr:.2f}"),
                (proposal.ladder_decision, f"{proposal.ladder_score}/100"),
            ]
        )
        + "</div>"
    )

    warnings_html = ""
    if proposal.warnings:
        w_items = "".join(f"<li>{w}</li>" for w in proposal.warnings)
        warnings_html = (
            f"<div style='background:#FEF3C7;border:1px solid #F59E0B;"
            f"border-radius:8px;padding:10px 14px;font-size:13px;color:#92400E;"
            f"margin-top:8px'><b>⚠ Warnings</b><ul style='margin:4px 0 0 16px'>"
            f"{w_items}</ul></div>"
        )

    # --- LLM coaching note ---
    ctx_data   = proposal.coaching_context
    ladder     = ctx_data.get("ladder")
    met        = ctx_data.get("metrics")
    zd         = cyc.get("zones", {})
    z = Zones(**{k: v for k, v in zd.items() if k in Zones.__dataclass_fields__}) if zd else build_zones(40.0,0.0)
    days_left  = ctx_data.get("days_left", 84)
    wks_left   = days_left // 7
    next_focus = proposal.focus
    context    = build_context(
        name       = s.get("athlete", {}).get("name", "Athlete"),
        goal_race  = cyc["race"].get("name", ""),
        goal_time  = _fmt_duration(cyc["race"].get("goal_time_s", 0)),
        weeks_left = wks_left,
        phase      = proposal.phase,
        zones_dict = zones_summary(z),
        metrics    = met,
        result     = ladder,
        next_focus = next_focus,
    ) if met and ladder else ""

    note = coaching_note(context) if context else (
        f"Readiness: {r_label} · {proposal.ladder_decision} · "
        f"Target {proposal.target_km:.0f} km next week."
    )

    # --- Next-week preview ---
    phase_rows = []
    for dp in proposal.sessions:
        sess = dp.session
        phase_rows.append({
            "week_num": 99, "phase": proposal.phase,
            "focus": proposal.focus, "target_km": proposal.target_km,
            "days": [{"date": str(dp.date), "weekday": dp.weekday,
                      "session_type": sess.type, "description": sess.description,
                      "distance_km": sess.distance_km, "targets": sess.targets}
                     for dp in proposal.sessions],
        })
    next_html = _plan_phase_html([{
        "week_num": 1, "phase": proposal.phase, "focus": proposal.focus,
        "target_km": proposal.target_km,
        "days": [{"date": str(dp.date), "weekday": dp.weekday,
                  "session_type": dp.session.type, "description": dp.session.description,
                  "distance_km": dp.session.distance_km, "targets": dp.session.targets}
                 for dp in proposal.sessions],
    }], proposal.phase, hr_max, hr_rest)

    status = (
        f"<div style='background:#DCFCE7;border:1px solid #16A34A;border-radius:8px;"
        f"padding:10px 14px;font-size:13px;color:#166534'>"
        f"✓ Check-in processed · {len(added)} new run(s) · "
        f"{proposal.ladder_decision} · readiness {r_label}</div>"
        + warnings_html
    )

    return status, metrics_html, note, gr.update(value=load_history_on_start())


# ---------------------------------------------------------------------------
# Tab 4 — Adjustments handlers
# ---------------------------------------------------------------------------

def apply_adjustment(from_wk, num_wks, no_club, easy_only, vol_pct_str, user_msg):
    s   = state_mod.load()
    cyc = active_cycle(s)
    if not cyc or not cyc.get("plan"):
        return "⚠ No plan to adjust."
    from coach.adjustments import apply as adj_apply, build_adjustment_context
    zd   = cyc.get("zones", {})
    plan = cyc["plan"]
    vol  = 1.0
    try:
        vol = float(str(vol_pct_str).replace("%","").strip()) / 100
    except Exception:
        pass
    plan, log = adj_apply(plan, int(from_wk or 1), int(num_wks or 0),
                          bool(no_club), bool(easy_only), vol, zd)
    cyc["plan"] = plan
    s = set_active_cycle(s, cyc)
    state_mod.save(s)
    r = cyc["race"]
    days_left = max(0, (date.fromisoformat(r["date"]) - date.today()).days)
    ctx = build_adjustment_context(
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
    note = coaching_note(ctx) if ctx else "\n".join(log) or "No changes applied."
    return note


# ---------------------------------------------------------------------------
# Tab 5 — Garmin / history handlers
# ---------------------------------------------------------------------------

def garmin_connect_handler(email, password):
    if not email or not password:
        return "⚠ Enter email and password."
    ok, msg = connect(email, password)
    if msg == "MFA_REQUIRED":
        return "MFA required — enter the code sent to you and click Submit MFA."
    return ("✓ " if ok else "✗ ") + msg


def garmin_mfa_handler(email, password, mfa_code):
    ok, msg = submit_mfa(email, password, mfa_code)
    return ("✓ " if ok else "✗ ") + msg


def garmin_sync_handler(days_back):
    records, msg = sync_activities(int(days_back or 30))
    if not records:
        return msg, gr.update()
    s    = state_mod.load()
    cyc  = active_cycle(s)
    hist = s.get("history", [])
    existing = {r["date"] for r in hist}
    hr_max  = s.get("athlete", {}).get("hr_max", 177)
    hr_rest = s.get("athlete", {}).get("hr_rest", 50)
    z = None
    if cyc:
        zd = cyc.get("zones", {})
        if zd:
            z = Zones(**{k: v for k, v in zd.items() if k in Zones.__dataclass_fields__})
    added = 0
    for r in records:
        if r.get("date") not in existing:
            r["training_load"] = compute_load(r, hr_max, hr_rest, z)
            hist.append(r)
            added += 1
    hist.sort(key=lambda r: r.get("date", ""), reverse=True)
    s["history"] = hist
    state_mod.save(s)
    df  = _format_history_df(pd.DataFrame(hist))
    return f"✓ {msg} · {added} new records", gr.update(value=df)


def garmin_clear_handler():
    clear_auth()
    return "Auth cleared — enter credentials to reconnect."


def garmin_status_html():
    status = connection_status()
    if status == "not_connected":
        return "<span style='color:#DC2626'>● Not connected</span>"
    email = status.replace("connected:", "") if ":" in status else ""
    return f"<span style='color:#16A34A'>● Connected{f' as {email}' if email else ''}</span>"


# ---------------------------------------------------------------------------
# Gradio Blocks app
# ---------------------------------------------------------------------------

CSS = """
#status-bar { margin-bottom: 8px }
.block { border-radius: 8px !important }
#plan-tabs .tab-nav button { font-weight: 600 }
"""

def build_app():
    s        = _startup_state()
    cyc      = active_cycle(s)
    athlete  = s.get("athlete", {})
    run_days = []
    strength = []
    cycling  = []
    sat_km   = 20.0
    run_days_default = []
    if cyc:
        cfg = cyc.get("config", {})
        run_days_default = [WEEKDAY_LABELS[d] for d in cfg.get("default_run_days", [])]
        strength = [WEEKDAY_LABELS[d] for d in cfg.get("strength_days", [])]
        cycling  = [WEEKDAY_LABELS[d] for d in cfg.get("cycling_days", [])]
        for cr in cfg.get("club_runs", []):
            if cr.get("id") == "lrp_saturday":
                sat_km = float(cr.get("default_suggested_km") or 20.0)

    with gr.Blocks(title="LRP Coach", theme=gr.themes.Soft(), css=CSS) as demo:
        # ── Status bar ──────────────────────────────────────────────────────
        status_bar = gr.HTML(load_status_bar, elem_id="status-bar")

        with gr.Tabs():

            # ── Tab 1: Setup ─────────────────────────────────────────────
            with gr.Tab("Setup & Plan"):
                gr.Markdown("### Athlete profile")
                with gr.Row():
                    name_in    = gr.Textbox(label="Name",
                                            value=athlete.get("name", ""))
                    hr_max_in  = gr.Number(label="HRmax (bpm)", precision=0,
                                           value=athlete.get("hr_max", 177))
                    hr_rest_in = gr.Number(label="Resting HR (bpm)", precision=0,
                                           value=athlete.get("hr_rest", 50))

                gr.Markdown("### Race goal")
                with gr.Row():
                    race_name_in = gr.Textbox(
                        label="Race name",
                        value=cyc["race"].get("name","") if cyc else "")
                    race_date_in = gr.Textbox(
                        label="Race date (YYYY-MM-DD)",
                        value=cyc["race"].get("date","") if cyc else "",
                        placeholder="2026-11-09")
                with gr.Row():
                    dist_sel_in = gr.Dropdown(DISTANCE_OPTS, label="Distance",
                                              value=DISTANCE_OPTS[0])
                    custom_km_in = gr.Number(label="Custom distance (km)",
                                             value=42.195, visible=False)
                with gr.Row():
                    goal_hh_in = gr.Number(label="Goal H", precision=0, value=3, minimum=0)
                    goal_mm_in = gr.Number(label="Goal M", precision=0, value=45, minimum=0, maximum=59)
                    goal_ss_in = gr.Number(label="Goal S", precision=0, value=0,  minimum=0, maximum=59)

                dist_sel_in.change(
                    lambda v: gr.update(visible=(v == "custom")),
                    inputs=dist_sel_in, outputs=custom_km_in)

                gr.Markdown("### Benchmarks (most recent races)")
                with gr.Row():
                    b1_dist_in = gr.Number(label="Bench 1 — distance (m)", value=10000)
                    b1_th_in   = gr.Number(label="Time H", precision=0, value=0)
                    b1_tm_in   = gr.Number(label="Time M", precision=0, value=52)
                    b1_date_in = gr.Textbox(label="Date (YYYY-MM-DD)", placeholder="2026-04-13")
                with gr.Row():
                    b2_dist_in = gr.Number(label="Bench 2 — distance (m)", value=None)
                    b2_th_in   = gr.Number(label="Time H", precision=0, value=None)
                    b2_tm_in   = gr.Number(label="Time M", precision=0, value=None)
                    b2_date_in = gr.Textbox(label="Date (YYYY-MM-DD)")

                gr.Markdown("### Weekly schedule")
                run_days_in  = gr.CheckboxGroup(WEEKDAY_LABELS, label="Run days",
                                                value=run_days_default)
                strength_in  = gr.CheckboxGroup(WEEKDAY_LABELS, label="Strength days",
                                                value=strength)
                cycling_in   = gr.CheckboxGroup(WEEKDAY_LABELS, label="Cycling / Zwift days",
                                                value=cycling)

                gr.Markdown("### LRP club runs")
                gr.HTML("<div style='background:#EFF6FF;border:1px solid #BFDBFE;"
                        "border-radius:8px;padding:10px 14px;font-size:13px'>"
                        "<b>Monday:</b> fixed 10 km easy from Nation — group pace, never moved by the optimizer.<br>"
                        "<b>Saturday:</b> long run from Jardin du Luxembourg — you set the distance each week at check-in.</div>")
                sat_default_in = gr.Number(label="Saturday default suggested distance (km)",
                                           value=sat_km, precision=1)

                gr.Markdown("### Constraints")
                with gr.Row():
                    vol_cap_in = gr.Number(label="Weekly volume cap (km)",
                                           value=75.0, precision=1)
                    max_runs_in = gr.Number(label="Max runs per week",
                                            value=5, precision=0)
                    injury_in   = gr.Dropdown(["none", "light", "moderate"],
                                              label="Injury status", value="none")
                    allow_inc_in = gr.Checkbox(label="Allow progressive volume increase",
                                               value=True)

                gen_btn  = gr.Button("Generate Plan", variant="primary", size="lg")
                setup_msg = gr.HTML()

                gr.Markdown("---")
                gr.Markdown("### Training plan")
                plan_phase_radio = gr.Radio(
                    ["Base","Build","Peak","Taper"], value="Base",
                    label="Phase", container=False)
                plan_html = gr.HTML(load_plan_on_start)

                gen_btn.click(
                    generate_plan_handler,
                    inputs=[
                        name_in, hr_max_in, hr_rest_in,
                        dist_sel_in, custom_km_in,
                        race_name_in, race_date_in,
                        goal_hh_in, goal_mm_in, goal_ss_in,
                        b1_dist_in, b1_th_in, b1_tm_in, b1_date_in,
                        b2_dist_in, b2_th_in, b2_tm_in, b2_date_in,
                        run_days_in, strength_in, cycling_in,
                        sat_default_in, vol_cap_in, max_runs_in, injury_in,
                        allow_inc_in,
                    ],
                    outputs=[setup_msg, plan_phase_radio, plan_html],
                )
                plan_phase_radio.change(render_plan_phase,
                                        inputs=plan_phase_radio,
                                        outputs=plan_html)

            # ── Tab 2: Weekly check-in ────────────────────────────────────
            with gr.Tab("Weekly Check-in"):
                gr.Markdown("### This week's check-in")
                with gr.Row():
                    sat_km_in = gr.Number(
                        label="Saturday long-run distance (km)",
                        value=None, precision=1,
                        info="Leave blank to use the suggested distance from the plan")
                    feeling_in = gr.Slider(1, 5, step=0.5, value=3.0,
                                           label="Feeling (1 = exhausted · 5 = great)")

                gr.Markdown("### Availability this week *(leave blank = default schedule)*")
                with gr.Row():
                    avail_in   = gr.CheckboxGroup(WEEKDAY_LABELS, label="Days available")
                    unavail_in = gr.CheckboxGroup(WEEKDAY_LABELS, label="Days unavailable")
                with gr.Row():
                    mon_dec_in = gr.Dropdown(
                        ["keep", "skip", "substitute"], value="keep",
                        label="Monday LRP decision")
                    sat_dec_in = gr.Dropdown(
                        ["keep", "skip", "substitute"], value="keep",
                        label="Saturday LRP decision")
                    sub_day_in = gr.Dropdown(WEEKDAY_LABELS, label="Substitute to day",
                                             value="Fri")

                gr.Markdown("### Upload FIT files")
                fit_files_in = gr.File(label="Drag FIT files here (multi-file)",
                                       file_count="multiple", file_types=[".fit"])

                checkin_btn  = gr.Button("Process check-in", variant="primary", size="lg")
                checkin_msg  = gr.HTML()
                metrics_html = gr.HTML()
                note_out     = gr.Textbox(label="Coaching note", lines=8, interactive=False)
                next_week_html = gr.HTML(label="Next week proposal")
                hist_df_state  = gr.State(value=None)

                checkin_btn.click(
                    process_checkin,
                    inputs=[
                        fit_files_in, feeling_in,
                        sat_km_in, avail_in, unavail_in,
                        mon_dec_in, sat_dec_in, sub_day_in,
                    ],
                    outputs=[checkin_msg, metrics_html, note_out, hist_df_state],
                ).then(lambda: load_status_bar(), outputs=status_bar)

            # ── Tab 3: Adjustments ────────────────────────────────────────
            with gr.Tab("Adjustments"):
                gr.Markdown(
                    "Apply physio / travel / illness overrides to future plan weeks. "
                    "The pain gate overrides the optimizer completely."
                )
                with gr.Row():
                    adj_from_in  = gr.Number(label="From week #", value=1, precision=0)
                    adj_num_in   = gr.Number(label="Number of weeks (0 = all)", value=0, precision=0)
                with gr.Row():
                    no_club_in   = gr.Checkbox(label="Replace LRP club runs with easy solo runs")
                    easy_only_in = gr.Checkbox(label="Quality sessions → easy runs (injury / illness)")
                    vol_pct_in   = gr.Textbox(label="Volume adjustment (%)", value="100",
                                              placeholder="80 for −20 %")
                msg_in   = gr.Textbox(label="Your message to the coach (optional)", lines=3)
                adj_btn  = gr.Button("Apply & get coaching note", variant="primary")
                adj_note = gr.Textbox(label="Coaching note", lines=8, interactive=False)

                adj_btn.click(
                    apply_adjustment,
                    inputs=[adj_from_in, adj_num_in, no_club_in,
                             easy_only_in, vol_pct_in, msg_in],
                    outputs=adj_note,
                )

            # ── Tab 4: History & Garmin ───────────────────────────────────
            with gr.Tab("History"):
                hist_df_state2 = gr.State(value=None)
                hist_page_st   = gr.State(value=0)

                gr.Markdown("### Activity log")
                hist_html = gr.HTML()
                with gr.Row():
                    hist_prev_btn = gr.Button("← Prev", size="sm")
                    with gr.Column(scale=2):
                        hist_page_info = gr.HTML()
                    hist_next_btn = gr.Button("Next →", size="sm")

                def _hist_reset(df):
                    html, pg, info = _hist_page_html(df, 0)
                    return html, 0, info
                def _hist_prev(df, pg):
                    html, pg, info = _hist_page_html(df, int(pg or 0) - 1)
                    return html, pg, info
                def _hist_next(df, pg):
                    html, pg, info = _hist_page_html(df, int(pg or 0) + 1)
                    return html, pg, info

                demo.load(
                    load_history_on_start,
                    outputs=hist_df_state2,
                ).then(_hist_reset,
                       inputs=hist_df_state2,
                       outputs=[hist_html, hist_page_st, hist_page_info])

                hist_prev_btn.click(_hist_prev,
                                    inputs=[hist_df_state2, hist_page_st],
                                    outputs=[hist_html, hist_page_st, hist_page_info])
                hist_next_btn.click(_hist_next,
                                    inputs=[hist_df_state2, hist_page_st],
                                    outputs=[hist_html, hist_page_st, hist_page_info])

                gr.Markdown("---")
                gr.Markdown("### Garmin Connect sync")
                with gr.Row():
                    garmin_status_html_out = gr.HTML(garmin_status_html)
                    days_back_in = gr.Slider(7, 90, step=7, value=30,
                                             label="Days to sync", scale=2)
                    sync_btn     = gr.Button("Sync from Garmin", size="sm", scale=1)
                sync_msg = gr.HTML()

                gr.Markdown("#### First-time login")
                with gr.Row():
                    g_email_in = gr.Textbox(label="Garmin email",
                                            value=load_email() or "", scale=2)
                    g_pass_in  = gr.Textbox(label="Password", type="password", scale=2)
                    g_login_btn= gr.Button("Connect", size="sm", scale=1)
                g_login_msg = gr.HTML()

                with gr.Row():
                    g_mfa_in  = gr.Textbox(label="MFA code (if requested)", scale=2)
                    g_mfa_btn = gr.Button("Submit MFA", size="sm", scale=1)

                g_clear_btn = gr.Button("Clear saved auth", size="sm", variant="secondary")

                sync_btn.click(
                    garmin_sync_handler,
                    inputs=days_back_in,
                    outputs=[sync_msg, hist_df_state2],
                ).then(_hist_reset,
                       inputs=hist_df_state2,
                       outputs=[hist_html, hist_page_st, hist_page_info])

                g_login_btn.click(
                    garmin_connect_handler,
                    inputs=[g_email_in, g_pass_in],
                    outputs=g_login_msg,
                )
                g_mfa_btn.click(
                    garmin_mfa_handler,
                    inputs=[g_email_in, g_pass_in, g_mfa_in],
                    outputs=g_login_msg,
                )
                g_clear_btn.click(garmin_clear_handler, outputs=g_login_msg)

    return demo


if __name__ == "__main__":
    _check_gradio_client_patch()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
