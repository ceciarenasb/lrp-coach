"""
JSON persistence for the athlete's state.

Schema (v2): multi-race, config-driven.
  {
    "athlete": { name, hr_max, hr_rest },
    "cycles":  [ { id, status, race, config, benchmarks, zones, plan,
                   state_model, weekly_overrides, check_in_history } ],
    "history": [ run records with training_load ]
  }

Migration: first load() of an old flat-schema file wraps it automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_PATH = Path(__file__).parent.parent / "data" / "state.json"


# ---------------------------------------------------------------------------
# Core I/O (unchanged API)
# ---------------------------------------------------------------------------

def load() -> dict:
    if not _PATH.exists():
        return {}
    with open(_PATH) as f:
        raw = json.load(f)
    if "cycles" not in raw:
        raw = _migrate_v1(raw)
        _write(raw)
    return raw


def save(data: dict) -> None:
    _PATH.parent.mkdir(exist_ok=True)
    if _PATH.exists():
        _PATH.replace(_PATH.with_suffix(".backup.json"))
    _write(data)


def update(key: str, value) -> None:
    s = load()
    s[key] = value
    save(s)


def _write(data: dict) -> None:
    with open(_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Cycle helpers
# ---------------------------------------------------------------------------

def active_cycle(state: dict) -> Optional[dict]:
    """Return the first cycle with status='active', or None."""
    for c in state.get("cycles", []):
        if c.get("status") == "active":
            return c
    return None


def set_active_cycle(state: dict, cycle: dict) -> dict:
    """Replace the active cycle in state (by id) and return updated state."""
    cycles = state.get("cycles", [])
    for i, c in enumerate(cycles):
        if c.get("id") == cycle.get("id"):
            cycles[i] = cycle
            state["cycles"] = cycles
            return state
    # Not found — append
    state.setdefault("cycles", []).append(cycle)
    return state


def archive_cycle(state: dict, cycle_id: str) -> dict:
    """Mark a cycle as archived."""
    from datetime import date
    for c in state.get("cycles", []):
        if c.get("id") == cycle_id:
            c["status"]      = "archived"
            c["archived_at"] = date.today().isoformat()
    return state


def new_cycle(
    name: str,
    race_date: str,
    distance_km: float,
    distance_label: str,
    goal_time_s: int,
    config: dict,
) -> dict:
    """Create a fresh cycle dict with sensible defaults."""
    import re
    cid = re.sub(r"[^a-z0-9-]", "-", f"{distance_label}-{race_date}".lower())
    return {
        "id":     cid,
        "status": "active",
        "race": {
            "name":           name,
            "date":           race_date,
            "distance_km":    distance_km,
            "distance_label": distance_label,
            "goal_time_s":    goal_time_s,
        },
        "config": {
            "volume_cap_km":   config.get("volume_cap_km",   75.0),
            "max_runs_per_week": config.get("max_runs_per_week", 5),
            "club_runs":       config.get("club_runs", _default_club_runs()),
            "default_run_days": config.get("default_run_days", [0, 1, 3, 5]),
            "strength_days":   config.get("strength_days", []),
            "cycling_days":    config.get("cycling_days", []),
            "injury_level":    config.get("injury_level", "none"),
            "injury_notes":    config.get("injury_notes", ""),
        },
        "benchmarks":      [],
        "zones":           {},
        "plan":            [],
        "state_model": {
            "fitness": 20.0, "fatigue": 10.0, "form": 10.0,
            "k1_days": 42.0, "k2_days": 7.0,  "last_updated": "",
        },
        "weekly_overrides":  {},
        "check_in_history":  [],
    }


def _default_club_runs() -> list:
    return [
        {
            "id":               "lrp_monday",
            "day":              0,
            "type":             "easy",
            "distance_km":      10.0,
            "pinned_day":       True,
            "pinned_distance":  True,
            "no_target_pace":   True,
            "description":      "LRP Mon from Nation — group pace",
        },
        {
            "id":                    "lrp_saturday",
            "day":                   5,
            "type":                  "long",
            "default_suggested_km":  None,
            "pinned_day":            True,
            "pinned_distance":       False,
            "description":           "LRP Sat long run from Jardin du Luxembourg",
        },
    ]


# ---------------------------------------------------------------------------
# V1 → V2 migration
# ---------------------------------------------------------------------------

def _migrate_v1(old: dict) -> dict:
    """Wrap a flat v1 state dict into the v2 multi-race schema."""
    profile  = old.get("profile", {})
    schedule = old.get("schedule", {})

    cid = "migrated-marathon"
    cycle: dict = {
        "id":     cid,
        "status": "active",
        "race": {
            "name":           profile.get("goal_race", ""),
            "date":           profile.get("marathon_date", ""),
            "distance_km":    42.195,
            "distance_label": "marathon",
            "goal_time_s":    profile.get("goal_time_s", 0),
        },
        "config": {
            "volume_cap_km":    75.0,
            "max_runs_per_week": 5,
            "club_runs":        _default_club_runs(),
            "default_run_days": schedule.get("run_days", [0, 1, 3, 5]),
            "strength_days":    schedule.get("strength_days", []),
            "cycling_days":     schedule.get("cycling_days", []),
            "injury_level":     profile.get("injury_level", "none"),
            "injury_notes":     profile.get("injury_notes", ""),
        },
        "benchmarks": [],
        "zones":      old.get("zones", {}),
        "plan":       old.get("plan", []),
        "state_model": {
            "fitness": 20.0, "fatigue": 10.0, "form": 10.0,
            "k1_days": 42.0, "k2_days": 7.0,  "last_updated": "",
        },
        "weekly_overrides": {},
        "check_in_history": [],
    }

    # Carry benchmarks from v1 profile fields
    for i in (1, 2):
        dist = profile.get(f"b{i}_dist")
        ts   = profile.get(f"b{i}_time_s")
        if dist and ts:
            cycle["benchmarks"].append({
                "distance_m": float(dist),
                "time_s":     float(ts),
                "date":       profile.get(f"b{i}_date", ""),
            })

    # Backfill training_load=None into history records that don't have it
    history = old.get("history", [])
    for r in history:
        r.setdefault("training_load", None)

    return {
        "athlete": {
            "name":    profile.get("name", ""),
            "hr_max":  old.get("hr_max", 177),
            "hr_rest": old.get("hr_rest", 50),
        },
        "cycles":  [cycle],
        "history": history,
    }
