"""FIT file parser — single file summary and multi-file aggregation."""

from __future__ import annotations

import fitdecode
import pandas as pd


def _parse_records(path: str) -> pd.DataFrame:
    rows = []
    with fitdecode.FitReader(path) as fit:
        for frame in fit:
            if frame.frame_type == fitdecode.FIT_FRAME_DATA and frame.name == "record":
                def g(name):
                    return frame.get_value(name) if frame.has_field(name) else None
                rows.append({
                    "timestamp":  g("timestamp"),
                    "distance":   g("distance"),
                    "heart_rate": g("heart_rate"),
                    "speed":      g("enhanced_speed") or g("speed"),
                    "altitude":   g("enhanced_altitude") or g("altitude"),
                    "cadence":    g("cadence"),
                })
    df = pd.DataFrame(rows)
    return df.dropna(subset=["timestamp"]).reset_index(drop=True)


def summarize(path: str) -> dict | None:
    """Extract key metrics from a single FIT file. Returns None on failure."""
    try:
        df = _parse_records(path)
        if df.empty:
            return None

        duration_s = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds()
        dist_km = (
            df["distance"].dropna().iloc[-1] / 1000
            if df["distance"].notna().any() else 0.0
        )
        avg_pace_s = (duration_s / dist_km) if dist_km > 0 else None  # sec/km

        hr = df["heart_rate"].dropna()
        avg_hr = float(hr.mean()) if not hr.empty else None
        max_hr = float(hr.max()) if not hr.empty else None

        hr_drift = None
        if len(hr) > 10:
            half = len(df) // 2
            first  = df["heart_rate"].iloc[:half].dropna().mean()
            second = df["heart_rate"].iloc[half:].dropna().mean()
            if first and first > 0:
                hr_drift = round((second - first) / first * 100, 1)

        # Cadence (steps per minute): FIT stores single-foot cadence; double it
        cad = df["cadence"].dropna()
        avg_cadence = round(float(cad.mean()) * 2) if not cad.empty else None

        # Elevation gain: sum of positive altitude deltas
        alt = df["altitude"].dropna()
        elevation_gain_m = None
        if len(alt) > 1:
            diffs = alt.diff().dropna()
            elevation_gain_m = round(float(diffs[diffs > 0].sum()), 0)

        return {
            "date":               str(df["timestamp"].iloc[0].date()),
            "distance_km":        round(dist_km, 2),
            "duration_s":         int(duration_s),
            "avg_pace_s":         round(avg_pace_s) if avg_pace_s else None,
            "avg_hr":             round(avg_hr) if avg_hr else None,
            "max_hr":             round(max_hr) if max_hr else None,
            "hr_drift_pct":       hr_drift,
            "avg_cadence_spm":    avg_cadence,
            "elevation_gain_m":   int(elevation_gain_m) if elevation_gain_m is not None else None,
        }
    except Exception as exc:
        return {"error": str(exc), "date": "unknown", "distance_km": 0}


def summarize_many(paths: list) -> pd.DataFrame:
    """Process a list of FIT file paths into a sorted DataFrame."""
    records = [r for p in paths if (r := summarize(p)) and "error" not in r]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).sort_values("date", ascending=False).reset_index(drop=True)
