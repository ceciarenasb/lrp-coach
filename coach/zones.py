"""
Training zone calculation: Daniels VDOT + Critical Velocity (SVC) hybrid.

Z2  Easy         67–74 % VO2max      (Daniels E)
Z3  Threshold    88 % VO2max         (Daniels T)
Z4  SVC          100 % CV            (vitesse critique — sits between T and I)
Z5  Interval     98 % VO2max         (Daniels I)
Z6  Rep          105 % CV            (Daniels R)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Zones:
    vdot: float
    cv_mps: float       # Critical Velocity, m/s

    # All paces in sec/km (lower = faster)
    easy_lo: int        # Z2 slower bound
    easy_hi: int        # Z2 faster bound
    marathon: int       # M-pace
    threshold: int      # Z3 T-pace
    cv_interval: int    # Z4 SVC pace
    interval: int       # Z5 I-pace
    rep: int            # Z6 R-pace


def _v_from_pct_vo2(vdot: float, pct: float) -> float:
    """Velocity (m/min) at pct × VO2max via Daniels' quadratic."""
    target = vdot * pct
    a, b, c = 0.000104, 0.182258, -4.60 - target
    disc = b ** 2 - 4 * a * c
    return (-b + math.sqrt(max(disc, 0.0))) / (2 * a)


def _spk(v_mpm: float) -> int:
    """m/min → sec/km."""
    return round(60_000 / v_mpm)


def vdot_from_race(distance_m: float, time_s: float) -> float:
    """Jack Daniels VDOT from a race result."""
    v = distance_m / time_s * 60          # m/min
    d = time_s / 60                        # duration, min
    pct = (0.8
           + 0.1894393 * math.exp(-0.012778 * d)
           + 0.2989558 * math.exp(-0.1932605 * d))
    vo2 = -4.60 + 0.182258 * v + 0.000104 * v ** 2
    return vo2 / pct


def cv_from_two_efforts(d1_m: float, t1_s: float, d2_m: float, t2_s: float) -> float:
    """Critical Velocity (m/s) via Monod-Billat: CV = (D2 − D1) / (T2 − T1)."""
    dt = t2_s - t1_s
    return (d2_m - d1_m) / dt if abs(dt) > 1 else 0.0


def cv_from_vdot(vdot: float) -> float:
    """Estimate CV from VDOT when only one race is available.
    CV ≈ speed at 93 % VO2max (roughly a 20–40 min maximal effort).
    """
    return _v_from_pct_vo2(vdot, 0.93) / 60  # → m/s


def build_zones(vdot: float, cv_mps: float) -> Zones:
    if cv_mps <= 0:
        cv_mps = cv_from_vdot(vdot)
    cv_mpm = cv_mps * 60
    return Zones(
        vdot=round(vdot, 1),
        cv_mps=round(cv_mps, 3),
        easy_lo=_spk(_v_from_pct_vo2(vdot, 0.67)),
        easy_hi=_spk(_v_from_pct_vo2(vdot, 0.74)),
        marathon=_spk(_v_from_pct_vo2(vdot, 0.84)),
        threshold=_spk(_v_from_pct_vo2(vdot, 0.88)),
        cv_interval=_spk(cv_mpm),
        interval=_spk(_v_from_pct_vo2(vdot, 0.98)),
        rep=_spk(cv_mpm * 1.05),
    )


def vdot_recency_factor(date_str: str) -> float:
    """
    Discount a benchmark VDOT based on how old it is.
    A 2-year-old race is less reliable for current training zones.
    """
    from datetime import date
    try:
        bench = date.fromisoformat(date_str)
        age = (date.today() - bench).days
    except Exception:
        return 1.0
    if age < 90:   return 1.00
    if age < 180:  return 0.99
    if age < 365:  return 0.97
    if age < 730:  return 0.94
    return 0.90


def infer_vdot_adjustment(current_zones: "Zones", history: list) -> "Zones | None":
    """
    Analyse recent easy runs (last 8 weeks) and return updated Zones if
    fitness has shifted by ≥1 VDOT point, else None.

    Qualifying run: HR < 155 (or no HR data), distance 5–25 km, pace within
    20 % of the current easy zone. Requires ≥4 qualifying runs.
    ~4 % sustained pace improvement ≈ 1 VDOT point (Daniels approximation).
    """
    from datetime import date, timedelta
    cutoff = (date.today() - timedelta(weeks=8)).isoformat()

    paces = []
    for r in history:
        if r.get("date", "") < cutoff:
            continue
        pace_s = r.get("avg_pace_s")
        km     = r.get("distance_km", 0)
        hr     = r.get("avg_hr")
        if not pace_s or km < 5 or km > 25:
            continue
        if hr and hr > 155:
            continue
        lo = current_zones.easy_hi * 0.85
        hi = current_zones.easy_lo * 1.10
        if lo <= pace_s <= hi:
            paces.append(pace_s)

    if len(paces) < 4:
        return None

    avg = sum(paces) / len(paces)
    zone_mid = (current_zones.easy_lo + current_zones.easy_hi) / 2
    drift = (zone_mid - avg) / zone_mid   # > 0 → running faster than zone

    if abs(drift) < 0.04:
        return None

    vdot_delta = drift / 0.04
    new_vdot = max(25.0, min(85.0, round(current_zones.vdot + vdot_delta, 1)))
    if abs(new_vdot - current_zones.vdot) < 1.0:
        return None

    new_cv = current_zones.cv_mps * (new_vdot / current_zones.vdot)
    return build_zones(new_vdot, new_cv)


def pace_zones_extended(z: "Zones") -> list:
    """All workout pace targets for display."""
    def blend(a, b, t):
        return round(a + (b - a) * t)

    pace_5k = _spk(_v_from_pct_vo2(z.vdot, 0.955))
    return [
        {"workout": "Recovery jog",     "pace": fmt_pace(round(z.easy_lo * 1.10)), "rpe": "1–2", "notes": "After races or hard blocks"},
        {"workout": "Easy  (Z2)",       "pace": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}", "rpe": "3–4", "notes": "Bulk of weekly mileage"},
        {"workout": "Long run",         "pace": fmt_pace(blend(z.easy_hi, z.easy_lo, 0.3)), "rpe": "4–5", "notes": "90 min – 3 h, start easy"},
        {"workout": "Marathon pace",    "pace": fmt_pace(z.marathon),              "rpe": "6",   "notes": "Race pace; M-pace segments"},
        {"workout": "Tempo  (Z3)",      "pace": fmt_pace(z.threshold),             "rpe": "7",   "notes": "20–40 min continuous"},
        {"workout": "Cruise intervals", "pace": fmt_pace(blend(z.threshold, z.cv_interval, 0.5)), "rpe": "7–8", "notes": "5 × 1 km, 1 min rest"},
        {"workout": "SVC  (Z4)",        "pace": fmt_pace(z.cv_interval),           "rpe": "8",   "notes": "8–10 × 1 km, 2 min rest"},
        {"workout": "5 K pace",         "pace": fmt_pace(pace_5k),                 "rpe": "8–9", "notes": ""},
        {"workout": "1600 m / I pace",  "pace": fmt_pace(z.interval),              "rpe": "9",   "notes": "4–5 reps · 3–4 min rest"},
        {"workout": "1000 m",           "pace": fmt_pace(z.interval),              "rpe": "9",   "notes": "5–6 reps · 3 min rest"},
        {"workout": "800 m",            "pace": fmt_pace(blend(z.interval, z.rep, 0.45)), "rpe": "9–10", "notes": "6–8 reps · 2–3 min rest"},
        {"workout": "400 m  (R pace)",  "pace": fmt_pace(z.rep),                   "rpe": "10",  "notes": "6–12 reps · full rest"},
        {"workout": "200 m",            "pace": fmt_pace(round(z.rep * 0.96)),      "rpe": "10",  "notes": "8–12 reps · full rest"},
    ]


def hr_zones(hr_max: int, hr_rest: int = 50) -> list:
    """6-zone Karvonen (HRR) model — mirrors Cecilia's actual coach configuration."""
    hrr = hr_max - hr_rest
    def kv(pct): return round(hr_rest + pct * hrr)
    bands = [
        (1, "Zone 1",        0.60, 0.70, "#94A3B8", "Séance à basse intensité"),
        (2, "Zone 1 IM",     0.70, 0.77, "#10B981", "Sous SV1 jusqu'à SV1"),
        (3, "Zone 2 HIM",    0.77, 0.85, "#F59E0B", "Entre les deux seuils (SV1–SV2)"),
        (4, "Zone 2 M",      0.85, 0.90, "#F97316", "Proche ou à SV2 / intensité critique"),
        (5, "Zone 3 HIT i",  0.90, 0.94, "#EF4444", "Dérive VO₂ vers VO₂max"),
        (6, "Zone 3 HIT ii", 0.94, 1.00, "#DC2626", "Travail court à VO₂max"),
    ]
    return [
        {"zone": n, "name": nm, "lo": kv(lo), "hi": kv(hi), "color": c, "desc": d}
        for n, nm, lo, hi, c, d in bands
    ]


def marathon_time_from_vdot(vdot: float) -> int:
    """Marathon finish time (seconds) for a given VDOT — binary search."""
    lo, hi = 5400, 21600  # 1:30 → 6:00
    for _ in range(40):
        mid = (lo + hi) // 2
        if vdot_from_race(42195, mid) > vdot:
            lo = mid
        else:
            hi = mid
    return (lo + hi) // 2


def fmt_pace(sec_per_km: int | float | None) -> str:
    if not sec_per_km or sec_per_km <= 0:
        return "n/a"
    m, s = divmod(int(sec_per_km), 60)
    return f"{m}:{s:02d} /km"


def zones_summary(z: Zones) -> dict:
    return {
        "VDOT": z.vdot,
        "Critical Velocity (SVC)": f"{z.cv_mps * 3.6:.1f} km/h  ·  {fmt_pace(z.cv_interval)}",
        "Z2 Easy": f"{fmt_pace(z.easy_lo)} – {fmt_pace(z.easy_hi)}",
        "Marathon pace (M)": fmt_pace(z.marathon),
        "Z3 Threshold / Tempo (T)": fmt_pace(z.threshold),
        "Z4 SVC Intervals": fmt_pace(z.cv_interval),
        "Z5 Intervals (I)": fmt_pace(z.interval),
        "Z6 Reps (R)": fmt_pace(z.rep),
    }
