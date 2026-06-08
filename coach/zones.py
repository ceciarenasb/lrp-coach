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
