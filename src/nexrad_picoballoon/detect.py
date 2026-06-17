"""Interpretable candidate detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def score_gates(features: pd.DataFrame) -> pd.DataFrame:
    """Add rejection flags and an interpretable candidate score per gate."""

    scored = features.copy()
    weak_echo = scored["reflectivity_dbz"].between(-5.0, 25.0)
    moving = scored["velocity_ms"].abs() >= 1.0
    compact_width = scored["spectrum_width_ms"].between(0.1, 3.5)
    nonmet_hint = scored["rhohv"].between(0.45, 0.92)
    texture_hint = (
        scored["rhohv_texture"].fillna(0.0).ge(0.02)
        | scored["zdr_texture"].fillna(0.0).ge(0.2)
        | scored["phidp_texture"].fillna(0.0).ge(1.0)
    )
    near_radar_clutter = scored["range_m"].lt(10_000.0)

    score = (
        weak_echo.astype(float) * 0.25
        + moving.astype(float) * 0.25
        + compact_width.astype(float) * 0.15
        + nonmet_hint.astype(float) * 0.2
        + texture_hint.astype(float) * 0.15
        - near_radar_clutter.astype(float) * 0.25
    )
    scored["candidate_score"] = np.clip(score, 0.0, 1.0)
    scored["rejection_flags"] = np.where(near_radar_clutter, "near_radar_clutter", "")
    return scored


def detect_candidates(features: pd.DataFrame, min_score: float = 0.65) -> pd.DataFrame:
    """Collapse scored gates into per-scan candidate objects.

    The MVP groups all qualifying gates by site and scan time. A follow-on
    connected-component pass can split multiple objects in the same scan.
    """

    scored = score_gates(features)
    selected = scored[scored["candidate_score"] >= min_score]
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "detection_id",
                "site",
                "scan_time",
                "latitude",
                "longitude",
                "altitude_m",
                "score",
                "gate_count",
                "mean_reflectivity_dbz",
                "mean_velocity_ms",
                "rejection_flags",
            ]
        )

    rows = []
    for (site, scan_time), group in selected.groupby(["site", "scan_time"], dropna=False):
        detection_id = f"{site}_{str(scan_time).replace(':', '').replace('-', '')}_000"
        flags = tuple(sorted(flag for flag in group["rejection_flags"].unique() if flag))
        rows.append(
            {
                "detection_id": detection_id,
                "site": site,
                "scan_time": scan_time,
                "latitude": float(group["latitude"].mean()),
                "longitude": float(group["longitude"].mean()),
                "altitude_m": float(group["altitude_m"].mean()),
                "score": float(group["candidate_score"].mean()),
                "gate_count": int(len(group)),
                "mean_reflectivity_dbz": float(group["reflectivity_dbz"].mean()),
                "mean_velocity_ms": float(group["velocity_ms"].mean()),
                "rejection_flags": ",".join(flags),
            }
        )
    return pd.DataFrame(rows)


def write_candidates(candidates: pd.DataFrame, out_path: Path) -> Path:
    """Write candidate detections as Parquet."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(out_path, index=False)
    return out_path
