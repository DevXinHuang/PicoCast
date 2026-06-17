"""Candidate association across scans."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from nexrad_picoballoon.wind import constant_wind_prior, wind_consistency_score


def build_tracks(candidates: pd.DataFrame) -> pd.DataFrame:
    """Build simple site-level tracks from time-ordered candidate detections."""

    if candidates.empty:
        return pd.DataFrame(
            columns=[
                "track_id",
                "site",
                "detection_ids",
                "start_time",
                "end_time",
                "mean_speed_ms",
                "continuity_score",
                "wind_consistency_score",
            ]
        )

    parsed = candidates.copy()
    parsed["scan_time_dt"] = pd.to_datetime(parsed["scan_time"], utc=True, errors="coerce")
    rows = []
    for site, group in parsed.sort_values("scan_time_dt").groupby("site", dropna=False):
        detection_ids = tuple(group["detection_id"].astype(str))
        start = group["scan_time_dt"].iloc[0]
        end = group["scan_time_dt"].iloc[-1]
        duration_s = max((end - start).total_seconds(), 1.0)
        lat_span = float(group["latitude"].iloc[-1] - group["latitude"].iloc[0])
        lon_span = float(group["longitude"].iloc[-1] - group["longitude"].iloc[0])
        meters_per_deg = 111_000.0
        observed_u = lon_span * meters_per_deg / duration_s
        observed_v = lat_span * meters_per_deg / duration_s
        speed = (observed_u**2 + observed_v**2) ** 0.5
        continuity = min(1.0, len(group) / 3.0)
        prior = constant_wind_prior()
        rows.append(
            {
                "track_id": f"{site}_track_000",
                "site": site,
                "detection_ids": ",".join(detection_ids),
                "start_time": _iso(start),
                "end_time": _iso(end),
                "mean_speed_ms": speed,
                "continuity_score": continuity,
                "wind_consistency_score": wind_consistency_score(observed_u, observed_v, prior),
            }
        )
    return pd.DataFrame(rows)


def _iso(value: pd.Timestamp | datetime) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value.isoformat()


def write_tracks(tracks: pd.DataFrame, out_path: Path) -> Path:
    """Write associated tracks as Parquet."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out_path, index=False)
    return out_path
