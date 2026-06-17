"""Shared records for the batch research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RadarVolume:
    """Metadata for one decoded radar volume or sweep collection."""

    site: str
    scan_time: datetime
    source_path: Path
    sweeps: int
    fields: tuple[str, ...]


@dataclass(frozen=True)
class CandidateDetection:
    """A compact candidate target produced by per-gate scoring."""

    detection_id: str
    site: str
    scan_time: datetime
    latitude: float
    longitude: float
    altitude_m: float
    score: float
    gate_count: int
    mean_reflectivity_dbz: float
    mean_velocity_ms: float
    rejection_flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateTrack:
    """A time-ordered sequence of associated candidate detections."""

    track_id: str
    site: str
    detection_ids: tuple[str, ...]
    start_time: datetime
    end_time: datetime
    mean_speed_ms: float
    continuity_score: float
    wind_consistency_score: float
