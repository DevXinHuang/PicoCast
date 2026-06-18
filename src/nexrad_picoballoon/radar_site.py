"""Radar site coordinate resolver.

Extracts radar site latitude, longitude, and altitude from local NEXRAD
scan files using Py-ART.  Falls back to ``radar_sites:`` overrides in
the case ``config.yaml`` if present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return cfg


def _find_nexrad_file(config_path: Path, radar_site: str) -> Path | None:
    """Find the first available raw NEXRAD file for a radar site."""

    case_dir = config_path.parent
    raw_dir = case_dir / "nexrad" / radar_site / "raw"
    if not raw_dir.is_dir():
        return None
    for path in sorted(raw_dir.iterdir()):
        if path.is_file() and not path.name.startswith("."):
            return path
    return None


def resolve_radar_location(
    config_path: Path,
    radar_site: str,
) -> dict[str, Any]:
    """Resolve radar site coordinates.

    Resolution order:
    1. ``radar_sites:<SITE>`` override in ``config.yaml``
    2. Py-ART metadata extracted from the first local NEXRAD file

    Returns
    -------
    dict with keys: site, lat, lon, alt_m
    """

    config = _load_config(config_path)

    # 1. Check for config override
    overrides = config.get("radar_sites", {})
    if isinstance(overrides, dict) and radar_site in overrides:
        site_cfg = overrides[radar_site]
        return {
            "site": radar_site,
            "lat": float(site_cfg["lat"]),
            "lon": float(site_cfg["lon"]),
            "alt_m": float(site_cfg.get("alt_m", 0)),
        }

    # 2. Extract from Py-ART
    try:
        import pyart  # noqa: F811
    except ImportError as exc:
        raise RuntimeError(
            "Py-ART is required to resolve radar site coordinates. "
            'Install it with: python -m pip install -e ".[radar]"'
        ) from exc

    nexrad_file = _find_nexrad_file(config_path, radar_site)
    if nexrad_file is None:
        raise FileNotFoundError(
            f"No NEXRAD files found for {radar_site} under "
            f"{config_path.parent / 'nexrad' / radar_site / 'raw'}. "
            "Download scans first or add a radar_sites: override to config.yaml."
        )

    radar = pyart.io.read_nexrad_archive(str(nexrad_file))
    return {
        "site": radar_site,
        "lat": float(radar.latitude["data"][0]),
        "lon": float(radar.longitude["data"][0]),
        "alt_m": float(radar.altitude["data"][0]),
    }
