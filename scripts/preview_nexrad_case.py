#!/usr/bin/env python
"""Render quick NEXRAD reflectivity previews with expected track overlays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    """Read a PicoCAST case YAML config."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return config


def output_filename(radar_site: str, scan_time_utc: str) -> str:
    """Build a stable preview filename from site and scan time."""

    stamp = scan_time_utc.replace("-", "").replace(":", "").replace("T", "_").replace("Z", "")
    return f"{radar_site}_{stamp}_reflectivity_track_overlay.png"


def reflectivity_field_name(radar) -> str:
    """Return the first usable reflectivity field name in a Py-ART radar object."""

    for candidate in ("reflectivity", "REF", "DBZH", "DBZ"):
        if candidate in radar.fields:
            return candidate
    available = ", ".join(sorted(radar.fields))
    raise ValueError(f"No reflectivity field found. Available fields: {available}")


def radar_origin(radar) -> tuple[float, float]:
    """Return radar latitude and longitude from a Py-ART radar object."""

    lat = float(radar.latitude["data"][0])
    lon = float(radar.longitude["data"][0])
    return lat, lon


def plot_preview_row(row: pd.Series, output_dir: Path) -> Path:
    """Open one NEXRAD file and save a low-elevation reflectivity preview."""

    try:
        import pyart
    except ImportError as exc:
        raise RuntimeError(
            'Preview plotting requires Py-ART. Install it with: python -m pip install -e ".[radar]"'
        ) from exc

    local_path = Path(str(row["scan_local_path"]))
    if not local_path.exists() or local_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing downloaded radar file: {local_path}")

    radar = pyart.io.read_nexrad_archive(str(local_path))
    field = reflectivity_field_name(radar)
    display = pyart.graph.RadarDisplay(radar)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    display.plot_ppi(field, sweep=0, ax=ax, vmin=-20, vmax=60, colorbar_label="dBZ")
    display.set_limits(xlim=(-160, 160), ylim=(-160, 160), ax=ax)

    try:
        radar_lat, radar_lon = radar_origin(radar)
        x_m, y_m = pyart.core.geographic_to_cartesian_aeqd(
            float(row["expected_lon_deg"]),
            float(row["expected_lat_deg"]),
            radar_lon,
            radar_lat,
        )
        x_km = float(np.asarray(x_m).squeeze() / 1000.0)
        y_km = float(np.asarray(y_m).squeeze() / 1000.0)
        ax.scatter([x_km], [y_km], c="red", marker="x", s=80, linewidths=2)
        ax.annotate("expected balloon", (x_km, y_km), color="red")
    except Exception as exc:  # noqa: BLE001 - overlay is helpful but noncritical.
        print(f"Warning: could not overlay expected position for {local_path.name}: {exc}")

    ax.set_title(f"{row['radar_site']} {row['scan_time_utc']} reflectivity")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(str(row["radar_site"]), str(row["scan_time_utc"]))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def render_previews(config_path: Path) -> list[Path]:
    """Render previews for all matched scans that have downloaded files."""

    config = load_config(config_path)
    radar_site = str(config["nexrad"]["primary_radar_site"])
    case_dir = config_path.parent
    matches_path = case_dir / "nexrad" / radar_site / "index" / "scan_track_matches.csv"
    output_dir = case_dir / "outputs" / "radar_preview"
    matches = pd.read_csv(matches_path)
    output_paths: list[Path] = []

    for _, row in matches.iterrows():
        try:
            output_path = plot_preview_row(row, output_dir)
            output_paths.append(output_path)
            print(f"Wrote {output_path}")
        except FileNotFoundError as exc:
            print(f"Warning: {exc}")
        except Exception as exc:  # noqa: BLE001 - keep batch preview rendering moving.
            print(f"Warning: failed to render {row.get('scan_filename', 'unknown')}: {exc}")

    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    args = parser.parse_args()
    outputs = render_previews(args.config)
    print(f"Rendered {len(outputs)} preview plots.")


if __name__ == "__main__":
    main()
