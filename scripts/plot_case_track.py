#!/usr/bin/env python
"""Plot a PicoCAST validation case expected track."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return config


def plot_case_track(config_path: Path) -> Path:
    """Plot longitude vs latitude, colored by altitude."""

    config = load_config(config_path)
    case_dir = config_path.parent
    track_path = case_dir / "expected_track.csv"
    output_path = case_dir / "outputs" / "track_preview.png"
    track = pd.read_csv(track_path)

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)
    points = ax.scatter(
        track["lon_deg"],
        track["lat_deg"],
        c=track["alt_km"],
        cmap="viridis",
        s=60,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.plot(track["lon_deg"], track["lat_deg"], color="0.35", linewidth=1.0, alpha=0.7)
    ax.annotate(
        "Start",
        (track["lon_deg"].iloc[0], track["lat_deg"].iloc[0]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.annotate(
        "End",
        (track["lon_deg"].iloc[-1], track["lat_deg"].iloc[-1]),
        xytext=(8, -12),
        textcoords="offset points",
    )
    ax.set_title(str(config.get("case_name", config["case_id"])))
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, alpha=0.25)
    colorbar = fig.colorbar(points, ax=ax)
    colorbar.set_label("Altitude (km)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    args = parser.parse_args()
    output_path = plot_case_track(args.config)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
