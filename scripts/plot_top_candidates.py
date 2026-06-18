#!/usr/bin/env python
"""Plot top near-track radar candidates for human inspection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    candidates_dir,
    load_config,
    radar_site_from_config,
)


def plot_top_candidates(
    config_path: Path,
    radar_site: str | None = None,
    top_n: int = 10,
) -> list[Path]:
    """Create per-candidate plots and summary plots."""

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    output_dir = candidates_dir(config_path, site)
    top_candidates = pd.read_csv(output_dir / "top_candidates.csv").head(top_n)
    gates = pd.read_csv(output_dir / "near_track_gates.csv")
    top_plot_dir = output_dir / "top_candidate_plots"
    summary_plot_dir = output_dir / "summary_plots"
    top_plot_dir.mkdir(parents=True, exist_ok=True)
    summary_plot_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for _, candidate in top_candidates.iterrows():
        subset = gates[
            (gates["scan_time_utc"] == candidate["scan_time_utc"])
            & (gates["search_window"] == candidate["search_window"])
        ].copy()
        output_paths.append(plot_candidate(candidate, subset, top_plot_dir))

    output_paths.extend(plot_summary_charts(output_dir, summary_plot_dir))
    return output_paths


def plot_candidate(candidate: pd.Series, gates: pd.DataFrame, output_dir: Path) -> Path:
    """Plot nearby gates and one candidate cluster center."""

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    valid = gates.dropna(subset=["reflectivity_dbz"])
    if valid.empty:
        ax.scatter(
            gates["gate_lon_deg"],
            gates["gate_lat_deg"],
            c="0.75",
            s=12,
            label="nearby gates",
        )
    else:
        points = ax.scatter(
            valid["gate_lon_deg"],
            valid["gate_lat_deg"],
            c=valid["reflectivity_dbz"],
            cmap="turbo",
            s=16,
            alpha=0.8,
            label="valid-reflectivity gates",
        )
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label("Reflectivity (dBZ)")

    ax.scatter(
        [candidate["expected_lon_deg"]],
        [candidate["expected_lat_deg"]],
        marker="x",
        c="black",
        s=90,
        linewidths=2,
        label="expected balloon",
    )
    ax.scatter(
        [candidate["candidate_lon_deg"]],
        [candidate["candidate_lat_deg"]],
        marker="*",
        c="red",
        s=140,
        edgecolors="black",
        linewidths=0.6,
        label="candidate center",
    )
    ax.set_title(
        f"{candidate['radar_site']} {candidate['scan_time_utc']} "
        f"score {candidate['candidate_score']:.2f}, alt {candidate['candidate_alt_m']:.0f} m"
    )
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    stamp = (
        str(candidate["scan_time_utc"])
        .replace("-", "")
        .replace(":", "")
        .replace("T", "_")
        .replace("Z", "")
    )
    output_path = (
        output_dir
        / (
            f"rank_{int(candidate['candidate_rank']):02d}_{candidate['radar_site']}_"
            f"{stamp}_score_{candidate['candidate_score']:.2f}.png"
        )
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def plot_summary_charts(output_dir: Path, summary_plot_dir: Path) -> list[Path]:
    """Create score, reflectivity, and altitude summary plots."""

    scores = pd.read_csv(output_dir / "candidate_scores.csv")
    summaries = pd.read_csv(output_dir / "scan_gate_summary.csv")
    paths = [
        plot_time_series(
            scores,
            y_column="candidate_score",
            y_label="Candidate score",
            title="Candidate score vs time",
            output_path=summary_plot_dir / "candidate_score_vs_time.png",
        ),
        plot_time_series(
            summaries[summaries["search_window"] == "normal"],
            y_column="max_reflectivity_dbz",
            y_label="Max reflectivity (dBZ)",
            title="Max reflectivity near expected track vs time",
            output_path=summary_plot_dir / "max_reflectivity_vs_time.png",
        ),
        plot_altitude_summary(scores, summary_plot_dir / "altitude_vs_time.png"),
    ]
    return paths


def plot_time_series(
    frame: pd.DataFrame,
    *,
    y_column: str,
    y_label: str,
    title: str,
    output_path: Path,
) -> Path:
    """Plot one timestamped metric."""

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    data = frame.copy()
    data["scan_time_utc_dt"] = pd.to_datetime(data["scan_time_utc"], utc=True)
    ax.scatter(data["scan_time_utc_dt"], data[y_column], s=28)
    ax.plot(data["scan_time_utc_dt"], data[y_column], alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Scan time (UTC)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def plot_altitude_summary(scores: pd.DataFrame, output_path: Path) -> Path:
    """Plot expected and candidate altitude over scan time."""

    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    data = scores.copy()
    data["scan_time_utc_dt"] = pd.to_datetime(data["scan_time_utc"], utc=True)
    ax.scatter(data["scan_time_utc_dt"], data["expected_alt_m"], s=20, label="expected altitude")
    ax.scatter(data["scan_time_utc_dt"], data["candidate_alt_m"], s=20, label="candidate altitude")
    ax.set_title("Expected and candidate altitude vs time")
    ax.set_xlabel("Scan time (UTC)")
    ax.set_ylabel("Altitude (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site to process, default from config")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top candidates to plot")
    args = parser.parse_args()
    paths = plot_top_candidates(args.config, radar_site=args.radar_site, top_n=args.top_n)
    print(f"Wrote {len(paths)} plots.")


if __name__ == "__main__":
    main()
