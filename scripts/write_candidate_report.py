#!/usr/bin/env python
"""Write a cautious human-readable near-track candidate report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    candidates_dir,
    load_config,
    matches_path,
    radar_site_from_config,
)


def write_candidate_report(config_path: Path, radar_site: str | None = None) -> Path:
    """Write candidate_report.md for a case/radar site."""

    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    case_id = str(config["case_id"])
    output_dir = candidates_dir(config_path, site)
    matches = pd.read_csv(matches_path(config_path, site))
    summaries = pd.read_csv(output_dir / "scan_gate_summary.csv")
    scores = pd.read_csv(output_dir / "candidate_scores.csv")
    top = pd.read_csv(output_dir / "top_candidates.csv")

    scans_analyzed = summaries["scan_time_utc"].nunique()
    scans_inside = int(matches["inside_track_window"].sum())
    scans_with_possible_returns = int(
        summaries.groupby("scan_time_utc")["has_possible_return"].any().sum()
    )
    report_path = output_dir / "candidate_report.md"
    report_path.write_text(
        build_report(
            case_id=case_id,
            radar_site=site,
            scans_analyzed=scans_analyzed,
            scans_inside=scans_inside,
            scans_with_possible_returns=scans_with_possible_returns,
            scores=scores,
            top=top,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {report_path}")
    return report_path


def build_report(
    *,
    case_id: str,
    radar_site: str,
    scans_analyzed: int,
    scans_inside: int,
    scans_with_possible_returns: int,
    scores: pd.DataFrame,
    top: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Build Markdown report text with cautious language."""

    label_counts = scores["candidate_label"].value_counts().to_dict() if not scores.empty else {}
    lines = [
        f"# PicoCAST K7UAZ 2026-03-22 {radar_site} Candidate Report",
        "",
        "## Summary",
        "",
        f"- Case ID: `{case_id}`",
        f"- Radar site: `{radar_site}`",
        f"- Scans analyzed: {scans_analyzed}",
        f"- Scans inside telemetry window: {scans_inside}",
        f"- Scans with possible returns: {scans_with_possible_returns}",
        f"- Candidate clusters scored: {len(scores)}",
        f"- Candidate labels: {label_counts}",
        "",
        "This report highlights near-track radar features for visual inspection. "
        "A high-priority candidate radar return is not a balloon association by itself "
        "and requires visual inspection and multi-radar confirmation.",
        "",
        "## Top Candidates",
        "",
    ]
    if top.empty:
        lines.extend(["No candidate radar return rows were produced.", ""])
    else:
        display_columns = [
            "candidate_rank",
            "scan_time_utc",
            "search_window",
            "candidate_score",
            "candidate_label",
            "horizontal_distance_km",
            "vertical_distance_m",
            "max_reflectivity_dbz",
            "n_gates",
        ]
        lines.extend([markdown_table(top[display_columns]), ""])

    lines.extend(
        [
            "## Plot Paths",
            "",
            f"- Top candidate plots: `{output_dir / 'top_candidate_plots'}`",
            f"- Summary plots: `{output_dir / 'summary_plots'}`",
            "",
            "## Interpretation Notes",
            "",
            "- Use the plots to inspect whether each near-track radar feature is compact "
            "or part of a broader weather/clutter field.",
            "- Prefer candidates that are close to the expected track, close in altitude, "
            "compact, and temporally plausible.",
            "- Do not treat `strong_candidate` as conclusive evidence; it only means "
            "high-priority near-track radar candidate.",
            "- Follow-up should compare neighboring radar sites and inspect Level II moments "
            "around the same scan times.",
            "",
            "## Files",
            "",
            f"- Candidate scores: `{output_dir / 'candidate_scores.csv'}`",
            f"- Top candidates: `{output_dir / 'top_candidates.csv'}`",
            f"- Gate clusters: `{output_dir / 'gate_clusters.csv'}`",
            f"- Near-track gates: `{output_dir / 'near_track_gates.csv'}`",
            f"- Scan gate summary: `{output_dir / 'scan_gate_summary.csv'}`",
            "",
        ]
    )
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without optional dependencies."""

    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(format_cell(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def format_cell(value) -> str:
    """Format a Markdown table cell."""

    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site to process, default from config")
    args = parser.parse_args()
    write_candidate_report(args.config, radar_site=args.radar_site)


if __name__ == "__main__":
    main()
