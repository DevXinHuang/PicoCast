"""Evaluation summaries for candidate outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_candidates(candidates: pd.DataFrame, truth_dir: Path | None = None) -> str:
    """Create a small Markdown evaluation summary."""

    candidate_count = len(candidates)
    sites = sorted(candidates["site"].dropna().unique()) if "site" in candidates else []
    mean_score = (
        float(candidates["score"].mean()) if candidate_count and "score" in candidates else 0.0
    )
    truth_note = "No truth files supplied."
    if truth_dir and truth_dir.exists():
        truth_files = sorted(path.name for path in truth_dir.iterdir() if path.is_file())
        truth_note = f"Truth files present: {len(truth_files)}."

    return "\n".join(
        [
            "# Candidate Evaluation Summary",
            "",
            f"- Candidate detections: {candidate_count}",
            f"- Sites: {', '.join(sites) if sites else 'none'}",
            f"- Mean score: {mean_score:.3f}",
            f"- Truth data: {truth_note}",
            "",
            "This MVP summary reports detector output volume and basic scoring only. "
            "Probability of detection, false alarms per radar-hour, RMSE, and track "
            "continuity require controlled truth data.",
            "",
        ]
    )


def write_summary(summary: str, out_path: Path) -> Path:
    """Write a Markdown evaluation summary."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    return out_path
