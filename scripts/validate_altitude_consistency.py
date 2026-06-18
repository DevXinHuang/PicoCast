#!/usr/bin/env python
"""Altitude-first validation of near-track radar candidates.

Uses balloon telemetry altitude — which is much more reliable than the
Maidenhead-derived horizontal position — as the primary discriminator for
candidate quality.

Outputs (to <case_dir>/outputs/candidates/<SITE>/altitude_validation/):
  - altitude_time_candidate_overlay.png
  - vertical_mismatch_vs_time.png
  - nearby_gate_altitude_distribution.png
  - altitude_priority_rank_vs_original_rank.png
  - altitude_prioritized_candidates.csv
  - altitude_validation_report.md
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (  # noqa: E402
    candidates_dir,
    load_config,
    radar_site_from_config,
)

# ---------------------------------------------------------------------------
# Altitude consistency scoring
# ---------------------------------------------------------------------------

ALTITUDE_THRESHOLDS = [
    (250.0, 1.0, "excellent_altitude_match"),
    (500.0, 0.8, "strong_altitude_match"),
    (1000.0, 0.6, "moderate_altitude_match"),
    (2000.0, 0.3, "weak_altitude_match"),
]
DEFAULT_LABEL = "poor_altitude_match"
DEFAULT_SCORE = 0.0


def altitude_consistency_score(abs_vertical_m: float) -> float:
    """Return 0–1 score based on absolute vertical mismatch."""
    if np.isnan(abs_vertical_m):
        return DEFAULT_SCORE
    for threshold, score, _ in ALTITUDE_THRESHOLDS:
        if abs_vertical_m <= threshold:
            return score
    return DEFAULT_SCORE


def altitude_consistency_label(abs_vertical_m: float) -> str:
    """Return categorical label based on absolute vertical mismatch."""
    if np.isnan(abs_vertical_m):
        return DEFAULT_LABEL
    for threshold, _, label in ALTITUDE_THRESHOLDS:
        if abs_vertical_m <= threshold:
            return label
    return DEFAULT_LABEL


# ---------------------------------------------------------------------------
# Altitude interpolation
# ---------------------------------------------------------------------------


def interpolate_expected_altitude(
    scan_time_utc: str,
    track: pd.DataFrame,
) -> float:
    """Linearly interpolate expected altitude at a scan time.

    Uses the expected_track telemetry points.  If the scan falls outside
    the track time range the nearest endpoint altitude is returned.
    """
    scan_dt = pd.Timestamp(scan_time_utc)
    track_times = pd.to_datetime(track["time_utc"])
    if scan_dt <= track_times.iloc[0]:
        return float(track["alt_m"].iloc[0])
    if scan_dt >= track_times.iloc[-1]:
        return float(track["alt_m"].iloc[-1])
    return float(np.interp(
        scan_dt.timestamp(),
        track_times.map(lambda t: t.timestamp()),
        track["alt_m"].astype(float),
    ))


# ---------------------------------------------------------------------------
# Build altitude-prioritized candidate table
# ---------------------------------------------------------------------------


def build_altitude_prioritized(
    candidates: pd.DataFrame,
    track: pd.DataFrame,
) -> pd.DataFrame:
    """Add altitude columns and re-rank by altitude-first criteria."""
    df = candidates.copy()

    # Signed and absolute vertical mismatch
    df["signed_vertical_m"] = df["candidate_alt_m"] - df["expected_alt_m"]
    df["abs_vertical_distance_m"] = df["signed_vertical_m"].abs()

    # Interpolated expected altitude (for scan times between telemetry pts)
    df["interpolated_expected_alt_m"] = df["scan_time_utc"].apply(
        lambda t: interpolate_expected_altitude(t, track)
    )
    df["signed_vertical_interp_m"] = (
        df["candidate_alt_m"] - df["interpolated_expected_alt_m"]
    )
    df["abs_vertical_interp_m"] = df["signed_vertical_interp_m"].abs()

    # Use the interpolated value for scoring (more accurate at each scan time)
    df["altitude_consistency_score"] = df["abs_vertical_interp_m"].apply(
        altitude_consistency_score
    )
    df["altitude_consistency_label"] = df["abs_vertical_interp_m"].apply(
        altitude_consistency_label
    )

    # Preserve original rank
    df["original_candidate_rank"] = df["candidate_rank"]

    # Altitude-priority sort:
    # 1. altitude_consistency_score DESC
    # 2. compactness_score DESC
    # 3. candidate_score DESC (original composite)
    # 4. horizontal_distance_km ASC (later tiebreaker — grid-center distance)
    df = df.sort_values(
        [
            "altitude_consistency_score",
            "compactness_score",
            "candidate_score",
            "horizontal_distance_km",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    df["altitude_priority_rank"] = np.arange(1, len(df) + 1)

    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

PLOT_DPI = 180
LABEL_RANKS = {1, 3, 6, 7, 9}


def _parse_utc(ts: str) -> datetime:
    """Parse a UTC timestamp string."""
    ts = ts.rstrip("Z")
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=UTC)


def plot_altitude_time_overlay(
    df: pd.DataFrame,
    track: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 1: Expected altitude curve + candidate altitudes vs time."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Expected altitude curve from telemetry
    track_times = pd.to_datetime(track["time_utc"])
    ax.plot(
        track_times, track["alt_m"],
        "o-", color="#2196F3", linewidth=2, markersize=6,
        label="Expected balloon altitude (telemetry)", zorder=5,
    )

    # ±500 m band (interpolated)
    fine_times = pd.date_range(track_times.min(), track_times.max(), periods=200)
    fine_alts = np.interp(
        [t.timestamp() for t in fine_times],
        [t.timestamp() for t in track_times],
        track["alt_m"].astype(float),
    )
    ax.fill_between(
        fine_times, fine_alts - 500, fine_alts + 500,
        alpha=0.12, color="#2196F3", label="±500 m band",
    )

    # All candidates — scatter colored by candidate_score
    cand_times = pd.to_datetime(df["scan_time_utc"])
    scatter = ax.scatter(
        cand_times, df["candidate_alt_m"],
        c=df["candidate_score"], cmap="RdYlGn", s=30,
        edgecolors="#333", linewidth=0.5, zorder=3,
        label="All candidates (color = score)",
    )
    plt.colorbar(scatter, ax=ax, label="candidate_score", shrink=0.7, pad=0.02)

    # Highlight top candidates
    for _, row in df.iterrows():
        orig_rank = int(row["original_candidate_rank"])
        if orig_rank in LABEL_RANKS:
            t = _parse_utc(row["scan_time_utc"])
            ax.scatter(
                [t], [row["candidate_alt_m"]],
                s=120, facecolors="none", edgecolors="#E91E63",
                linewidths=2, zorder=6,
            )
            ax.annotate(
                f"R{orig_rank}",
                (t, row["candidate_alt_m"]),
                xytext=(6, 8), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#E91E63",
            )

    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_ylabel("Altitude (m)", fontsize=11)
    ax.set_title(
        "Candidate Altitude vs. Expected Balloon Altitude",
        fontsize=14, fontweight="bold",
    )
    ax.text(
        0.5, -0.10,
        "Horizontal position from Maidenhead grid squares (~several km uncertainty). "
        "Altitude from telemetry is more reliable.\n"
        "Expected altitude at scan time is linearly interpolated from telemetry points.",
        transform=ax.transAxes, ha="center", fontsize=8, color="#666",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_vertical_mismatch(
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot 2: Signed vertical mismatch vs time."""
    fig, ax = plt.subplots(figsize=(14, 6))

    cand_times = pd.to_datetime(df["scan_time_utc"])

    # Guide lines
    for dist in [250, 500, 1000]:
        ax.axhline(dist, color="#bbb", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.axhline(-dist, color="#bbb", linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(
            cand_times.max(), dist, f"+{dist} m",
            ha="right", va="bottom", fontsize=7, color="#999",
        )
        ax.text(
            cand_times.max(), -dist, f"−{dist} m",
            ha="right", va="top", fontsize=7, color="#999",
        )

    ax.axhline(0, color="black", linewidth=1, zorder=4)

    # All candidates
    ax.scatter(
        cand_times, df["signed_vertical_interp_m"],
        c="#aaa", s=20, alpha=0.5, edgecolors="none", label="All candidates",
    )

    # Highlight top candidates
    for _, row in df.iterrows():
        orig_rank = int(row["original_candidate_rank"])
        if orig_rank in LABEL_RANKS:
            t = _parse_utc(row["scan_time_utc"])
            ax.scatter(
                [t], [row["signed_vertical_interp_m"]],
                s=100, edgecolors="#E91E63", facecolors="#FF5252",
                linewidths=1.5, zorder=6,
            )
            ax.annotate(
                f"R{orig_rank}",
                (t, row["signed_vertical_interp_m"]),
                xytext=(6, 8), textcoords="offset points",
                fontsize=9, fontweight="bold", color="#E91E63",
            )

    ax.set_xlabel("Scan Time (UTC)", fontsize=11)
    ax.set_ylabel("Signed Vertical Mismatch (m)\n(candidate − expected)", fontsize=11)
    ax.set_title(
        "Vertical Mismatch vs. Time (signed: candidate_alt − expected_alt)",
        fontsize=13, fontweight="bold",
    )
    ax.text(
        0.5, -0.12,
        "Expected altitude linearly interpolated from telemetry. "
        "Radar beam-center altitude has beam-width uncertainty, especially at range.",
        transform=ax.transAxes, ha="center", fontsize=8, color="#666",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_gate_altitude_distribution(
    df: pd.DataFrame,
    track: pd.DataFrame,
    gates_path: Path,
    out_path: Path,
    n_panels: int = 6,
) -> None:
    """Plot 3: Gate altitude histograms for key scan times."""
    # Pick scan times spread across the flight
    unique_scans = sorted(df["scan_time_utc"].unique())
    if len(unique_scans) <= n_panels:
        selected_scans = unique_scans
    else:
        indices = np.linspace(0, len(unique_scans) - 1, n_panels, dtype=int)
        selected_scans = [unique_scans[i] for i in indices]

    n = len(selected_scans)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    # Load gate altitudes for these scan times from near_track_gates.csv
    scan_set = set(selected_scans)
    gate_alts: dict[str, list[float]] = {s: [] for s in selected_scans}

    if gates_path.exists():
        for chunk in pd.read_csv(gates_path, chunksize=200_000):
            mask = chunk["scan_time_utc"].isin(scan_set)
            for scan_time in selected_scans:
                matched = chunk[mask & (chunk["scan_time_utc"] == scan_time)]
                if not matched.empty:
                    gate_alts[scan_time].extend(matched["gate_alt_m"].dropna().tolist())

    for idx, scan_time in enumerate(selected_scans):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        alts = gate_alts[scan_time]
        exp_alt = interpolate_expected_altitude(scan_time, track)

        # Best candidate at this scan time
        scan_cands = df[df["scan_time_utc"] == scan_time]
        best_cand_alt = None
        best_rank = None
        if not scan_cands.empty:
            best = scan_cands.iloc[0]
            best_cand_alt = best["candidate_alt_m"]
            best_rank = int(best["original_candidate_rank"])

        if alts:
            ax.hist(alts, bins=40, color="#78909C", alpha=0.7, edgecolor="#546E7A")
        ax.axvline(
            exp_alt, color="#2196F3", linewidth=2, linestyle="--",
            label=f"Expected: {exp_alt:.0f} m",
        )
        if best_cand_alt is not None:
            ax.axvline(
                best_cand_alt, color="#E91E63", linewidth=2,
                label=f"Best cand (R{best_rank}): {best_cand_alt:.0f} m",
            )

        time_short = scan_time.split("T")[1].rstrip("Z")[:8]
        ax.set_title(f"{time_short} UTC ({len(alts)} gates)", fontsize=10)
        ax.set_xlabel("Altitude (m)", fontsize=9)
        ax.set_ylabel("Gate count", fontsize=9)
        ax.legend(fontsize=7, loc="upper right")

    # Hide empty subplots
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Nearby Radar Gate Altitude Distributions at Key Scan Times",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_rank_comparison(
    df: pd.DataFrame,
    out_path: Path,
    n_top: int = 30,
) -> None:
    """Plot 4: Altitude priority rank vs original rank."""
    top = df[df["original_candidate_rank"] <= n_top].copy()

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot([1, n_top], [1, n_top], "k--", alpha=0.3, label="No change line")

    ax.scatter(
        top["original_candidate_rank"],
        top["altitude_priority_rank"],
        c=top["altitude_consistency_score"],
        cmap="RdYlGn", s=80, edgecolors="#333", linewidth=0.5, zorder=5,
    )

    for _, row in top.iterrows():
        orig = int(row["original_candidate_rank"])
        alt_rank = int(row["altitude_priority_rank"])
        if orig in LABEL_RANKS or abs(orig - alt_rank) >= 5:
            ax.annotate(
                f"R{orig}→{alt_rank}",
                (orig, alt_rank),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("Original Candidate Rank (composite score)", fontsize=11)
    ax.set_ylabel("Altitude Priority Rank", fontsize=11)
    ax.set_title(
        "Altitude-First Rank vs. Original Rank\n"
        "(points above diagonal = demoted by altitude; below = promoted)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(0.5, n_top + 0.5)
    ax.set_ylim(0.5, n_top + 0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(
    df: pd.DataFrame,
    track: pd.DataFrame,
    case_id: str,
    site: str,
    out_path: Path,
) -> None:
    """Generate cautious altitude validation report."""
    # Summary stats
    top = df[df["altitude_priority_rank"] <= 10]
    exc = (df["altitude_consistency_label"] == "excellent_altitude_match").sum()
    strong = (df["altitude_consistency_label"] == "strong_altitude_match").sum()
    moderate = (df["altitude_consistency_label"] == "moderate_altitude_match").sum()
    total = len(df)

    # Focused ranks table
    focus_ranks = {1, 3, 6, 7, 9}
    focus_rows = df[df["original_candidate_rank"].isin(focus_ranks)].sort_values(
        "original_candidate_rank"
    )

    report = textwrap.dedent(f"""\
    # Altitude Validation Report — {case_id}

    **Radar site:** {site}
    **Generated:** {datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")}

    ---

    ## Position Uncertainty Context

    The balloon horizontal position is estimated from 6-character Maidenhead grid
    squares. At this latitude, one grid square is several kilometers across, so
    horizontal offsets of a few km may still be consistent with the telemetry
    region.

    Balloon altitude, by contrast, is reported directly in the telemetry and is
    treated as substantially more reliable than the horizontal position.
    **Altitude consistency is therefore used as the primary discriminator for
    candidate quality.**

    Expected altitude at each radar scan time is **linearly interpolated** from
    the telemetry points. This interpolation assumes roughly constant ascent rate
    between telemetry reports, which may introduce small errors if the balloon
    experienced altitude changes between reports.

    > **Radar beam-center caveat:** The radar gate altitude reported here is the
    > *beam-center* altitude for each gate. The physical radar beam has finite
    > width (beam-width), and this width increases with range from the radar.
    > At the ranges involved (~50–200 km from {site}), the beam may span several
    > hundred meters vertically. A candidate whose beam-center altitude is off by
    > a few hundred meters could still have the balloon within the beam volume.

    ---

    ## Altitude Consistency Summary

    | Category | Count | % of {total} candidates |
    |----------|------:|----:|
    | Excellent (≤ 250 m) | {exc} | {exc/total*100:.1f}% |
    | Strong (≤ 500 m) | {strong} | {strong/total*100:.1f}% |
    | Moderate (≤ 1000 m) | {moderate} | {moderate/total*100:.1f}% |
    | Total candidates scored | {total} | 100% |

    ---

    ## Top 10 Altitude-Prioritized Candidates

    | Alt Rank | Orig Rank | Scan Time | Alt Match | Vert Mismatch | H-dist | Score | Label |
    |:--------:|:---------:|-----------|-----------|:-------------:|:------:|:-----:|-------|
    """)

    for _, row in top.iterrows():
        scan_short = row["scan_time_utc"].split("T")[1].rstrip("Z")[:8]
        report += (
            f"| {int(row['altitude_priority_rank'])} "
            f"| {int(row['original_candidate_rank'])} "
            f"| {scan_short} "
            f"| {row['altitude_consistency_label'].replace('_', ' ')} "
            f"| {row['signed_vertical_interp_m']:+.0f} m "
            f"| {row['horizontal_distance_km']:.1f} km "
            f"| {row['candidate_score']:.3f} "
            f"| {row['candidate_label']} |\n"
        )

    report += textwrap.dedent("""
    ---

    ## Focus: Original Ranks 1, 3, 6, 7, 9

    | Orig Rank | Alt Rank | Scan Time | Signed Vert | Abs Vert | Alt Label | Cand Score |
    |:---------:|:--------:|-----------|:-----------:|:--------:|-----------|:----------:|
    """)

    for _, row in focus_rows.iterrows():
        scan_short = row["scan_time_utc"].split("T")[1].rstrip("Z")[:8]
        report += (
            f"| {int(row['original_candidate_rank'])} "
            f"| {int(row['altitude_priority_rank'])} "
            f"| {scan_short} "
            f"| {row['signed_vertical_interp_m']:+.0f} m "
            f"| {row['abs_vertical_interp_m']:.0f} m "
            f"| {row['altitude_consistency_label'].replace('_', ' ')} "
            f"| {row['candidate_score']:.3f} |\n"
        )

    # Altitude trend assessment
    top_alts_follow = False
    if len(focus_rows) >= 3:
        focus_sorted = focus_rows.sort_values("scan_time_utc")
        alt_diffs = focus_sorted["abs_vertical_interp_m"]
        top_alts_follow = (alt_diffs <= 1000).mean() >= 0.6

    if top_alts_follow:
        trend_text = (
            "The majority of focused candidates (ranks 1, 3, 6, 7, 9) have "
            "altitude mismatches within 1000 m of the expected balloon altitude, "
            "suggesting **altitude consistency with the telemetry profile**. "
            "This is a necessary but not sufficient condition for a genuine "
            "balloon return."
        )
    else:
        trend_text = (
            "The focused candidates show mixed altitude agreement. Some candidates "
            "match the expected altitude profile well, while others do not. "
            "This does **not** rule out a genuine balloon return among the "
            "altitude-consistent candidates, but it means not all top-ranked "
            "candidates can be explained by balloon altitude alone."
        )

    report += textwrap.dedent(f"""
    ---

    ## Altitude Trend Assessment

    {trend_text}

    ---

    ## Terminology

    This analysis uses the term **"altitude-consistent near-track candidate"** to
    describe radar returns that:
    1. Appear near the expected Maidenhead grid-square region
    2. Have radar gate altitudes consistent with the balloon telemetry altitude

    This does **not** constitute a detection claim. Many factors — including
    weather returns, ground clutter, and biological targets — can produce radar
    returns at similar altitudes. The altitude consistency merely raises the
    prior probability that a given candidate is the balloon.

    ---

    ## Caveats

    - Horizontal position derives from 6-character Maidenhead grid squares
      (~4.6 × 7.1 km at this latitude). Distance-from-grid-center is not
      distance-from-balloon.
    - Radar gate altitude is the beam-center altitude. The beam has finite
      width, increasing with range from {site}. A mismatch of a few hundred
      meters does not necessarily mean the balloon was outside the beam.
    - Expected altitude at scan time is linearly interpolated from telemetry
      points, assuming roughly constant ascent rate between reports.
    - Altitude consistency is necessary but not sufficient evidence for
      identifying a picoballoon radar return.
    """)

    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def validate_altitude_consistency(
    config_path: Path,
    radar_site: str | None = None,
) -> Path:
    """Run altitude-first validation and generate all outputs."""
    config = load_config(config_path)
    site = radar_site_from_config(config, radar_site)
    case_id = config["case_id"]
    case_dir = config_path.parent
    cand_dir = candidates_dir(config_path, site)

    # Output directory
    out_dir = cand_dir / "altitude_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    track = pd.read_csv(case_dir / "expected_track.csv")
    track = track.sort_values("time_utc").reset_index(drop=True)

    candidates = pd.read_csv(cand_dir / "candidate_scores.csv")

    # Build altitude-prioritized table
    df = build_altitude_prioritized(candidates, track)

    # Write CSV
    csv_path = out_dir / "altitude_prioritized_candidates.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Plot 1: Altitude vs time
    plot_altitude_time_overlay(
        df, track, out_dir / "altitude_time_candidate_overlay.png",
    )

    # Plot 2: Vertical mismatch vs time (signed)
    plot_vertical_mismatch(df, out_dir / "vertical_mismatch_vs_time.png")

    # Plot 3: Gate altitude distributions
    gates_path = cand_dir / "near_track_gates.csv"
    plot_gate_altitude_distribution(
        df, track, gates_path,
        out_dir / "nearby_gate_altitude_distribution.png",
    )

    # Plot 4: Rank comparison
    plot_rank_comparison(
        df, out_dir / "altitude_priority_rank_vs_original_rank.png",
    )

    # Report
    generate_report(df, track, case_id, site, out_dir / "altitude_validation_report.md")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--radar-site", help="Radar site, default from config")
    args = parser.parse_args()
    validate_altitude_consistency(args.config, radar_site=args.radar_site)


if __name__ == "__main__":
    main()
