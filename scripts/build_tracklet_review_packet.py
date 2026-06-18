#!/usr/bin/env python
"""Build a compact, cautious review packet for plausible regional tracklets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import horizontal_distance_km, load_config  # noqa: E402

CAUTIOUS_STATUS = (
    "telemetry-consistent near-track radar features requiring visual inspection "
    "and multi-radar confirmation"
)


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required review-packet input is missing: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_ids(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(";") if part.strip()]


def clean_float(value: object, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_review_inputs(config_path: Path) -> dict[str, pd.DataFrame]:
    out_dir = config_path.parent / "outputs" / "discovery"
    return {
        "tracklets": read_required_csv(out_dir / "plausible_tracklets.csv"),
        "points": read_required_csv(out_dir / "plausible_tracklet_points.csv"),
        "telemetry": read_optional_csv(out_dir / "regional_tracklet_telemetry_comparison.csv"),
        "associations": read_optional_csv(out_dir / "plausible_cross_radar_associations.csv"),
        "stability": read_optional_csv(
            config_path.parent / "outputs" / "sweeps" / "tracklet_stability.csv"
        ),
        "expected_track": read_optional_csv(config_path.parent / "expected_track.csv"),
    }


def merge_tracklet_context(
    tracklets: pd.DataFrame,
    telemetry: pd.DataFrame,
    stability: pd.DataFrame,
) -> pd.DataFrame:
    merged = tracklets.copy()

    if not telemetry.empty:
        cols = [
            "tracklet_id",
            "mean_distance_to_corridor_km",
            "max_distance_to_corridor_km",
            "median_altitude_difference_m",
            "speed_consistency_ratio",
            "telemetry_match_label",
        ]
        available = [col for col in cols if col in telemetry.columns]
        merged = merged.merge(telemetry[available], on="tracklet_id", how="left")

    if stability.empty:
        merged["stability_label"] = ""
        merged["detection_fraction"] = np.nan
        merged["detection_count"] = np.nan
        merged["member_tracklet_count"] = np.nan
        merged["n_possible_sweeps"] = np.nan
        return merged

    stability_rows = []
    for _, row in merged.iterrows():
        match = best_stability_match(row, stability)
        stability_rows.append(match)
    stability_df = pd.DataFrame(stability_rows)
    return pd.concat([merged.reset_index(drop=True), stability_df.reset_index(drop=True)], axis=1)


def best_stability_match(tracklet: pd.Series, stability: pd.DataFrame) -> dict:
    same_site = stability[stability["radar_site"] == tracklet["radar_site"]].copy()
    if same_site.empty:
        return empty_stability_context()

    t_start = pd.Timestamp(tracklet["start_time_utc"])
    t_end = pd.Timestamp(tracklet["end_time_utc"])
    duration = max((t_end - t_start).total_seconds(), 1.0)

    scored_rows = []
    for _, row in same_site.iterrows():
        s_start = pd.Timestamp(row["start_time_utc"])
        s_end = pd.Timestamp(row["end_time_utc"])
        overlap = max((min(t_end, s_end) - max(t_start, s_start)).total_seconds(), 0.0)
        coverage = overlap / duration
        start_delta_min = abs((t_start - s_start).total_seconds()) / 60.0
        scored_rows.append((coverage, -start_delta_min, row))

    scored_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = scored_rows[0][2]
    return {
        "stability_signature": best.get("tracklet_signature", ""),
        "stability_label": best.get("stability_label", ""),
        "detection_fraction": clean_float(best.get("detection_fraction")),
        "detection_count": clean_float(best.get("detection_count")),
        "member_tracklet_count": clean_float(best.get("member_tracklet_count")),
        "n_possible_sweeps": clean_float(best.get("n_possible_sweeps")),
    }


def empty_stability_context() -> dict:
    return {
        "stability_signature": "",
        "stability_label": "",
        "detection_fraction": np.nan,
        "detection_count": np.nan,
        "member_tracklet_count": np.nan,
        "n_possible_sweeps": np.nan,
    }


def representative_sort_key(row: pd.Series) -> tuple:
    corridor = clean_float(row.get("mean_distance_to_corridor_km"), default=9999.0)
    return (
        clean_float(row.get("spaghetti_score"), default=9999.0),
        clean_float(row.get("median_abs_vertical_mismatch_m"), default=9999.0),
        corridor,
        -clean_float(row.get("duration_min"), default=0.0),
        -clean_float(row.get("tracklet_score"), default=0.0),
    )


def point_keys(points: pd.DataFrame) -> set[str]:
    if "cluster_id" in points.columns:
        values = points["cluster_id"].dropna().astype(str)
        if not values.empty:
            return set(values)

    keys = []
    for _, row in points.iterrows():
        keys.append(
            "|".join(
                [
                    str(row["scan_time_utc"]),
                    f"{clean_float(row['cluster_lat_deg']):.4f}",
                    f"{clean_float(row['cluster_lon_deg']):.4f}",
                ]
            )
        )
    return set(keys)


def overlap_fraction(a: set[str], b: set[str]) -> float:
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0
    return len(a & b) / denom


def fallback_tracklet_match(a_points: pd.DataFrame, b_points: pd.DataFrame) -> bool:
    merged = a_points.merge(
        b_points,
        on="scan_time_utc",
        suffixes=("_a", "_b"),
    )
    if merged.empty:
        return False

    temporal_overlap = len(merged) / min(len(a_points), len(b_points))
    h_dist = horizontal_distance_km(
        merged["cluster_lat_deg_a"].to_numpy(),
        merged["cluster_lon_deg_a"].to_numpy(),
        merged["cluster_lat_deg_b"].to_numpy(),
        merged["cluster_lon_deg_b"].to_numpy(),
    )
    v_dist = np.abs(
        merged["cluster_alt_m_a"].to_numpy(dtype=float)
        - merged["cluster_alt_m_b"].to_numpy(dtype=float)
    )
    return bool(
        temporal_overlap >= 0.5
        and float(np.nanmedian(h_dist)) <= 8.0
        and float(np.nanmedian(v_dist)) <= 750.0
    )


def tracklets_match(
    first_id: str,
    second_id: str,
    points_by_tracklet: dict[str, pd.DataFrame],
) -> bool:
    first_points = points_by_tracklet[first_id]
    second_points = points_by_tracklet[second_id]
    key_overlap = overlap_fraction(point_keys(first_points), point_keys(second_points))
    if key_overlap >= 0.5:
        return True
    return fallback_tracklet_match(first_points, second_points)


def review_priority_score(row: pd.Series) -> float:
    alt = clean_float(row.get("median_abs_vertical_mismatch_m"), default=9999.0)
    corridor = clean_float(row.get("mean_distance_to_corridor_km"), default=9999.0)
    spaghetti = clean_float(row.get("spaghetti_score"), default=9999.0)
    duration = clean_float(row.get("duration_min"), default=0.0)
    tracklet_score = clean_float(row.get("tracklet_score"), default=0.0)
    stability = clean_float(row.get("detection_fraction"), default=0.0)

    alt_score = float(np.clip((750.0 - alt) / 750.0, 0.0, 1.0))
    corridor_score = float(np.clip((20.0 - corridor) / 20.0, 0.0, 1.0))
    smooth_score = float(np.clip((250.0 - spaghetti) / 250.0, 0.0, 1.0))
    duration_score = float(np.clip(duration / 60.0, 0.0, 1.0))
    track_score = float(np.clip(tracklet_score / 1.5, 0.0, 1.0))

    score = (
        0.25 * alt_score
        + 0.20 * corridor_score
        + 0.15 * smooth_score
        + 0.15 * duration_score
        + 0.15 * track_score
        + 0.10 * stability
    )
    return round(score, 4)


def build_tracklet_families(
    tracklets: pd.DataFrame,
    points: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if tracklets.empty:
        return pd.DataFrame(), {}

    points_by_tracklet = {
        tid: group.sort_values("scan_time_utc").reset_index(drop=True)
        for tid, group in points.groupby("tracklet_id")
    }

    family_rows = []
    tracklet_to_family = {}

    for site, site_df in tracklets.groupby("radar_site"):
        sorted_site = site_df.sort_values(
            by=["start_time_utc", "duration_min", "tracklet_score"],
            ascending=[True, False, False],
        )
        families: list[list[str]] = []

        for _, row in sorted_site.iterrows():
            tid = row["tracklet_id"]
            if tid not in points_by_tracklet:
                continue
            matched_index = None
            for idx, member_ids in enumerate(families):
                has_match = any(
                    tracklets_match(tid, member_id, points_by_tracklet)
                    for member_id in member_ids
                )
                if has_match:
                    matched_index = idx
                    break
            if matched_index is None:
                families.append([tid])
            else:
                families[matched_index].append(tid)

        for idx, member_ids in enumerate(families, start=1):
            family_id = f"{site}_F{idx:03d}"
            member_rows = tracklets[tracklets["tracklet_id"].isin(member_ids)].copy()
            representative = min(
                [row for _, row in member_rows.iterrows()],
                key=representative_sort_key,
            )
            representative_id = representative["tracklet_id"]
            rep_points = points_by_tracklet[representative_id]

            for tid in member_ids:
                tracklet_to_family[tid] = family_id

            family_rows.append({
                "family_id": family_id,
                "radar_site": site,
                "representative_tracklet_id": representative_id,
                "member_tracklet_ids": ";".join(sorted(member_ids)),
                "n_members": len(member_ids),
                "start_time_utc": rep_points["scan_time_utc"].min(),
                "end_time_utc": rep_points["scan_time_utc"].max(),
                "n_points_representative": int(representative["n_points"]),
                "duration_min": round(clean_float(representative["duration_min"]), 1),
                "median_abs_vertical_mismatch_m": round(
                    clean_float(representative["median_abs_vertical_mismatch_m"]), 1
                ),
                "mean_distance_to_corridor_km": round(
                    clean_float(representative.get("mean_distance_to_corridor_km")), 2
                ),
                "spaghetti_score": round(clean_float(representative["spaghetti_score"]), 2),
                "tracklet_score": round(clean_float(representative["tracklet_score"]), 4),
                "quality_label": representative.get("quality_label", ""),
                "stability_label": representative.get("stability_label", ""),
                "detection_fraction": round(
                    clean_float(representative.get("detection_fraction")), 4
                ),
                "review_priority_score": review_priority_score(representative),
            })

    family_df = pd.DataFrame(family_rows)
    if not family_df.empty:
        family_df = family_df.sort_values(
            ["review_priority_score", "radar_site", "family_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    return family_df, tracklet_to_family


def build_cross_radar_queue(
    associations: pd.DataFrame,
    tracklet_to_family: dict[str, str],
) -> pd.DataFrame:
    if associations.empty:
        return pd.DataFrame(
            columns=[
                "cross_radar_rank",
                "association_id",
                "radar_sites",
                "tracklet_ids",
                "family_ids",
                "n_associations_in_family_pair",
                "time_overlap_min",
                "median_horizontal_difference_km",
                "median_altitude_difference_m",
                "cross_radar_score",
                "association_label",
                "review_reason",
            ]
        )

    queue = associations.copy()
    queue["family_ids"] = queue["tracklet_ids"].apply(
        lambda value: ";".join(tracklet_to_family.get(tid, "") for tid in parse_ids(value))
    )
    queue["_family_pair_key"] = queue["family_ids"].apply(
        lambda value: ";".join(sorted(part for part in parse_ids(value) if part))
    )
    label_rank = {
        "strong_cross_radar_candidate": 0,
        "moderate_cross_radar_candidate": 1,
        "weak_cross_radar_candidate": 2,
    }
    queue["_label_rank"] = queue["association_label"].map(label_rank).fillna(9)
    queue = queue.sort_values(
        [
            "_label_rank",
            "cross_radar_score",
            "time_overlap_min",
            "median_horizontal_difference_km",
            "median_altitude_difference_m",
        ],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)
    pair_counts = queue.groupby("_family_pair_key").size().to_dict()
    queue = queue.drop_duplicates("_family_pair_key", keep="first").reset_index(drop=True)
    queue["n_associations_in_family_pair"] = queue["_family_pair_key"].map(pair_counts).fillna(1)
    queue["n_associations_in_family_pair"] = queue["n_associations_in_family_pair"].astype(int)
    queue.insert(0, "cross_radar_rank", np.arange(1, len(queue) + 1))
    queue["review_reason"] = queue.apply(
        lambda row: (
            f"{row['association_label']} with {row['time_overlap_min']:.1f} min overlap, "
            f"{row['median_horizontal_difference_km']:.1f} km median horizontal separation, "
            f"{row['median_altitude_difference_m']:.0f} m median altitude separation "
            f"({row['n_associations_in_family_pair']} raw association variants in this family pair)"
        ),
        axis=1,
    )
    return queue.drop(columns=["_label_rank", "_family_pair_key"])


def build_combined_review_queue(
    family_df: pd.DataFrame,
    cross_radar_queue: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, row in cross_radar_queue.iterrows():
        rows.append({
            "item_type": "cross_radar_association",
            "item_id": row["association_id"],
            "radar_sites": row["radar_sites"],
            "tracklet_ids": row["tracklet_ids"],
            "family_ids": row["family_ids"],
            "review_priority_score": row["cross_radar_score"],
            "review_reason": row["review_reason"],
        })

    for _, row in family_df.iterrows():
        rows.append({
            "item_type": "tracklet_family",
            "item_id": row["family_id"],
            "radar_sites": row["radar_site"],
            "tracklet_ids": row["representative_tracklet_id"],
            "family_ids": row["family_id"],
            "review_priority_score": row["review_priority_score"],
            "review_reason": (
                f"{row['quality_label']} representative; "
                f"{row['median_abs_vertical_mismatch_m']:.0f} m median vertical mismatch, "
                f"{row['mean_distance_to_corridor_km']:.1f} km mean corridor distance"
            ),
        })

    queue = pd.DataFrame(rows)
    if queue.empty:
        return queue
    queue.insert(0, "review_rank", np.arange(1, len(queue) + 1))
    return queue


def plot_review_item(
    item: pd.Series,
    points: pd.DataFrame,
    expected_track: pd.DataFrame,
    output_dir: Path,
) -> str:
    tracklet_ids = parse_ids(item["tracklet_ids"])
    subset = points[points["tracklet_id"].isin(tracklet_ids)].copy()
    if subset.empty:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    ax_map, ax_alt = axes

    if not expected_track.empty and {"lon_deg", "lat_deg"}.issubset(expected_track.columns):
        ax_map.plot(
            expected_track["lon_deg"],
            expected_track["lat_deg"],
            color="black",
            linewidth=1.5,
            alpha=0.45,
            label="expected track",
        )

    for tid, group in subset.groupby("tracklet_id"):
        group = group.sort_values("scan_time_utc")
        ax_map.plot(group["cluster_lon_deg"], group["cluster_lat_deg"], marker="o", label=tid)
        times = pd.to_datetime(group["scan_time_utc"], utc=True)
        ax_alt.plot(times, group["cluster_alt_m"], marker="o", label=tid)
        ax_alt.plot(
            times,
            group["expected_alt_m"],
            color="black",
            alpha=0.25,
            linewidth=1.0,
        )

    ax_map.set_title("Tracklet map")
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(True, alpha=0.25)
    ax_map.legend(fontsize=8)

    ax_alt.set_title("Altitude vs time")
    ax_alt.set_xlabel("UTC")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.grid(True, alpha=0.25)
    ax_alt.legend(fontsize=8)
    fig.autofmt_xdate()

    safe_id = str(item["item_id"]).replace(";", "_").replace("/", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"review_rank_{int(item['review_rank']):02d}_{safe_id}.png"
    fig.suptitle(f"{item['item_type']}: {item['item_id']}", fontsize=12)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path.relative_to(output_dir.parent))


def write_review_report(
    config: dict,
    report_path: Path,
    family_df: pd.DataFrame,
    cross_radar_queue: pd.DataFrame,
    review_queue: pd.DataFrame,
) -> str:
    case_id = config["case_id"]
    lines = [
        f"# PicoCAST Tonight Review Packet - {case_id}",
        "",
        "## Summary",
        "",
        f"This packet ranks {CAUTIOUS_STATUS}. It is a triage artifact for human review, "
        "not a detection claim.",
        "",
        "## Counts",
        "",
        f"- Tracklet families: {len(family_df)}",
        f"- Cross-radar review candidates: {len(cross_radar_queue)}",
        f"- Total review queue items: {len(review_queue)}",
        "",
        "## Top Review Queue",
        "",
        "| Rank | Type | Item | Tracklets | Score | Reason | Plot |",
        "| :---: | :--- | :--- | :--- | :---: | :--- | :--- |",
    ]

    for _, row in review_queue.head(10).iterrows():
        plot_path = row.get("plot_path", "")
        lines.append(
            f"| {int(row['review_rank'])} | `{row['item_type']}` | `{row['item_id']}` | "
            f"`{row['tracklet_ids']}` | {clean_float(row['review_priority_score']):.3f} | "
            f"{row['review_reason']} | {plot_path} |"
        )

    lines.extend([
        "",
        "## Cross-Radar Candidates",
        "",
        "| Rank | Association | Tracklets | Label | Overlap | Median H Sep | Median V Sep |",
        "| :---: | :--- | :--- | :--- | :---: | :---: | :---: |",
    ])
    for _, row in cross_radar_queue.head(10).iterrows():
        lines.append(
            f"| {int(row['cross_radar_rank'])} | `{row['association_id']}` | "
            f"`{row['tracklet_ids']}` | `{row['association_label']}` | "
            f"{row['time_overlap_min']:.1f} min | "
            f"{row['median_horizontal_difference_km']:.1f} km | "
            f"{row['median_altitude_difference_m']:.0f} m |"
        )

    lines.extend([
        "",
        "## Interpretation Guardrails",
        "",
        "- Treat these rows as candidate radar returns near a known telemetry corridor.",
        "- Prioritize visual inspection of the ranked plots and dashboard before any modeling.",
        "- Do not use this packet to identify the balloon by itself; radar artifacts, clutter, "
        "and weather-adjacent returns remain possible.",
        "- The next scientific step is visual inspection and multi-radar confirmation.",
        "",
    ])

    text = "\n".join(lines)
    report_path.write_text(text, encoding="utf-8")
    return text


def build_review_packet(config_path: Path, top_n: int = 10) -> dict[str, Path]:
    config = load_config(config_path)
    case_dir = config_path.parent
    out_dir = case_dir / "outputs" / "discovery" / "review_packet"
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    for stale_plot in plot_dir.glob("review_rank_*.png"):
        stale_plot.unlink()

    inputs = load_review_inputs(config_path)
    tracklets = merge_tracklet_context(
        inputs["tracklets"],
        inputs["telemetry"],
        inputs["stability"],
    )
    family_df, tracklet_to_family = build_tracklet_families(tracklets, inputs["points"])
    cross_radar_queue = build_cross_radar_queue(inputs["associations"], tracklet_to_family)
    review_queue = build_combined_review_queue(family_df, cross_radar_queue)

    if not review_queue.empty:
        plot_paths = []
        for _, row in review_queue.head(top_n).iterrows():
            plot_paths.append(
                plot_review_item(row, inputs["points"], inputs["expected_track"], plot_dir)
            )
        review_queue.loc[review_queue.index[: len(plot_paths)], "plot_path"] = plot_paths
        review_queue["plot_path"] = review_queue["plot_path"].fillna("")

    family_path = out_dir / "tracklet_family_summary.csv"
    tracklet_queue_path = out_dir / "tracklet_review_queue.csv"
    cross_queue_path = out_dir / "cross_radar_review_queue.csv"
    report_path = out_dir / "tonight_review_packet.md"

    family_df.to_csv(family_path, index=False)
    review_queue.to_csv(tracklet_queue_path, index=False)
    cross_radar_queue.to_csv(cross_queue_path, index=False)
    write_review_report(config, report_path, family_df, cross_radar_queue, review_queue)

    return {
        "family_summary": family_path,
        "tracklet_review_queue": tracklet_queue_path,
        "cross_radar_review_queue": cross_queue_path,
        "report": report_path,
        "plot_dir": plot_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument("--top-n", type=int, default=10, help="Number of review plots to render")
    args = parser.parse_args()

    outputs = build_review_packet(args.config, top_n=args.top_n)
    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")


if __name__ == "__main__":
    main()
