#!/usr/bin/env python
# ruff: noqa: E501, E402
"""Phase 1: Build machine-learning-ready feature tables and labeling templates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (
    load_config,
)


def main():
    parser = argparse.ArgumentParser(description="Build ML feature tables.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")
    parser.add_argument("--primary-sites", action="store_true", help="Process primary sites only")
    parser.add_argument("--all-sites", action="store_true", help="Process all sites")
    parser.add_argument("--radar-site", type=str, help="Process a single specific radar site")

    args = parser.parse_args()
    config = load_config(args.config)
    case_dir = args.config.parent

    # Output directory
    out_ml_dir = case_dir / "outputs" / "ml"
    out_ml_dir.mkdir(parents=True, exist_ok=True)

    # Read geometry report
    geom_csv = case_dir / "nexrad" / "regional_radar_geometry.csv"
    if not geom_csv.exists():
        raise FileNotFoundError(f"Geometry report {geom_csv} not found.")

    geom_df = pd.read_csv(geom_csv)
    included_radars = geom_df[geom_df["geometry_status"] == "include"]["radar_site"].tolist()

    discovery = config.get("discovery", {})
    primary = discovery.get("radar_sites_primary", [])
    secondary = discovery.get("radar_sites_secondary", [])

    if args.radar_site:
        target_sites = [args.radar_site]
    elif args.all_sites:
        target_sites = primary + secondary
    elif args.primary_sites:
        target_sites = primary
    else:
        target_sites = primary

    active_sites = [r for r in target_sites if r in included_radars]
    out_dir = case_dir / "outputs" / "discovery"

    # 1. Load raw discovered clusters parquet
    clusters_path = out_dir / "regional_discovered_clusters.parquet"
    if not clusters_path.exists():
        print(f"Clusters parquet not found at {clusters_path}. Writing empty tables.")
        pd.DataFrame().to_parquet(out_ml_dir / "cluster_features.parquet")
        pd.DataFrame().to_parquet(out_ml_dir / "tracklet_features.parquet")
        pd.DataFrame(columns=["object_id", "object_type", "radar_site", "start_time_utc", "end_time_utc", "n_points_or_gates", "key_features", "dashboard_path", "manual_label", "reviewer_notes"]).to_csv(out_ml_dir / "manual_labels_template.csv", index=False)
        return

    clusters_df = pd.read_parquet(clusters_path)

    # Load tracklet files per site
    tracklet_summaries = []
    tracklet_points = []

    for site in active_sites:
        t_csv = out_dir / site / "candidate_tracklets.csv"
        p_csv = out_dir / site / "tracklet_points.csv"

        if t_csv.exists() and p_csv.exists():
            tdf = pd.read_csv(t_csv)
            pdf = pd.read_csv(p_csv)
            if not tdf.empty:
                tracklet_summaries.append(tdf)
                tracklet_points.append(pdf)

    # Load diagnostics and associations
    diag_csv = out_dir / "tracklet_quality_diagnostics.csv"
    assoc_csv = out_dir / "cross_radar_tracklet_associations.csv"

    diag_df = pd.read_csv(diag_csv) if diag_csv.exists() else pd.DataFrame()
    assoc_df = pd.read_csv(assoc_csv) if assoc_csv.exists() else pd.DataFrame()

    has_tracklets = len(tracklet_summaries) > 0
    merged_tracklets_df = pd.concat(tracklet_summaries, ignore_index=True) if has_tracklets else pd.DataFrame()
    merged_points_df = pd.concat(tracklet_points, ignore_index=True) if has_tracklets else pd.DataFrame()

    # 2. Build cluster features
    print("Building cluster features table...")
    # Map points in tracklets
    tracklet_lookup = {}
    if not merged_points_df.empty:
        for _, grp in merged_points_df.groupby("tracklet_id"):
            tid = grp.iloc[0]["tracklet_id"]
            for _, pt in grp.iterrows():
                # Key: (radar_site, scan_time_utc, cluster_id)
                key = (pt["radar_site"], str(pt["scan_time_utc"]), str(pt["cluster_id"]))
                tracklet_lookup[key] = tid

    is_plausible_lookup = {}
    if not diag_df.empty:
        is_plausible_lookup = dict(zip(diag_df["tracklet_id"], diag_df["status"] == "plausible", strict=True))

    is_in_tracklet = []
    tracklet_id_col = []
    is_plausible_col = []

    for _, row in clusters_df.iterrows():
        key = (row["radar_site"], str(row["scan_time_utc"]), str(row["cluster_id"]))
        tid = tracklet_lookup.get(key, None)
        if tid:
            is_in_tracklet.append(True)
            tracklet_id_col.append(tid)
            is_plausible_col.append(is_plausible_lookup.get(tid, False))
        else:
            is_in_tracklet.append(False)
            tracklet_id_col.append(None)
            is_plausible_col.append(False)

    clusters_df["is_in_tracklet"] = is_in_tracklet
    clusters_df["tracklet_id"] = tracklet_id_col
    clusters_df["is_plausible"] = is_plausible_col

    clusters_df.to_parquet(out_ml_dir / "cluster_features.parquet", index=False)
    print(f"Wrote cluster features: {out_ml_dir / 'cluster_features.parquet'}")

    # 3. Build tracklet features
    print("Building tracklet features table...")
    if not merged_tracklets_df.empty:
        # Join quality diagnostics
        if not diag_df.empty:
            merged_tracklets_df = pd.merge(
                merged_tracklets_df,
                diag_df[["tracklet_id", "spaghetti_score", "quality_label", "status"]],
                on="tracklet_id",
                how="left"
            )
        else:
            merged_tracklets_df["spaghetti_score"] = 0.0
            merged_tracklets_df["quality_label"] = "weak_plausible_tracklet"
            merged_tracklets_df["status"] = "plausible"

        # Join association stats
        is_associated = []
        n_associations = []
        max_assoc_label = []

        assoc_count = {}
        assoc_max_label = {}
        if not assoc_df.empty:
            for _, r in assoc_df.iterrows():
                tids = str(r["tracklet_ids"]).split(";")
                label = r["association_label"]
                for tid in tids:
                    assoc_count[tid] = assoc_count.get(tid, 0) + 1
                    # Rank labels: strong > moderate > weak
                    if tid not in assoc_max_label:
                        assoc_max_label[tid] = label
                    else:
                        current = assoc_max_label[tid]
                        if label == "strong_cross_radar_candidate" or (label == "moderate_cross_radar_candidate" and current == "weak_cross_radar_candidate"):
                            assoc_max_label[tid] = label

        for _, row in merged_tracklets_df.iterrows():
            tid = row["tracklet_id"]
            cnt = assoc_count.get(tid, 0)
            is_associated.append(cnt > 0)
            n_associations.append(cnt)
            max_assoc_label.append(assoc_max_label.get(tid, "none"))

        merged_tracklets_df["is_associated"] = is_associated
        merged_tracklets_df["n_associations"] = n_associations
        merged_tracklets_df["max_association_label"] = max_assoc_label

        # Aggregate cluster-level metrics per tracklet
        agg_features = []
        for tid, grp in merged_points_df.groupby("tracklet_id"):
            agg_features.append({
                "tracklet_id": tid,
                "mean_n_gates": grp["n_gates"].mean(),
                "max_n_gates": grp["n_gates"].max(),
                "mean_max_reflectivity_dbz": grp["max_reflectivity_dbz"].mean(),
                "max_max_reflectivity_dbz": grp["max_reflectivity_dbz"].max(),
                "mean_mean_reflectivity_dbz": grp["mean_reflectivity_dbz"].mean(),
                "mean_rhohv_mean": grp["rhohv_mean"].mean() if "rhohv_mean" in grp.columns else np.nan,
                "mean_compactness_km": grp["compactness_km"].mean() if "compactness_km" in grp.columns else np.nan,
                "mean_range_km": grp["range_km"].mean(),
                "mean_velocity_mean_ms": grp["velocity_mean_ms"].mean() if "velocity_mean_ms" in grp.columns else np.nan,
                "mean_spectrum_width_mean_ms": grp["spectrum_width_mean_ms"].mean() if "spectrum_width_mean_ms" in grp.columns else np.nan,
            })

        agg_df = pd.DataFrame(agg_features)
        merged_tracklets_df = pd.merge(merged_tracklets_df, agg_df, on="tracklet_id", how="left")
        merged_tracklets_df.to_parquet(out_ml_dir / "tracklet_features.parquet", index=False)
        print(f"Wrote tracklet features: {out_ml_dir / 'tracklet_features.parquet'}")
    else:
        pd.DataFrame().to_parquet(out_ml_dir / "tracklet_features.parquet")
        print("No tracklets found. Wrote empty tracklet features table.")

    # 4. Generate manual_labels_template.csv
    print("Generating manual labels template...")
    template_rows = []

    # Tracklets (both plausible and rejected)
    if not merged_tracklets_df.empty:
        for _, row in merged_tracklets_df.iterrows():
            tid = row["tracklet_id"]
            site = row["radar_site"]
            start_t = row["start_time_utc"]
            end_t = row["end_time_utc"]
            n_pts = int(row["n_points"])
            spag = row.get("spaghetti_score", 0.0)
            q_label = row.get("quality_label", "unknown")
            med_speed = row["median_segment_speed_kmh"]
            vert_mis = row["median_abs_vertical_mismatch_m"]

            key_feats = f"points={n_pts}, duration={row['duration_min']:.1f}m, speed_med={med_speed:.1f}km/h, vert_mismatch_med={vert_mis:.1f}m, spaghetti={spag:.1f}, label={q_label}"
            dash_path = f"docs/discovery/plausible_tracklet_dashboard.html?tracklet={tid}"

            template_rows.append({
                "object_id": tid,
                "object_type": "tracklet",
                "radar_site": site,
                "start_time_utc": start_t,
                "end_time_utc": end_t,
                "n_points_or_gates": n_pts,
                "key_features": key_feats,
                "dashboard_path": dash_path,
                "manual_label": "",
                "reviewer_notes": "",
            })

    # Clusters (balloon_like_cluster_score >= 0.5)
    high_score_clusters = clusters_df[clusters_df["balloon_like_cluster_score"] >= 0.5]
    for _, row in high_score_clusters.iterrows():
        cid = f"{row['radar_site']}_C_{row['cluster_id']}"
        site = row["radar_site"]
        t = str(row["scan_time_utc"])
        n_gates = int(row["n_gates"])
        dbz = row["max_reflectivity_dbz"]
        score = row["balloon_like_cluster_score"]
        rng = row["range_km"]
        corr = row["inside_or_near_grid_corridor"]

        key_feats = f"gates={n_gates}, dbz_max={dbz:.1f}, score={score:.2f}, range={rng:.1f}km, inside_corridor={corr}"
        dash_path = f"docs/discovery/plausible_tracklet_dashboard.html?radar={site}&time={t}"

        template_rows.append({
            "object_id": cid,
            "object_type": "cluster",
            "radar_site": site,
            "start_time_utc": t,
            "end_time_utc": t,
            "n_points_or_gates": n_gates,
            "key_features": key_feats,
            "dashboard_path": dash_path,
            "manual_label": "",
            "reviewer_notes": "",
        })

    template_df = pd.DataFrame(template_rows)
    template_df.to_csv(out_ml_dir / "manual_labels_template.csv", index=False)
    print(f"Wrote labeling template: {out_ml_dir / 'manual_labels_template.csv'} with {len(template_df)} items.")


if __name__ == "__main__":
    main()
