#!/usr/bin/env python
# ruff: noqa: E501, E402
"""Phase 2: Active-learning selection of examples for labeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import (
    load_config,
)


def main():
    parser = argparse.ArgumentParser(description="Select examples for labeling.")
    parser.add_argument("config", type=Path, help="Path to config.yaml")

    args = parser.parse_args()
    load_config(args.config)
    case_dir = args.config.parent

    ml_dir = case_dir / "outputs" / "ml"
    template_path = ml_dir / "manual_labels_template.csv"
    tracklet_features_path = ml_dir / "tracklet_features.parquet"

    if not template_path.exists() or not tracklet_features_path.exists():
        raise FileNotFoundError("ML feature tables not found. Run build_ml_feature_table.py first.")

    template_df = pd.read_csv(template_path)
    tracklet_feat_df = pd.read_parquet(tracklet_features_path)

    if template_df.empty:
        print("Manual labels template is empty. Writing empty labeling queue.")
        pd.DataFrame(columns=template_df.columns).to_csv(ml_dir / "labeling_queue.csv", index=False)
        return

    # Split template into tracklets and clusters
    template_tracklets = template_df[template_df["object_type"] == "tracklet"].copy()
    template_clusters = template_df[template_df["object_type"] == "cluster"].copy()

    selected_ids = []

    # 1. Selection logic for Tracklets (from tracklet_feat_df)
    if not tracklet_feat_df.empty:
        # A. Top Scoring Candidates
        top_scoring = tracklet_feat_df.sort_values("tracklet_score", ascending=False).head(3)["tracklet_id"].tolist()
        selected_ids.extend(top_scoring)

        # B. Obvious Bad Spaghetti Paths
        bad_spaghetti = tracklet_feat_df.sort_values("spaghetti_score", ascending=False).head(2)["tracklet_id"].tolist()
        selected_ids.extend(bad_spaghetti)

        # C. Borderline Candidates
        # - Smoothness near 0.4
        tracklet_feat_df["smoothness_borderline_dist"] = (tracklet_feat_df["path_smoothness_score"] - 0.4).abs()
        # - Vertical mismatch near 750
        tracklet_feat_df["alt_borderline_dist"] = (tracklet_feat_df["median_abs_vertical_mismatch_m"] - 750.0).abs()
        # - Speed near 100 or 5
        tracklet_feat_df["speed_borderline_dist"] = (tracklet_feat_df["median_segment_speed_kmh"] - 100.0).abs()

        border_smooth = tracklet_feat_df.sort_values("smoothness_borderline_dist").head(2)["tracklet_id"].tolist()
        border_alt = tracklet_feat_df.sort_values("alt_borderline_dist").head(2)["tracklet_id"].tolist()
        border_speed = tracklet_feat_df.sort_values("speed_borderline_dist").head(2)["tracklet_id"].tolist()

        selected_ids.extend(border_smooth + border_alt + border_speed)

        # D. Duplicate-looking / Overlapping tracklets (associated tracklets)
        associated = tracklet_feat_df[tracklet_feat_df["is_associated"]].head(2)["tracklet_id"].tolist()
        selected_ids.extend(associated)

        # E. Radar Site Diversity
        # Ensure at least one tracklet from each site is selected if available
        all_sites = tracklet_feat_df["radar_site"].unique()
        for site in all_sites:
            site_tracklets = tracklet_feat_df[tracklet_feat_df["radar_site"] == site]
            already_selected_for_site = [tid for tid in selected_ids if tid in site_tracklets["tracklet_id"].values]
            if not already_selected_for_site:
                # Add the top scoring tracklet from this site
                fallback = site_tracklets.sort_values("tracklet_score", ascending=False).head(1)["tracklet_id"].tolist()
                selected_ids.extend(fallback)

    # Filter selected tracklets from template, removing duplicates but keeping order
    seen = set()
    ordered_selected_tracklets = []
    for tid in selected_ids:
        if tid not in seen:
            seen.add(tid)
            ordered_selected_tracklets.append(tid)

    selected_tracklets_df = template_tracklets[template_tracklets["object_id"].isin(ordered_selected_tracklets)].copy()
    # Sort selected_tracklets_df in the order of ordered_selected_tracklets
    selected_tracklets_df["sort_order"] = selected_tracklets_df["object_id"].apply(ordered_selected_tracklets.index)
    selected_tracklets_df = selected_tracklets_df.sort_values("sort_order").drop(columns=["sort_order"])

    # 2. Selection logic for Clusters
    selected_clusters_df = pd.DataFrame()
    if not template_clusters.empty:
        # Extract features from key_features to rank
        scores = []
        for _, row in template_clusters.iterrows():
            # Parse balloon_like_cluster_score from string
            parts = row["key_features"].split(", ")
            score_part = [p for p in parts if p.startswith("score=")]
            score = float(score_part[0].split("=")[1]) if score_part else 0.0
            scores.append(score)
        template_clusters["score"] = scores

        # A. Top Scoring Cluster
        top_cluster = template_clusters.sort_values("score", ascending=False).head(2)
        # B. Borderline Cluster (score near 0.5)
        template_clusters["border_dist"] = (template_clusters["score"] - 0.5).abs()
        border_cluster = template_clusters.sort_values("border_dist").head(2)

        selected_clusters_df = pd.concat([top_cluster, border_cluster]).drop_duplicates(subset=["object_id"])
        selected_clusters_df = selected_clusters_df.drop(columns=["score", "border_dist"], errors="ignore")

    # Combine tracklets and clusters into the queue
    queue_df = pd.concat([selected_tracklets_df, selected_clusters_df], ignore_index=True)
    queue_df.to_csv(ml_dir / "labeling_queue.csv", index=False)

    print(f"Wrote labeling queue: {ml_dir / 'labeling_queue.csv'} with {len(queue_df)} items.")


if __name__ == "__main__":
    main()
