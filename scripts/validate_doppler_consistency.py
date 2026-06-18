#!/usr/bin/env python3
"""Validate Doppler radial velocity and radar moments for tracklet candidates."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from candidate_utils import load_config, local_xy_km

def main():
    parser = argparse.ArgumentParser(description="Validate Doppler consistency.")
    parser.add_argument("config_path", type=Path)
    parser.add_argument("--primary-sites", action="store_true")
    parser.add_argument("--tracklet-ids", nargs="+", default=[])
    parser.add_argument("--top-n", type=int)
    args = parser.parse_args()

    config = load_config(args.config_path)
    discovery_dir = args.config_path.parent / "outputs" / "discovery"
    
    tracklets_csv = discovery_dir / "plausible_tracklets.csv"
    points_csv = discovery_dir / "plausible_tracklet_points.csv"
    
    if not tracklets_csv.exists() or not points_csv.exists():
        print("Discovery outputs not found.")
        return

    tracklets = pd.read_csv(tracklets_csv)
    points = pd.read_csv(points_csv)

    if args.primary_sites:
        primary_sites = config.get("discovery", {}).get("radar_sites_primary", [])
        if not primary_sites and "primary_radar_site" in config:
            primary_sites = [config["primary_radar_site"]]
        tracklets = tracklets[tracklets["radar_site"].isin(primary_sites)]

    if args.tracklet_ids:
        tracklets = tracklets[tracklets["tracklet_id"].isin(args.tracklet_ids)]
        
    if args.top_n:
        tracklets = tracklets.head(args.top_n)

    if len(tracklets) == 0:
        print("No tracklets match the criteria.")
        return

    # Extract radar site metadata
    radar_metadata = config.get("radar_sites", {})

    all_enriched_points = []
    tracklet_summaries = []

    for _, t in tracklets.iterrows():
        tid = t["tracklet_id"]
        site = t["radar_site"]
        
        t_points = points[points["tracklet_id"] == tid].copy()
        if len(t_points) == 0:
            continue
            
        t_points = t_points.sort_values("scan_time_utc").reset_index(drop=True)
        
        if site not in radar_metadata:
            print(f"Warning: No metadata for radar {site}")
            continue
            
        site_lat = radar_metadata[site]["lat"]
        site_lon = radar_metadata[site]["lon"]
        site_alt = radar_metadata[site]["alt_m"]

        # Calculate x, y, z in meters
        x_km, y_km = local_xy_km(t_points["cluster_lat_deg"], t_points["cluster_lon_deg"], site_lat, site_lon)
        x_m = x_km * 1000.0
        y_m = y_km * 1000.0
        z_m = t_points["cluster_alt_m"].to_numpy() - site_alt

        # Calculate time differences in seconds
        scan_times = pd.to_datetime(t_points["scan_time_utc"])
        dt_forward = scan_times.diff(-1).dt.total_seconds().abs().to_numpy()
        
        vx = np.zeros(len(t_points))
        vy = np.zeros(len(t_points))
        vz = np.zeros(len(t_points))
        
        if len(t_points) > 1:
            dx = np.diff(x_m)
            dy = np.diff(y_m)
            dz = np.diff(z_m)
            dt = dt_forward[:-1]
            
            # Avoid division by zero if dt is 0
            dt = np.where(dt == 0, 1.0, dt)
            
            vx[:-1] = dx / dt
            vy[:-1] = dy / dt
            vz[:-1] = dz / dt
            
            # Backward diff for the last point
            dt_last = (scan_times.iloc[-1] - scan_times.iloc[-2]).total_seconds()
            if dt_last == 0: dt_last = 1.0
            vx[-1] = dx[-1] / dt_last
            vy[-1] = dy[-1] / dt_last
            vz[-1] = dz[-1] / dt_last

        r_m = np.sqrt(x_m**2 + y_m**2 + z_m**2)
        # Avoid division by zero
        r_m = np.where(r_m == 0, 1.0, r_m)
        
        ux = x_m / r_m
        uy = y_m / r_m
        uz = z_m / r_m

        expected_rv = vx * ux + vy * uy + vz * uz
        t_points["expected_radial_velocity_ms"] = expected_rv
        
        if "velocity_mean_ms" in t_points.columns:
            t_points["observed_radial_velocity_ms"] = t_points["velocity_mean_ms"]
        else:
            t_points["observed_radial_velocity_ms"] = np.nan

        t_points["radial_velocity_residual_ms"] = t_points["observed_radial_velocity_ms"] - t_points["expected_radial_velocity_ms"]
        t_points["abs_radial_velocity_residual_ms"] = t_points["radial_velocity_residual_ms"].abs()
        
        # Additional required columns
        t_points["doppler_velocity_available"] = t_points["observed_radial_velocity_ms"].notna()
        t_points["spectrum_width_available"] = t_points["spectrum_width_mean_ms"].notna() if "spectrum_width_mean_ms" in t_points.columns else False
        t_points["rhohv_available"] = t_points["rhohv_mean"].notna() if "rhohv_mean" in t_points.columns else False
        
        t_points["residual_signed_normal"] = t_points["observed_radial_velocity_ms"] - t_points["expected_radial_velocity_ms"]
        t_points["residual_signed_flipped"] = t_points["observed_radial_velocity_ms"] + t_points["expected_radial_velocity_ms"]

        def get_label(row):
            if not row["doppler_velocity_available"]:
                return "missing_doppler"
            res = row["abs_radial_velocity_residual_ms"]
            width = row.get("spectrum_width_mean_ms", np.nan)
            rho = row.get("rhohv_mean", np.nan)
            
            consistent = True
            if pd.notna(res) and res > 5.0:
                consistent = False
            # width and rho are context, but if they are very bad we might flag it. 
            # Per user review, they shouldn't immediately kill it, but we follow previous logic for labeling.
            # User noted KIWA SW is 4.33 which is above 4. Let's keep strict for labeling.
            if pd.notna(width) and width > 4.0:
                consistent = False
            if pd.notna(rho) and rho < 0.9:
                consistent = False
                
            return "doppler_consistent" if consistent else "doppler_inconsistent"
            
        t_points["doppler_consistency_label"] = t_points.apply(get_label, axis=1)
        
        def get_notes(row):
            notes = []
            if not row["doppler_velocity_available"]:
                notes.append("No Doppler")
            else:
                if pd.notna(row["abs_radial_velocity_residual_ms"]) and row["abs_radial_velocity_residual_ms"] > 5.0:
                    notes.append("High residual")
            return "; ".join(notes)
            
        t_points["doppler_notes"] = t_points.apply(get_notes, axis=1)

        # Tracklet Summary calculations
        valid_rv_pts = t_points[t_points["doppler_velocity_available"]]
        n_pts = len(t_points)
        n_valid = len(valid_rv_pts)
        
        med_obs = valid_rv_pts["observed_radial_velocity_ms"].median() if n_valid > 0 else np.nan
        med_exp = valid_rv_pts["expected_radial_velocity_ms"].median() if n_valid > 0 else np.nan
        med_abs_res = valid_rv_pts["abs_radial_velocity_residual_ms"].median() if n_valid > 0 else np.nan
        max_abs_res = valid_rv_pts["abs_radial_velocity_residual_ms"].max() if n_valid > 0 else np.nan
        med_sw = valid_rv_pts["spectrum_width_mean_ms"].median() if n_valid > 0 and "spectrum_width_mean_ms" in valid_rv_pts.columns else np.nan
        med_rho = valid_rv_pts["rhohv_mean"].median() if n_valid > 0 and "rhohv_mean" in valid_rv_pts.columns else np.nan
        
        sign_convention = "unknown"
        if n_valid > 0:
            med_abs_norm = valid_rv_pts["residual_signed_normal"].abs().median()
            med_abs_flip = valid_rv_pts["residual_signed_flipped"].abs().median()
            
            if med_abs_flip < med_abs_norm - 2.0 and med_abs_flip < 5.0:
                sign_convention = "flipped_sign_better"
            elif med_abs_norm < 5.0 or med_abs_norm <= med_abs_flip:
                sign_convention = "normal_sign_better"
            else:
                sign_convention = "neither_sign_matches"

        # Overall label for tracklet
        if n_valid == 0:
            trk_label = "missing_doppler"
        elif med_abs_res > 5.0:
            trk_label = "doppler_inconsistent"
        else:
            trk_label = "doppler_consistent"

        summary = {
            "tracklet_id": tid,
            "radar_site": site,
            "n_points": n_pts,
            "n_valid_doppler_points": n_valid,
            "fraction_valid_doppler": n_valid / n_pts if n_pts > 0 else 0,
            "median_observed_radial_velocity_ms": med_obs,
            "median_expected_radial_velocity_ms": med_exp,
            "median_abs_radial_velocity_residual_ms": med_abs_res,
            "max_abs_radial_velocity_residual_ms": max_abs_res,
            "median_spectrum_width_ms": med_sw,
            "median_rhohv": med_rho,
            "doppler_sign_convention": sign_convention,
            "doppler_consistency_label": trk_label,
            "doppler_notes": "Needs visual inspection"
        }
        tracklet_summaries.append(summary)
        
        all_enriched_points.append(t_points)

    if not all_enriched_points:
        print("No valid points generated.")
        return

    result_df = pd.concat(all_enriched_points, ignore_index=True)
    summary_df = pd.DataFrame(tracklet_summaries)
    
    out_dir = discovery_dir / "doppler_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    result_csv = out_dir / "doppler_validated_points.csv"
    result_df.to_csv(result_csv, index=False)
    print(f"Saved {result_csv}")

    summary_csv = out_dir / "doppler_tracklet_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv}")

    # Generate plots
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Expected vs Observed Radial Velocity
    valid_rv = result_df.dropna(subset=["expected_radial_velocity_ms", "observed_radial_velocity_ms"])
    if not valid_rv.empty:
        plt.figure(figsize=(8, 6))
        
        for tid_scatter in valid_rv["tracklet_id"].unique():
            subset = valid_rv[valid_rv["tracklet_id"] == tid_scatter]
            plt.scatter(subset["expected_radial_velocity_ms"], subset["observed_radial_velocity_ms"], label=tid_scatter)
        
        min_val = min(valid_rv["expected_radial_velocity_ms"].min(), valid_rv["observed_radial_velocity_ms"].min())
        max_val = max(valid_rv["expected_radial_velocity_ms"].max(), valid_rv["observed_radial_velocity_ms"].max())
        plt.plot([min_val, max_val], [min_val, max_val], "k--", label="Ideal (Residual = 0)")
        
        plt.title("Expected vs. Observed Radial Velocity")
        plt.xlabel("Expected Radial Velocity (m/s) [Projected segment motion]")
        plt.ylabel("Observed Radial Velocity (m/s) [NEXRAD Velocity]")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.savefig(fig_dir / "expected_vs_observed_rv.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {fig_dir / 'expected_vs_observed_rv.png'}")

    # 2. Time series plots per tracklet
    for tid in result_df["tracklet_id"].unique():
        t_data = result_df[result_df["tracklet_id"] == tid]
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        times = pd.to_datetime(t_data["scan_time_utc"])
        
        # RV
        axs[0].plot(times, t_data["expected_radial_velocity_ms"], marker='o', label="Expected RV", color="blue")
        if "observed_radial_velocity_ms" in t_data.columns and t_data["observed_radial_velocity_ms"].notna().any():
            axs[0].plot(times, t_data["observed_radial_velocity_ms"], marker='x', label="Observed RV", color="red")
        axs[0].set_ylabel("Radial Velocity (m/s)")
        axs[0].set_title(f"{tid} Doppler Validation")
        axs[0].legend()
        axs[0].grid(True, linestyle=":", alpha=0.6)
        
        # Spectrum Width
        if "spectrum_width_mean_ms" in t_data.columns and t_data["spectrum_width_mean_ms"].notna().any():
            axs[1].plot(times, t_data["spectrum_width_mean_ms"], marker='s', color="orange")
            axs[1].axhline(y=4.0, color='r', linestyle='--', alpha=0.5, label="Threshold (4 m/s)")
            axs[1].legend()
        axs[1].set_ylabel("Spectrum Width (m/s)")
        axs[1].grid(True, linestyle=":", alpha=0.6)
        
        # RhoHV
        if "rhohv_mean" in t_data.columns and t_data["rhohv_mean"].notna().any():
            axs[2].plot(times, t_data["rhohv_mean"], marker='d', color="green")
            axs[2].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label="Threshold (0.9)")
            axs[2].legend()
        axs[2].set_ylabel("RhoHV")
        axs[2].set_xlabel("Time (UTC)")
        axs[2].grid(True, linestyle=":", alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(fig_dir / f"{tid}_doppler_timeseries.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved {fig_dir / f'{tid}_doppler_timeseries.png'}")

if __name__ == "__main__":
    main()
