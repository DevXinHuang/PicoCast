import pandas as pd
import numpy as np
from pathlib import Path
import typer
import yaml
from pyproj import Geod
from typing import List

app = typer.Typer()

def label_wind_consistency(speed_diff, bearing_diff):
    if pd.isna(speed_diff) or pd.isna(bearing_diff):
        return "insufficient_wind_data"
        
    bearing_diff = abs(bearing_diff)
    bearing_diff = min(bearing_diff, 360 - bearing_diff) # shortest angular distance
    speed_diff = abs(speed_diff)
    
    if speed_diff < 15 and bearing_diff < 15:
        return "strong_atmospheric_wind_consistency"
    elif speed_diff < 30 and bearing_diff < 30:
        return "moderate_atmospheric_wind_consistency"
    elif speed_diff < 50 and bearing_diff < 45:
        return "weak_atmospheric_wind_consistency"
    else:
        return "wind_inconsistent"

@app.command()
def main(
    config_path: str,
    primary_sites: bool = False,
    tracklet_ids: List[str] = typer.Option(None, "--tracklet-ids", help="List of tracklet IDs to validate")
):
    """Validate candidate tracklet kinematics against atmospheric wind models."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    case_dir = config_path.parent
    
    pts_path = case_dir / "outputs" / "discovery" / "plausible_tracklet_points.csv"
    if not pts_path.exists():
        print(f"File not found: {pts_path}")
        return
        
    pts = pd.read_csv(pts_path)
    pts["scan_time_utc"] = pd.to_datetime(pts["scan_time_utc"])
    
    if tracklet_ids:
        pts = pts[pts["tracklet_id"].isin(tracklet_ids)]
        
    if pts.empty:
        print("No matching tracklets to validate.")
        return
        
    # Find hrrr points
    hrrr_path = case_dir / "outputs" / "wind_context" / "hrrr_wind_profile_points.csv"
    if not hrrr_path.exists():
        print(f"HRRR wind context not found at {hrrr_path}. Run download_wind_context.py first.")
        return
        
    wind = pd.read_csv(hrrr_path)
    wind = wind[wind["point_type"] == "candidate"]
    
    geod = Geod(ellps="WGS84")
    
    results = []
    
    for tid, group in pts.groupby("tracklet_id"):
        group = group.sort_values("scan_time_utc")
        lats = group["cluster_lat_deg"].values
        lons = group["cluster_lon_deg"].values
        times = group["scan_time_utc"].values
        
        # Calculate kinematics
        speeds = [np.nan]
        bearings = [np.nan]
        
        for i in range(1, len(lats)):
            dt_s = (times[i] - times[i-1]) / np.timedelta64(1, 's')
            if dt_s > 0:
                fwd_az, back_az, dist_m = geod.inv(lons[i-1], lats[i-1], lons[i], lats[i])
                speeds.append((dist_m / dt_s) * 3.6)
                bearings.append((fwd_az + 360) % 360)
            else:
                speeds.append(np.nan)
                bearings.append(np.nan)
                
        group["candidate_speed_kmh"] = speeds
        group["candidate_bearing_deg"] = bearings
        
        # Merge with wind data
        # We can merge on point_id if we created it as tracklet_id + _ + scan_time_utc string
        group["point_id"] = group["tracklet_id"] + "_" + group["scan_time_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        merged = pd.merge(group, wind, on="point_id", how="left")
        
        for _, row in merged.iterrows():
            c_speed = row["candidate_speed_kmh"]
            c_bearing = row["candidate_bearing_deg"]
            w_speed = row.get("wind_speed_kmh", np.nan)
            w_bearing = row.get("wind_to_direction_deg", np.nan)
            
            speed_diff = np.nan
            speed_ratio = np.nan
            bearing_diff = np.nan
            label = "insufficient_wind_data"
            
            if pd.notna(c_speed) and pd.notna(w_speed) and pd.notna(c_bearing) and pd.notna(w_bearing):
                speed_diff = c_speed - w_speed
                speed_ratio = c_speed / w_speed if w_speed > 0 else np.nan
                b_diff = c_bearing - w_bearing
                # Handle circular angle difference
                bearing_diff = (b_diff + 180) % 360 - 180
                
                label = label_wind_consistency(speed_diff, bearing_diff)
                
            results.append({
                "tracklet_id": row["tracklet_id"],
                "scan_time_utc": row["scan_time_utc"],
                "candidate_speed_kmh": c_speed,
                "candidate_bearing_deg": c_bearing,
                "hrrr_wind_speed_kmh": w_speed,
                "hrrr_wind_to_direction_deg": w_bearing,
                "speed_difference_kmh": speed_diff,
                "speed_ratio": speed_ratio,
                "bearing_difference_deg": bearing_diff,
                "wind_consistency_label": label
            })
            
    out_df = pd.DataFrame(results)
    out_path = case_dir / "outputs" / "wind_context" / "wind_validated_tracklets.csv"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_df.to_csv(out_path, index=False)
    
    print(f"Validated {len(out_df)} candidate points.")
    print(f"Results saved to {out_path}")
    
    # Print a quick summary for the tracklets
    print("\nWind Consistency Summary:")
    summary = out_df.groupby(["tracklet_id", "wind_consistency_label"]).size().unstack(fill_value=0)
    print(summary)

if __name__ == "__main__":
    app()
