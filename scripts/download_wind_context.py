import pandas as pd
import numpy as np
from pathlib import Path
import typer
import yaml
from herbie import Herbie
import warnings

# Suppress xarray/cfgrib warnings for cleaner output
warnings.filterwarnings('ignore')

app = typer.Typer()

def compute_wind_direction(u, v):
    """Calculate meteorological 'wind from' and 'wind to' directions in degrees."""
    to_dir = (90 - np.degrees(np.arctan2(v, u))) % 360
    from_dir = (to_dir + 180) % 360
    return from_dir, to_dir

def get_nearest_profile(ds, lat, lon):
    """Find the nearest horizontal grid cell using simple distance."""
    lats = ds.latitude.values
    lons = ds.longitude.values
    # Approximate euclidean distance for quick nearest neighbor search
    dist_sq = (lats - lat)**2 + (lons - lon)**2
    idx_y, idx_x = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    return ds.isel(y=idx_y, x=idx_x)

def interpolate_wind(profile, target_alt_m):
    """Interpolate U and V wind components to target geometric altitude."""
    # Convert to pandas dataframe and sort by geopotential height (gh)
    df = profile.to_dataframe().dropna(subset=['gh', 'u', 'v']).sort_values('gh')
    
    if df.empty:
        return None
        
    gh_array = df['gh'].values
    idx_upper = np.searchsorted(gh_array, target_alt_m)
    
    if idx_upper == 0:
        lower = upper = df.iloc[0]
        method = "extrapolated_below"
    elif idx_upper == len(df):
        lower = upper = df.iloc[-1]
        method = "extrapolated_above"
    else:
        lower = df.iloc[idx_upper - 1]
        upper = df.iloc[idx_upper]
        method = "interpolated"
        
    if lower['gh'] == upper['gh']:
        u = lower['u']
        v = lower['v']
    else:
        frac = (target_alt_m - lower['gh']) / (upper['gh'] - lower['gh'])
        u = float(lower['u'] + frac * (upper['u'] - lower['u']))
        v = float(lower['v'] + frac * (upper['v'] - lower['v']))
        
    speed_ms = np.sqrt(u**2 + v**2)
    from_dir, to_dir = compute_wind_direction(u, v)
    
    return {
        "pressure_level_hpa_lower": float(lower.name) if isinstance(lower.name, (float, int, np.number)) else getattr(lower, 'isobaricInhPa', None),
        "pressure_level_hpa_upper": float(upper.name) if isinstance(upper.name, (float, int, np.number)) else getattr(upper, 'isobaricInhPa', None),
        "hgt_lower_m": float(lower['gh']),
        "hgt_upper_m": float(upper['gh']),
        "u_wind_ms": u,
        "v_wind_ms": v,
        "wind_speed_ms": speed_ms,
        "wind_speed_kmh": speed_ms * 3.6,
        "wind_from_direction_deg": from_dir,
        "wind_to_direction_deg": to_dir,
        "interpolation_method": method
    }

@app.command()
def main(config_path: str, model: str = "hrrr"):
    """Download and extract wind context for candidates and expected track."""
    config_path = Path(config_path)
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    case_dir = config_path.parent
    
    # Load expected track (telemetry)
    track_path = case_dir / "expected_track.csv"
    if track_path.exists():
        track_df = pd.read_csv(track_path)
        track_df["type"] = "telemetry"
        track_df["time"] = pd.to_datetime(track_df["time_utc"])
        track_df["target_alt_m"] = track_df["alt_m"]
        track_df["lat"] = track_df["lat_deg"]
        track_df["lon"] = track_df["lon_deg"]
        track_df["point_id"] = track_df["point_id"]
    else:
        track_df = pd.DataFrame()
        
    # Load plausible tracklet points (candidates)
    pts_path = case_dir / "outputs" / "discovery" / "plausible_tracklet_points.csv"
    if pts_path.exists():
        pts_df = pd.read_csv(pts_path)
        pts_df["type"] = "candidate"
        pts_df["time"] = pd.to_datetime(pts_df["scan_time_utc"])
        pts_df["target_alt_m"] = pts_df["cluster_alt_m"]
        pts_df["lat"] = pts_df["cluster_lat_deg"]
        pts_df["lon"] = pts_df["cluster_lon_deg"]
        pts_df["point_id"] = pts_df["tracklet_id"] + "_" + pts_df["scan_time_utc"]
    else:
        pts_df = pd.DataFrame()
        
    if track_df.empty and pts_df.empty:
        print("No points found to extract wind context for.")
        return
        
    # Combine needed columns
    cols = ["type", "point_id", "time", "lat", "lon", "target_alt_m"]
    combined = pd.concat([
        track_df[cols] if not track_df.empty else pd.DataFrame(columns=cols),
        pts_df[cols] if not pts_df.empty else pd.DataFrame(columns=cols)
    ], ignore_index=True)
    
    # Add rounded hour to fetch
    combined["model_cycle_time"] = combined["time"].dt.round("h")
    unique_cycles = combined["model_cycle_time"].unique()
    
    out_dir = case_dir / "outputs" / "wind_context"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    inventory = []
    
    for cycle in unique_cycles:
        cycle_str = pd.Timestamp(cycle).strftime("%Y-%m-%d %H:%M")
        print(f"Fetching {model.upper()} for cycle: {cycle_str}")
        
        try:
            H = Herbie(cycle_str, model=model, product="prs", fxx=0)
            datasets = H.xarray(":(UGRD|VGRD|HGT):")
            if isinstance(datasets, list):
                # Find the dataset that has the pressure levels we want
                ds = None
                for d in datasets:
                    if 'isobaricInhPa' in d.coords:
                        if ds is None:
                            ds = d
                        else:
                            import xarray as xr
                            ds = xr.merge([ds, d], compat='override')
            else:
                ds = datasets
                
            inventory.append({
                "model": model,
                "cycle_time_utc": cycle_str,
                "status": "success",
                "file": str(H.get_localFilePath())
            })
        except Exception as e:
            print(f"Failed to fetch Herbie data for {cycle_str}: {e}")
            inventory.append({
                "model": model,
                "cycle_time_utc": cycle_str,
                "status": "failed",
                "file": ""
            })
            continue
            
        subset = combined[combined["model_cycle_time"] == cycle]
        
        for _, row in subset.iterrows():
            profile = get_nearest_profile(ds, row["lat"], row["lon"])
            res = interpolate_wind(profile, row["target_alt_m"])
            
            if res:
                res["point_id"] = row["point_id"]
                res["point_type"] = row["type"]
                res["source_model"] = model
                res["model_cycle_time_utc"] = cycle_str
                res["valid_time_utc"] = row["time"].strftime("%Y-%m-%dT%H:%M:%SZ")
                res["lat_deg"] = row["lat"]
                res["lon_deg"] = row["lon"]
                res["target_alt_m"] = row["target_alt_m"]
                res["notes"] = ""
                results.append(res)
                
    if results:
        res_df = pd.DataFrame(results)
        # Reorder columns as requested
        order = [
            "point_id", "point_type", "source_model", "model_cycle_time_utc", "valid_time_utc",
            "lat_deg", "lon_deg", "target_alt_m", "pressure_level_hpa_lower", "pressure_level_hpa_upper",
            "hgt_lower_m", "hgt_upper_m", "u_wind_ms", "v_wind_ms", "wind_speed_ms", "wind_speed_kmh",
            "wind_from_direction_deg", "wind_to_direction_deg", "interpolation_method", "notes"
        ]
        res_df = res_df[[c for c in order if c in res_df.columns]]
        out_file = out_dir / f"{model}_wind_profile_points.csv"
        res_df.to_csv(out_file, index=False)
        print(f"Wrote {len(res_df)} interpolated wind points to {out_file}")
        
    if inventory:
        inv_df = pd.DataFrame(inventory)
        inv_file = out_dir / "wind_context_inventory.csv"
        # Append or write new
        if inv_file.exists():
            inv_df.to_csv(inv_file, mode='a', header=False, index=False)
        else:
            inv_df.to_csv(inv_file, index=False)
        print(f"Updated inventory at {inv_file}")

if __name__ == "__main__":
    app()
