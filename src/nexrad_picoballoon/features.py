"""Per-gate feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import generic_filter

from nexrad_picoballoon.fields import normalize_field_mapping
from nexrad_picoballoon.geometry import gate_lat_lon_alt


def local_texture(values: np.ndarray, size: int = 3) -> np.ndarray:
    """Compute local standard deviation with NaN-tolerant windows."""

    return generic_filter(values, np.nanstd, size=size, mode="nearest")


def extract_gate_features(dataset: xr.Dataset, source_path: Path | None = None) -> pd.DataFrame:
    """Extract a flat per-gate feature table from a sweep-like dataset."""

    mapping = normalize_field_mapping(tuple(dataset.data_vars))
    required = ["reflectivity", "velocity", "spectrum_width", "rhohv", "zdr", "phidp"]
    missing = [name for name in required if name not in mapping]
    if missing:
        msg = f"Dataset is missing required fields: {', '.join(missing)}"
        raise ValueError(msg)

    azimuth = dataset["azimuth"].to_numpy()
    ranges = dataset["range"].to_numpy()
    elevation = dataset.get("elevation", xr.DataArray(np.full(azimuth.shape, 0.5))).to_numpy()

    arrays = {name: dataset[mapping[name]].to_numpy() for name in required}
    lat, lon, alt = gate_lat_lon_alt(
        float(dataset.attrs.get("radar_latitude", 0.0)),
        float(dataset.attrs.get("radar_longitude", 0.0)),
        float(dataset.attrs.get("radar_altitude_m", 0.0)),
        azimuth,
        ranges,
        elevation,
    )

    az_grid, range_grid = np.meshgrid(azimuth, ranges, indexing="ij")
    scan_time = str(dataset.attrs.get("scan_time", ""))
    site = str(dataset.attrs.get("site", "UNKNOWN"))

    frame = pd.DataFrame(
        {
            "site": site,
            "scan_time": scan_time,
            "source_path": str(source_path or ""),
            "azimuth_deg": az_grid.ravel(),
            "range_m": range_grid.ravel(),
            "latitude": lat.ravel(),
            "longitude": lon.ravel(),
            "altitude_m": alt.ravel(),
            "reflectivity_dbz": arrays["reflectivity"].ravel(),
            "velocity_ms": arrays["velocity"].ravel(),
            "spectrum_width_ms": arrays["spectrum_width"].ravel(),
            "rhohv": arrays["rhohv"].ravel(),
            "zdr_db": arrays["zdr"].ravel(),
            "phidp_deg": arrays["phidp"].ravel(),
            "rhohv_texture": local_texture(arrays["rhohv"]).ravel(),
            "zdr_texture": local_texture(arrays["zdr"]).ravel(),
            "phidp_texture": local_texture(arrays["phidp"]).ravel(),
        }
    )
    return frame


def write_feature_table(features: pd.DataFrame, out_path: Path) -> Path:
    """Write features as Parquet."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path, index=False)
    return out_path
