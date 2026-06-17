"""Archive-oriented input handling."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import xarray as xr

from nexrad_picoballoon.synthetic import make_synthetic_volume


def write_synthetic_volume(site: str, start: datetime, out_dir: Path) -> Path:
    """Write a tiny NetCDF volume used by tests and smoke workflows."""

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_synthetic_volume(site=site, scan_time=start)
    stamp = start.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{site}_{stamp}_synthetic.nc"
    dataset.to_netcdf(path)
    return path


def list_input_files(input_dir: Path) -> list[Path]:
    """Return supported local radar or synthetic files from a directory."""

    suffixes = {".nc", ".cdf", ".ar2v", ".gz", ".bz2"}
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in suffixes
    )


def load_synthetic_or_netcdf(path: Path) -> xr.Dataset:
    """Open a local synthetic NetCDF file.

    Real Level II files are decoded by ``decode.open_level2``.
    """

    if path.suffix.lower() not in {".nc", ".cdf"}:
        msg = (
            f"{path} is not a NetCDF synthetic fixture; use decode.open_level2 "
            "for Level II files."
        )
        raise ValueError(msg)
    return xr.open_dataset(path)
