"""Level II decoding boundary.

The MVP keeps custom binary parsing out of scope. Real NEXRAD Level II files are
opened through optional radar libraries, while synthetic NetCDF fixtures use the
lightweight local path.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import xarray as xr

from nexrad_picoballoon.fields import normalize_field_mapping
from nexrad_picoballoon.ingest import load_synthetic_or_netcdf
from nexrad_picoballoon.schemas import RadarVolume


def open_volume(path: Path) -> xr.Dataset:
    """Open a local radar volume or synthetic fixture as an xarray dataset."""

    if path.suffix.lower() in {".nc", ".cdf"}:
        return load_synthetic_or_netcdf(path)
    return open_level2(path)


def open_level2(path: Path) -> xr.Dataset:
    """Open a real NEXRAD Level II file via optional xradar support."""

    try:
        import xradar as xd  # type: ignore[import-not-found]
    except ImportError as exc:
        msg = (
            "Real Level II decoding requires optional radar dependencies. "
            'Install them with: python -m pip install -e ".[radar]"'
        )
        raise RuntimeError(msg) from exc

    try:
        return xd.io.open_nexradlevel2_datatree(path).to_dataset()
    except AttributeError as exc:
        msg = "Installed xradar version does not expose the expected NEXRAD Level II reader."
        raise RuntimeError(msg) from exc


def describe_volume(path: Path, dataset: xr.Dataset) -> RadarVolume:
    """Create a compact metadata record for a decoded dataset."""

    fields = tuple(normalize_field_mapping(tuple(dataset.data_vars)).keys())
    scan_time_value = dataset.attrs.get("scan_time")
    if isinstance(scan_time_value, str):
        scan_time = datetime.fromisoformat(scan_time_value.replace("Z", "+00:00"))
    else:
        scan_time = datetime.fromtimestamp(path.stat().st_mtime)

    return RadarVolume(
        site=str(dataset.attrs.get("site", "UNKNOWN")),
        scan_time=scan_time,
        source_path=path,
        sweeps=1,
        fields=fields,
    )
