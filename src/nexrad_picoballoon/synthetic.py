"""Small synthetic radar volumes for tests and demos."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import xarray as xr


def make_synthetic_volume(
    site: str = "KTEST",
    scan_time: datetime | None = None,
    target_range_index: int = 18,
    target_azimuth_index: int = 12,
) -> xr.Dataset:
    """Create a tiny sweep-like xarray dataset with one weak moving target."""

    scan_time = scan_time or datetime(2026, 1, 1, tzinfo=UTC)
    azimuth = np.linspace(0.0, 357.0, 36)
    ranges = np.arange(1_000.0, 31_000.0, 1_000.0)
    shape = (azimuth.size, ranges.size)

    reflectivity = np.full(shape, -12.0)
    velocity = np.zeros(shape)
    spectrum_width = np.full(shape, 0.4)
    rhohv = np.full(shape, 0.97)
    zdr = np.full(shape, 0.1)
    phidp = np.full(shape, 4.0)

    az_slice = slice(max(0, target_azimuth_index - 1), target_azimuth_index + 2)
    range_slice = slice(max(0, target_range_index - 1), target_range_index + 2)
    reflectivity[az_slice, range_slice] = 8.0
    velocity[az_slice, range_slice] = 6.5
    spectrum_width[az_slice, range_slice] = 0.8
    rhohv[az_slice, range_slice] = 0.72
    zdr[az_slice, range_slice] = 1.8
    phidp[az_slice, range_slice] = 12.0

    dataset = xr.Dataset(
        data_vars={
            "reflectivity": (("azimuth", "range"), reflectivity),
            "velocity": (("azimuth", "range"), velocity),
            "spectrum_width": (("azimuth", "range"), spectrum_width),
            "rhohv": (("azimuth", "range"), rhohv),
            "zdr": (("azimuth", "range"), zdr),
            "phidp": (("azimuth", "range"), phidp),
            "elevation": (("azimuth",), np.full(azimuth.size, 0.5)),
        },
        coords={
            "azimuth": azimuth,
            "range": ranges,
        },
        attrs={
            "site": site,
            "scan_time": scan_time.isoformat(),
            "radar_latitude": 35.0,
            "radar_longitude": -97.0,
            "radar_altitude_m": 370.0,
        },
    )
    return dataset
