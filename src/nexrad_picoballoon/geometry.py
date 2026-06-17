"""Radar gate geometry helpers."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0


def gate_lat_lon_alt(
    radar_latitude: float,
    radar_longitude: float,
    radar_altitude_m: float,
    azimuth_deg: np.ndarray,
    range_m: np.ndarray,
    elevation_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate gate latitude, longitude, and altitude.

    This is a lightweight spherical-earth approximation for candidate review,
    not a replacement for full radar georeferencing in Py-ART/wradlib.
    """

    azimuth_rad = np.deg2rad(azimuth_deg)[:, None]
    range_grid = range_m[None, :]
    elevation = np.asarray(elevation_deg)
    if elevation.ndim == 0:
        elevation_rad = np.deg2rad(float(elevation))
    else:
        elevation_rad = np.deg2rad(elevation)[:, None]

    ground_range = range_grid * np.cos(elevation_rad)
    north_m = ground_range * np.cos(azimuth_rad)
    east_m = ground_range * np.sin(azimuth_rad)

    lat = radar_latitude + np.rad2deg(north_m / EARTH_RADIUS_M)
    longitude_scale = EARTH_RADIUS_M * np.cos(np.deg2rad(radar_latitude))
    lon = radar_longitude + np.rad2deg(east_m / longitude_scale)
    alt = radar_altitude_m + range_grid * np.sin(elevation_rad)
    return lat, lon, alt
