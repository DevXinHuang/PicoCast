"""Maidenhead grid-square utilities.

Wraps the ``maidenhead`` package to provide grid-center, bounds, and
GeoJSON-ready polygon coordinates for Maidenhead locator strings.
"""

from __future__ import annotations

import math

import maidenhead


def grid_center(grid: str) -> tuple[float, float]:
    """Return the (lat, lon) center of a Maidenhead grid square."""

    _sw, _ne, center = maidenhead.to_location_rect(grid)
    return (center[0], center[1])


def grid_bounds(grid: str) -> tuple[float, float, float, float]:
    """Return (south, north, west, east) bounds of a Maidenhead grid square."""

    (south, west), (north, east), _center = maidenhead.to_location_rect(grid)
    return (south, north, west, east)


def grid_polygon_coords(grid: str) -> list[list[float]]:
    """Return a closed polygon ring in GeoJSON ``[lon, lat]`` order.

    The ring traces SW → SE → NE → NW → SW.
    """

    south, north, west, east = grid_bounds(grid)
    return [
        [west, south],
        [east, south],
        [east, north],
        [west, north],
        [west, south],
    ]


def grid_precision_chars(grid: str) -> int:
    """Return the number of characters in a Maidenhead locator."""

    return len(grid)


def grid_uncertainty_km(grid: str) -> float:
    """Approximate diagonal uncertainty of the grid square in km.

    Uses a spherical-earth approximation at the grid-square center latitude.
    """

    south, north, west, east = grid_bounds(grid)
    center_lat = (south + north) / 2.0
    lat_span_km = (north - south) * 111.32
    lon_span_km = (east - west) * 111.32 * math.cos(math.radians(center_lat))
    return math.sqrt(lat_span_km**2 + lon_span_km**2)
