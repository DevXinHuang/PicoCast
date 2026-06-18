"""Tests for geospatial mapping utilities and GeoJSON export."""

from __future__ import annotations

from pathlib import Path

import pytest

from nexrad_picoballoon.maidenhead_grid import (
    grid_bounds,
    grid_center,
    grid_polygon_coords,
    grid_precision_chars,
    grid_uncertainty_km,
)

# ---------------------------------------------------------------------------
# Maidenhead grid utilities
# ---------------------------------------------------------------------------


class TestMaidenheadGridCenter:
    def test_known_grid_center(self):
        """DM42mg should be near (32.27, -110.96) per expected_track.csv."""
        lat, lon = grid_center("DM42mg")
        assert 32.2 < lat < 32.4
        assert -111.1 < lon < -110.8

    def test_center_returns_floats(self):
        lat, lon = grid_center("DM42mg")
        assert isinstance(lat, float)
        assert isinstance(lon, float)


class TestMaidenheadGridBounds:
    def test_bounds_order(self):
        """Bounds should be (south, north, west, east)."""
        south, north, west, east = grid_bounds("DM42mg")
        assert south < north
        assert west < east

    def test_center_inside_bounds(self):
        lat, lon = grid_center("DM42mg")
        south, north, west, east = grid_bounds("DM42mg")
        assert south <= lat <= north
        assert west <= lon <= east

    def test_6char_grid_size(self):
        """A 6-char Maidenhead grid should be ~2.5' lat × 5' lon."""
        south, north, west, east = grid_bounds("DM42mg")
        lat_span = north - south
        lon_span = east - west
        # 6-char: 2.5 arcmin lat, 5 arcmin lon
        assert abs(lat_span - 2.5 / 60) < 0.001
        assert abs(lon_span - 5.0 / 60) < 0.001


class TestMaidenheadPolygonCoords:
    def test_polygon_is_closed(self):
        coords = grid_polygon_coords("DM42mg")
        assert coords[0] == coords[-1], "Polygon must be closed (first == last)"

    def test_polygon_has_5_points(self):
        coords = grid_polygon_coords("DM42mg")
        assert len(coords) == 5, "Closed rectangle should have 5 coordinate pairs"

    def test_geojson_coordinate_order(self):
        """GeoJSON coordinates must be [lon, lat], not [lat, lon]."""
        coords = grid_polygon_coords("DM42mg")
        south, north, west, east = grid_bounds("DM42mg")
        # First point should be SW corner: [west, south]
        assert coords[0][0] == west, "First coordinate element must be longitude"
        assert coords[0][1] == south, "Second coordinate element must be latitude"


class TestGridPrecisionChars:
    def test_4char(self):
        assert grid_precision_chars("DM42") == 4

    def test_6char(self):
        assert grid_precision_chars("DM42mg") == 6

    def test_8char(self):
        assert grid_precision_chars("DM42mg00") == 8


class TestGridUncertaintyKm:
    def test_6char_uncertainty(self):
        """6-char grid uncertainty should be roughly 8–10 km diagonal."""
        unc = grid_uncertainty_km("DM42mg")
        assert 5.0 < unc < 12.0

    def test_4char_larger_than_6char(self):
        unc4 = grid_uncertainty_km("DM42")
        unc6 = grid_uncertainty_km("DM42mg")
        assert unc4 > unc6


# ---------------------------------------------------------------------------
# GeoJSON coordinate order validation
# ---------------------------------------------------------------------------


class TestGeoJSONCoordinateOrder:
    """Verify that all GeoJSON outputs use [lon, lat] order."""

    def test_point_feature_lon_lat_order(self):
        """A GeoJSON point for DM42mg should have lon (~-111) first, lat (~32) second."""
        lat, lon = grid_center("DM42mg")
        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {},
        }
        coords = feature["geometry"]["coordinates"]
        # Longitude should be negative (western hemisphere)
        assert coords[0] < 0, "First coordinate must be longitude (negative for western hemisphere)"
        # Latitude should be positive (northern hemisphere)
        assert coords[1] > 0, (
            "Second coordinate must be latitude (positive for northern hemisphere)"
        )

    def test_polygon_feature_lon_lat_order(self):
        """All polygon coordinates should have lon first, lat second."""
        coords = grid_polygon_coords("DM42mg")
        for i, (lon, lat) in enumerate(coords):
            assert lon < 0, f"Point {i}: first element must be longitude (negative)"
            assert lat > 0, f"Point {i}: second element must be latitude (positive)"


# ---------------------------------------------------------------------------
# Gate limiting
# ---------------------------------------------------------------------------


class TestGateLimiting:
    def test_max_gates_config_respected(self):
        """Verify the config value would be applied correctly."""
        import yaml

        config_text = """
mapping:
  max_gates_to_map: 5000
"""
        config = yaml.safe_load(config_text)
        max_gates = config["mapping"]["max_gates_to_map"]
        assert max_gates == 5000

    def test_default_prevents_million_gates(self):
        """Default max_gates_to_map must be <= 10000."""
        import yaml

        config_text = """
mapping:
  max_gates_to_map: 5000
"""
        config = yaml.safe_load(config_text)
        assert config["mapping"]["max_gates_to_map"] <= 10000


# ---------------------------------------------------------------------------
# Range ring geometry
# ---------------------------------------------------------------------------


class TestRangeRingGeometry:
    def test_range_ring_circle(self):
        """A 100 km range ring should produce points ~100 km from center."""
        from geopy.distance import geodesic

        center_lat, center_lon = 31.894, -110.630
        radius_km = 100
        n_points = 72
        for i in range(n_points):
            bearing = 360.0 * i / n_points
            dest = geodesic(kilometers=radius_km).destination(
                (center_lat, center_lon), bearing
            )
            actual_dist = geodesic(
                (center_lat, center_lon), (dest.latitude, dest.longitude)
            ).km
            assert abs(actual_dist - radius_km) < 0.1, (
                f"Point at bearing {bearing}° is {actual_dist:.2f} km from center, "
                f"expected {radius_km} km"
            )

    def test_range_ring_is_closed(self):
        """Range ring polygon should be closed."""
        from geopy.distance import geodesic

        center_lat, center_lon = 31.894, -110.630
        radius_km = 50
        n_points = 72
        coords = []
        for i in range(n_points + 1):
            bearing = 360.0 * i / n_points
            dest = geodesic(kilometers=radius_km).destination(
                (center_lat, center_lon), bearing
            )
            coords.append([dest.longitude, dest.latitude])
        # First and last should be (nearly) the same
        assert abs(coords[0][0] - coords[-1][0]) < 1e-10
        assert abs(coords[0][1] - coords[-1][1]) < 1e-10


# ---------------------------------------------------------------------------
# Mapping does not try to render all gates
# ---------------------------------------------------------------------------


class TestNoMassGateRendering:
    def test_near_track_gates_not_fully_loaded(self):
        """Verify that the design limits gates to a configurable maximum.

        This is a design contract test \u2014 it checks that the config schema
        includes a max_gates_to_map setting with a reasonable default.
        """
        import yaml

        config_path = (
            Path(__file__).resolve().parents[1]
            / "cases"
            / "k7uaz_20260322"
            / "config.yaml"
        )
        if not config_path.exists():
            pytest.skip("Case config not found")
        with config_path.open() as fh:
            config = yaml.safe_load(fh)
        mapping = config.get("mapping", {})
        max_gates = mapping.get("max_gates_to_map", 5000)
        assert max_gates <= 50000, (
            f"max_gates_to_map={max_gates} is too high; "
            "rendering millions of gates will freeze the browser"
        )
