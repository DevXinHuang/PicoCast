from pathlib import Path

from nexrad_picoballoon.features import extract_gate_features
from nexrad_picoballoon.synthetic import make_synthetic_volume


def test_extract_gate_features_schema_and_geometry_bounds():
    dataset = make_synthetic_volume()
    features = extract_gate_features(dataset, source_path=Path("synthetic.nc"))

    expected = {
        "site",
        "scan_time",
        "azimuth_deg",
        "range_m",
        "latitude",
        "longitude",
        "altitude_m",
        "reflectivity_dbz",
        "velocity_ms",
        "spectrum_width_ms",
        "rhohv",
        "zdr_db",
        "phidp_deg",
        "rhohv_texture",
        "zdr_texture",
        "phidp_texture",
    }
    assert expected.issubset(features.columns)
    assert len(features) == 36 * 30
    assert features["latitude"].between(34.0, 36.0).all()
    assert features["longitude"].between(-98.0, -96.0).all()
    assert features["altitude_m"].gt(0.0).all()
