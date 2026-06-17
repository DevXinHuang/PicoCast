from nexrad_picoballoon.fields import canonical_field_name, normalize_field_mapping


def test_field_aliases_normalize_to_canonical_names():
    assert canonical_field_name("DBZH") == "reflectivity"
    assert canonical_field_name("VEL") == "velocity"
    assert canonical_field_name("RHOHV") == "rhohv"


def test_field_mapping_preserves_original_names():
    mapping = normalize_field_mapping(("DBZH", "VEL", "RHOHV"))
    assert mapping["reflectivity"] == "DBZH"
    assert mapping["velocity"] == "VEL"
    assert mapping["rhohv"] == "RHOHV"
