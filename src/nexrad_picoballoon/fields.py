"""Radar field normalization helpers."""

from __future__ import annotations

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "reflectivity": ("reflectivity", "DBZH", "ZH", "REF", "REFL"),
    "velocity": ("velocity", "VEL", "VR", "mean_radial_velocity"),
    "spectrum_width": ("spectrum_width", "WIDTH", "SW", "WRAD"),
    "rhohv": ("rhohv", "RHOHV", "cross_correlation_ratio", "CC"),
    "zdr": ("zdr", "ZDR", "differential_reflectivity"),
    "phidp": ("phidp", "PHIDP", "differential_phase"),
}


def canonical_field_name(name: str) -> str:
    """Return the canonical project field name for a radar field."""

    normalized = name.strip().lower()
    for canonical, aliases in FIELD_ALIASES.items():
        if normalized == canonical.lower() or normalized in {alias.lower() for alias in aliases}:
            return canonical
    return normalized


def normalize_field_mapping(fields: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Map canonical field names to the original field names available in a dataset."""

    mapping: dict[str, str] = {}
    for field in fields:
        canonical = canonical_field_name(field)
        mapping.setdefault(canonical, field)
    return mapping
