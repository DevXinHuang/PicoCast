import importlib.util
from pathlib import Path

SCRIPT_PATH = Path("scripts/preview_nexrad_case.py")
spec = importlib.util.spec_from_file_location("preview_nexrad_case", SCRIPT_PATH)
assert spec and spec.loader
preview_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview_module)


def test_preview_output_filename_is_stable():
    assert (
        preview_module.output_filename("KEMX", "2026-03-22T19:18:42Z")
        == "KEMX_20260322_191842_reflectivity_track_overlay.png"
    )
