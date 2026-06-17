import importlib.util
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path("scripts/match_track_to_radar_scans.py")
spec = importlib.util.spec_from_file_location("match_track_to_radar_scans", SCRIPT_PATH)
assert spec and spec.loader
match_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(match_module)


def test_scan_track_matches_created_from_fake_inputs(tmp_path):
    case_dir = tmp_path / "case"
    index_dir = case_dir / "nexrad" / "KEMX" / "index"
    index_dir.mkdir(parents=True)
    config_path = case_dir / "config.yaml"
    config_path.write_text(
        """
case_id: fake_case
nexrad:
  primary_radar_site: KEMX
""".lstrip(),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "time_utc": "2026-03-22T19:00:00Z",
                "lat_deg": 32.0,
                "lon_deg": -111.0,
                "alt_km": 2.0,
                "alt_m": 2000.0,
            },
            {
                "time_utc": "2026-03-22T20:00:00Z",
                "lat_deg": 33.0,
                "lon_deg": -110.0,
                "alt_km": 4.0,
                "alt_m": 4000.0,
            },
        ]
    ).to_csv(case_dir / "expected_track.csv", index=False)
    pd.DataFrame(
        [
            {
                "scan_time_utc": "2026-03-22T19:30:00Z",
                "filename": "KEMX20260322_193000_V06",
                "local_path": "raw/KEMX20260322_193000_V06",
                "download_error": "",
            },
            {
                "scan_time_utc": "2026-03-22T20:30:00Z",
                "filename": "KEMX20260322_203000_V06",
                "local_path": "raw/KEMX20260322_203000_V06",
                "download_error": "",
            },
        ]
    ).to_csv(index_dir / "nexrad_files.csv", index=False)

    output_path = match_module.write_scan_track_matches(config_path)
    matches = pd.read_csv(output_path)

    assert len(matches) == 2
    assert matches.iloc[0]["expected_lat_deg"] == 32.5
    assert matches.iloc[0]["expected_lon_deg"] == -110.5
    assert matches.iloc[0]["expected_alt_m"] == 3000.0
    assert matches.iloc[0]["time_offset_min"] == 30.0
    assert matches.iloc[1]["time_offset_min"] == 30.0
    assert bool(matches.iloc[0]["inside_track_window"]) is True
    assert bool(matches.iloc[1]["inside_track_window"]) is False
