import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path("scripts/download_nexrad_case.py")
spec = importlib.util.spec_from_file_location("download_nexrad_case", SCRIPT_PATH)
assert spec and spec.loader
download_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = download_module
spec.loader.exec_module(download_module)


class FakePaginator:
    def __init__(self, pages):
        self.pages = pages

    def paginate(self, Bucket, Prefix):
        self.bucket = Bucket
        self.prefix = Prefix
        return self.pages


class FakeS3:
    def __init__(self, pages=None):
        self.pages = pages or []
        self.downloads = []

    def get_paginator(self, operation_name):
        assert operation_name == "list_objects_v2"
        return FakePaginator(self.pages)

    def download_file(self, Bucket, Key, Filename):
        self.downloads.append((Bucket, Key, Filename))
        Path(Filename).write_bytes(b"radar")


def test_parse_scan_time_from_kemx_filename():
    scan_time = download_module.parse_scan_time_from_filename("KEMX20260322_191842_V06", "KEMX")

    assert scan_time == datetime(2026, 3, 22, 19, 18, 42, tzinfo=UTC)


def test_time_window_filter_is_inclusive():
    objects = [
        download_module.NexradObject("a/KEMX20260322_184500_V06", 1, _dt("2026-03-22T18:45:00Z")),
        download_module.NexradObject("a/KEMX20260322_190000_V06", 1, _dt("2026-03-22T19:00:00Z")),
        download_module.NexradObject("a/KEMX20260322_230000_V06", 1, _dt("2026-03-22T23:00:00Z")),
        download_module.NexradObject("a/KEMX20260322_230001_V06", 1, _dt("2026-03-22T23:00:01Z")),
    ]

    selected = download_module.filter_objects_by_window(
        objects,
        _dt("2026-03-22T18:45:00Z"),
        _dt("2026-03-22T23:00:00Z"),
    )

    assert [obj.filename for obj in selected] == [
        "KEMX20260322_184500_V06",
        "KEMX20260322_190000_V06",
        "KEMX20260322_230000_V06",
    ]


def test_mdm_files_are_ignored():
    item = {"Key": "2026/03/22/KEMX/KEMX20260322_191842_V06_MDM", "Size": 10}

    assert download_module.object_from_listing(item, "KEMX") is None


def test_mocked_s3_listing_and_download_writes_index(tmp_path):
    config_path = _write_config(tmp_path)
    pages = [
        {
            "Contents": [
                {"Key": "2026/03/22/KEMX/KEMX20260322_184937_V06", "Size": 4},
                {"Key": "2026/03/22/KEMX/KEMX20260322_185642_V06_MDM", "Size": 4},
                {"Key": "2026/03/22/KEMX/KEMX20260322_225453_V06", "Size": 4},
            ]
        }
    ]
    fake_s3 = FakeS3(pages)

    index_path = download_module.run_download(config_path, dry_run=False, s3=fake_s3)
    index = pd.read_csv(index_path)

    assert len(index) == 2
    assert len(fake_s3.downloads) == 2
    assert index["downloaded"].tolist() == [True, True]
    assert "MDM" not in ",".join(index["filename"].tolist())


def _dt(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _write_config(tmp_path: Path) -> Path:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    config_path = case_dir / "config.yaml"
    config_path.write_text(
        """
case_id: fake_case
nexrad:
  bucket: unidata-nexrad-level2
  date_utc: 2026-03-22
  start_time_utc: "2026-03-22T18:45:00Z"
  end_time_utc: "2026-03-22T23:00:00Z"
  primary_radar_site: KEMX
""".lstrip(),
        encoding="utf-8",
    )
    return config_path
