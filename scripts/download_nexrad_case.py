#!/usr/bin/env python
"""Download and index KEMX NEXRAD Level II scans for a PicoCAST case."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config

NEXRAD_INDEX_COLUMNS = [
    "case_id",
    "radar_site",
    "scan_time_utc",
    "s3_bucket",
    "s3_key",
    "s3_uri",
    "local_path",
    "filename",
    "downloaded",
    "file_size_bytes",
    "download_error",
]


class S3Client(Protocol):
    def get_paginator(self, operation_name: str): ...

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None: ...


@dataclass(frozen=True)
class NexradObject:
    """One candidate S3 object with a parsed scan timestamp."""

    key: str
    size: int
    scan_time_utc: datetime

    @property
    def filename(self) -> str:
        return Path(self.key).name


def load_config(config_path: Path) -> dict:
    """Read a PicoCAST case YAML config."""

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping.")
    return config


def parse_utc(value: str | datetime | date) -> datetime:
    """Parse config timestamps into timezone-aware UTC datetimes."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime(value.year, value.month, value.day, tzinfo=UTC)
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def nexrad_prefix(date_utc: str | datetime | date, radar_site: str) -> str:
    """Build the Unidata NEXRAD Level II archive prefix for one radar/date."""

    day = parse_utc(date_utc)
    return f"{day:%Y/%m/%d}/{radar_site}/"


def s3_uri(bucket: str, key: str) -> str:
    """Return an S3 URI for an object."""

    return f"s3://{bucket}/{key}"


def parse_scan_time_from_filename(filename: str, radar_site: str) -> datetime | None:
    """Parse filenames like KEMX20260322_191842_V06 into UTC scan times."""

    pattern = re.compile(rf"{re.escape(radar_site)}(\d{{8}})_(\d{{6}})")
    match = pattern.search(filename)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def is_metadata_file(filename: str) -> bool:
    """Return true for metadata/non-radar objects that should be ignored."""

    return "MDM" in filename.upper()


def object_from_listing(item: dict, radar_site: str) -> NexradObject | None:
    """Convert a raw S3 listing item into a radar object, or ignore it."""

    key = str(item["Key"])
    filename = Path(key).name
    if is_metadata_file(filename):
        return None
    scan_time = parse_scan_time_from_filename(filename, radar_site)
    if scan_time is None:
        return None
    return NexradObject(key=key, size=int(item.get("Size", 0)), scan_time_utc=scan_time)


def filter_objects_by_window(
    objects: list[NexradObject], start_time_utc: datetime, end_time_utc: datetime
) -> list[NexradObject]:
    """Keep objects with inclusive scan times inside the configured window."""

    return sorted(
        [
            obj
            for obj in objects
            if start_time_utc <= obj.scan_time_utc <= end_time_utc
        ],
        key=lambda obj: obj.scan_time_utc,
    )


def unsigned_s3_client() -> S3Client:
    """Create an anonymous S3 client for public NEXRAD archive access."""

    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_nexrad_objects(
    s3: S3Client, bucket: str, prefix: str, radar_site: str
) -> list[NexradObject]:
    """List radar objects under an S3 prefix."""

    paginator = s3.get_paginator("list_objects_v2")
    objects: list[NexradObject] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            obj = object_from_listing(item, radar_site)
            if obj is not None:
                objects.append(obj)
    return sorted(objects, key=lambda obj: obj.scan_time_utc)


def case_nexrad_paths(config_path: Path, radar_site: str) -> tuple[Path, Path, Path]:
    """Return raw/index/log directories for a case radar."""

    case_dir = config_path.parent
    radar_dir = case_dir / "nexrad" / radar_site
    return radar_dir / "raw", radar_dir / "index", radar_dir / "logs"


def download_selected_objects(
    s3: S3Client,
    objects: list[NexradObject],
    *,
    bucket: str,
    raw_dir: Path,
    dry_run: bool,
) -> tuple[list[dict], int, int, int]:
    """Download selected objects and return index rows plus summary counts."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    downloaded_count = 0
    skipped_count = 0
    error_count = 0

    for obj in objects:
        local_path = raw_dir / obj.filename
        already_present = local_path.exists() and local_path.stat().st_size > 0
        downloaded = False
        error = ""

        if dry_run:
            skipped_count += int(already_present)
        elif already_present:
            skipped_count += 1
        else:
            try:
                s3.download_file(bucket, obj.key, str(local_path))
                downloaded = True
                downloaded_count += 1
            except Exception as exc:  # noqa: BLE001 - failures are recorded per file.
                error = f"{type(exc).__name__}: {exc}"
                error_count += 1

        file_size = local_path.stat().st_size if local_path.exists() else 0
        rows.append(
            {
                "scan_time_utc": format_utc(obj.scan_time_utc),
                "s3_bucket": bucket,
                "s3_key": obj.key,
                "s3_uri": s3_uri(bucket, obj.key),
                "local_path": str(local_path),
                "filename": obj.filename,
                "downloaded": downloaded,
                "file_size_bytes": file_size,
                "download_error": error,
            }
        )

    return rows, downloaded_count, skipped_count, error_count


def write_index(index_path: Path, rows: list[dict], *, case_id: str, radar_site: str) -> Path:
    """Write the NEXRAD file index CSV."""

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEXRAD_INDEX_COLUMNS)
        writer.writeheader()
        for row in rows:
            full_row = {"case_id": case_id, "radar_site": radar_site, **row}
            writer.writerow({column: full_row.get(column, "") for column in NEXRAD_INDEX_COLUMNS})
    return index_path


def format_utc(value: datetime) -> str:
    """Format a UTC datetime as ISO with Z suffix."""

    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_failure_log(log_dir: Path, rows: list[dict]) -> Path | None:
    """Write a simple failure log for any per-file download errors."""

    failures = [row for row in rows if row.get("download_error")]
    if not failures:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "download_errors.log"
    with log_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(f"{row['filename']}: {row['download_error']}\n")
    return log_path


def run_download(config_path: Path, *, dry_run: bool = False, s3: S3Client | None = None) -> Path:
    """Run listing, optional download, and index writing for the configured case."""

    config = load_config(config_path)
    nexrad = config["nexrad"]
    case_id = str(config["case_id"])
    bucket = str(nexrad["bucket"])
    radar_site = str(nexrad["primary_radar_site"])
    start_time = parse_utc(str(nexrad["start_time_utc"]))
    end_time = parse_utc(str(nexrad["end_time_utc"]))
    prefix = nexrad_prefix(nexrad["date_utc"], radar_site)
    raw_dir, index_dir, log_dir = case_nexrad_paths(config_path, radar_site)
    index_path = index_dir / "nexrad_files.csv"

    s3 = s3 or unsigned_s3_client()
    available = list_nexrad_objects(s3, bucket, prefix, radar_site)
    selected = filter_objects_by_window(available, start_time, end_time)
    if not selected:
        raise RuntimeError(
            f"No {radar_site} files found in {s3_uri(bucket, prefix)} between "
            f"{format_utc(start_time)} and {format_utc(end_time)}."
        )

    rows, downloaded_count, skipped_count, error_count = download_selected_objects(
        s3, selected, bucket=bucket, raw_dir=raw_dir, dry_run=dry_run
    )
    index_path = write_index(index_path, rows, case_id=case_id, radar_site=radar_site)
    write_failure_log(log_dir, rows)

    print(f"Radar site: {radar_site}")
    print(f"UTC window: {format_utc(start_time)} to {format_utc(end_time)}")
    print(f"Available radar files: {len(available)}")
    print(f"Selected files: {len(selected)}")
    print(f"Downloaded files: {downloaded_count}")
    print(f"Skipped existing files: {skipped_count}")
    print(f"Download errors: {error_count}")
    print(f"Output index: {index_path}")
    if dry_run:
        print("Dry run: no files were downloaded.")
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List and index files without downloading",
    )
    args = parser.parse_args()
    run_download(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
