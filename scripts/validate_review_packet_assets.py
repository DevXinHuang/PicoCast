#!/usr/bin/env python
# ruff: noqa: E501
"""Validate all embedded assets and links in the review packet HTML files."""

from __future__ import annotations

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path


class AssetParser(HTMLParser):
    """HTML parser to extract sources and links."""

    def __init__(self) -> None:
        super().__init__()
        # List of tuples: (tag, attr, value)
        self.references: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = dict(attrs)
        if tag == "iframe" and "src" in attrs_dict:
            src = attrs_dict["src"]
            if src:
                self.references.append((tag, "src", src))
        elif tag == "img" and "src" in attrs_dict:
            src = attrs_dict["src"]
            if src:
                self.references.append((tag, "src", src))
        elif tag == "script" and "src" in attrs_dict:
            src = attrs_dict["src"]
            if src:
                self.references.append((tag, "src", src))
        elif tag == "link" and "href" in attrs_dict:
            href = attrs_dict["href"]
            if href:
                self.references.append((tag, "href", href))
        elif tag == "a" and "href" in attrs_dict:
            href = attrs_dict["href"]
            if href:
                self.references.append((tag, "href", href))


def is_external(url: str) -> bool:
    """Check if URL is external or a simple anchor."""
    url_lower = url.lower().strip()
    if not url_lower:
        return True
    if url_lower.startswith(("#", "http://", "https://", "mailto:", "tel:", "javascript:")):
        return True
    return False


def verify_case_sensitive_path(html_file: Path, ref: str) -> tuple[bool, str]:
    """Verify that a relative path exists and matches exact filename casing."""
    # Strip any query parameters or hash anchors
    clean_ref = ref.split("?")[0].split("#")[0]
    if not clean_ref.strip():
        return True, ""

    html_dir = html_file.parent

    # Split the relative path by '/' and '\'
    segments = clean_ref.replace("\\", "/").split("/")
    current = html_dir

    for seg in segments:
        if not seg or seg == ".":
            continue
        if seg == "..":
            current = current.parent
            continue

        if not current.is_dir():
            return False, f"Directory not found: {current.name} for path {clean_ref}"

        try:
            entries = [e.name for e in current.iterdir()]
        except Exception as e:
            return False, f"Failed to list directory {current}: {e}"

        if seg not in entries:
            # Check if it exists with a different case
            entries_lower = [e.lower() for e in entries]
            if seg.lower() in entries_lower:
                correct_case = entries[entries_lower.index(seg.lower())]
                return False, f"Case mismatch: expected '{correct_case}', got '{seg}'"
            return False, f"File or directory '{seg}' not found in {current}"

        current = current / seg

    return True, ""


def validate_directory(dir_path: Path) -> bool:
    """Scan directory and validate all local assets."""
    if not dir_path.is_dir():
        print(f"Error: Target directory does not exist: {dir_path}")
        return False

    html_files = sorted(list(dir_path.rglob("*.html")))
    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp"}
    image_files = [
        f for f in dir_path.rglob("*")
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    local_links_checked = 0
    missing_links = 0
    errors: list[str] = []

    for html_file in html_files:
        try:
            content = html_file.read_text(encoding="utf-8")
        except Exception as e:
            errors.append(f"Failed to read {html_file.relative_to(dir_path.parent)}: {e}")
            continue

        parser = AssetParser()
        try:
            parser.feed(content)
        except Exception as e:
            errors.append(f"Failed to parse HTML in {html_file.relative_to(dir_path.parent)}: {e}")
            continue

        for tag, attr, val in parser.references:
            # Check disallowed prefixes
            if val.strip().lower().startswith("file://"):
                missing_links += 1
                errors.append(
                    f"[{html_file.name}] Prohibited 'file://' link: <{tag} {attr}=\"{val}\">"
                )
                continue

            if val.strip().startswith("/review_packet/"):
                missing_links += 1
                errors.append(
                    f"[{html_file.name}] Prohibited absolute path: <{tag} {attr}=\"{val}\">"
                )
                continue

            # Skip external/anchor links
            if is_external(val):
                continue

            local_links_checked += 1
            valid, msg = verify_case_sensitive_path(html_file, val)
            if not valid:
                missing_links += 1
                errors.append(
                    f"[{html_file.name}] Invalid reference: <{tag} {attr}=\"{val}\"> ({msg})"
                )

    # Print results in required format
    print(f"Number of HTML files: {len(html_files)}")
    print(f"Number of image files: {len(image_files)}")
    print(f"Number of local links checked: {local_links_checked}")
    print(f"Number of missing links: {missing_links}")

    # GitHub Pages entrypoint
    entrypoint = dir_path / "radar_science_lab.html"
    print(f"GitHub Pages entrypoint: {entrypoint}")

    if errors:
        print("\nValidation Errors:")
        for err in errors:
            print(f"  - {err}")
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path("docs/review_packet"),
        help="Directory to scan (default: docs/review_packet)",
    )
    args = parser.parse_args()

    success = validate_directory(args.directory)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
