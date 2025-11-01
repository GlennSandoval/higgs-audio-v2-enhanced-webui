"""Utilities for validating audio fixture checksums used in regression tests."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterator

BASELINE_HASH_FILE = Path("output/BASELINE_AUDIO_HASHES.md")
ROW_PATTERN = re.compile(r"`([0-9a-f]{64})`\s*\|\s*`([^`]+)`")


def load_baseline_hashes() -> dict[str, str]:
    """Return the mapping of fixture paths to expected SHA-256 digests."""
    if not BASELINE_HASH_FILE.exists():
        raise FileNotFoundError(
            "Baseline audio hash file is missing: output/BASELINE_AUDIO_HASHES.md"
        )

    content = BASELINE_HASH_FILE.read_text(encoding="utf-8")
    matches = ROW_PATTERN.findall(content)

    if not matches:
        raise ValueError(
            "No checksum rows detected in output/BASELINE_AUDIO_HASHES.md"
        )

    return {path: digest for digest, path in matches}


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 digest for the provided file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_audio_fixture_hashes() -> Iterator[dict[str, str]]:
    """Yield mismatch details for any audio fixtures whose hashes drift."""
    baseline = load_baseline_hashes()

    for relative_path, expected in baseline.items():
        file_path = Path(relative_path)
        if not file_path.exists():
            yield {
                "status": "missing",
                "path": relative_path,
                "expected": expected,
            }
            continue

        actual = compute_sha256(file_path)
        if actual != expected:
            yield {
                "status": "mismatch",
                "path": relative_path,
                "expected": expected,
                "actual": actual,
            }


__all__ = [
    "BASELINE_HASH_FILE",
    "compute_sha256",
    "load_baseline_hashes",
    "validate_audio_fixture_hashes",
]
