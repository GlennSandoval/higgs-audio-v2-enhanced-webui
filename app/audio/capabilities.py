"""Helpers for detecting optional audio processing capabilities."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from shutil import which
from typing import Dict

_OPTIONAL_MODULES = {
    "pydub": "pydub",
    "scipy": "scipy",
}


@lru_cache(maxsize=1)
def has_ffmpeg() -> bool:
    """Return ``True`` when the system ``ffmpeg`` binary is discoverable."""
    return which("ffmpeg") is not None


@lru_cache(maxsize=None)
def has_module(module_name: str) -> bool:
    """Return ``True`` if *module_name* can be imported."""
    return importlib.util.find_spec(module_name) is not None


def dependency_report() -> Dict[str, bool]:
    """Return a dictionary summarising optional audio dependencies."""
    report = {
        "torchaudio": has_module("torchaudio"),
        "ffmpeg": has_ffmpeg(),
    }

    for label, module_name in _OPTIONAL_MODULES.items():
        report[label] = has_module(module_name)

    return report


def describe_missing_dependencies(report: Dict[str, bool]) -> list[str]:
    """Return human readable descriptions for missing audio features."""
    messages: list[str] = []

    if not report.get("ffmpeg", False):
        messages.append(
            "FFmpeg is not on PATH. Install it to enable MP3/advanced format support (e.g. `brew install ffmpeg`)."
        )

    if not report.get("pydub", False):
        messages.append("pydub is missing; install via `pip install pydub` for extended audio format support.")

    if not report.get("scipy", False):
        messages.append("scipy is missing; install via `pip install scipy` for waveform fallbacks.")

    return messages
