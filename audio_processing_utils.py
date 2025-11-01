"""Compatibility shim for legacy imports.

Phase 3 consolidated audio utilities under :mod:`app.audio`. This module keeps
the historic ``audio_processing_utils`` import path working while emitting a
deprecation warning. All functionality is re-exported from the new package.
"""

from __future__ import annotations

import warnings

from app.audio import (AudioBuffer, SpeakerSegment,
                       adaptive_volume_normalization, detect_speaker_segments,
                       enhance_multi_speaker_audio, normalize_audio_volume,
                       normalize_multi_speaker_segments)

warnings.warn(
    "audio_processing_utils is deprecated; import from app.audio.processing instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AudioBuffer",
    "SpeakerSegment",
    "normalize_audio_volume",
    "normalize_multi_speaker_segments",
    "adaptive_volume_normalization",
    "detect_speaker_segments",
    "enhance_multi_speaker_audio",
]