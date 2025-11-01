"""Audio tooling namespace for the Higgs Audio WebUI."""

from app.audio.capabilities import (dependency_report,
                                    describe_missing_dependencies, has_ffmpeg)
from app.audio.processing import (SpeakerSegment,
                                  adaptive_volume_normalization,
                                  detect_speaker_segments,
                                  enhance_multi_speaker_audio,
                                  normalize_audio_volume,
                                  normalize_multi_speaker_segments)
from app.audio.types import AudioBuffer, SpeakerTimestamp

__all__ = [
    "AudioBuffer",
    "SpeakerSegment",
    "SpeakerTimestamp",
    "normalize_audio_volume",
    "normalize_multi_speaker_segments",
    "adaptive_volume_normalization",
    "detect_speaker_segments",
    "enhance_multi_speaker_audio",
    "has_ffmpeg",
    "dependency_report",
    "describe_missing_dependencies",
]
