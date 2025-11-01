"""Audio regression and normalization tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.audio.processing import (enhance_multi_speaker_audio,
                                  normalize_audio_volume)
from app.audio.types import AudioBuffer
from tests.utils.audio_fixtures import validate_audio_fixture_hashes


def _rms(samples: np.ndarray) -> float:
    return float(math.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def test_normalize_audio_volume_hits_target_rms() -> None:
    sample_rate = 24_000
    duration_seconds = 1.0
    time_axis = np.linspace(0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    waveform = (0.01 * np.sin(2 * np.pi * 440 * time_axis)).astype(np.float32)

    buffer = AudioBuffer(waveform, sample_rate)
    normalized = normalize_audio_volume(buffer, target_rms=0.1, max_gain=20.0)

    assert _rms(normalized.samples) == pytest.approx(0.1, rel=0.1)


def test_segment_based_enhancement_balances_segments() -> None:
    sample_rate = 8_000
    segment_length = sample_rate // 2
    first_segment = np.full(segment_length, 0.3, dtype=np.float32)
    second_segment = np.full(segment_length, 0.12, dtype=np.float32)
    silence = np.zeros(sample_rate // 8, dtype=np.float32)
    waveform = np.concatenate([first_segment, silence, second_segment])

    buffer = AudioBuffer(waveform, sample_rate)
    enhanced = enhance_multi_speaker_audio(
        buffer,
        normalization_method="segment-based",
        target_rms=0.1,
    )

    first_rms = _rms(enhanced.samples[:segment_length])
    silence_rms = _rms(enhanced.samples[segment_length : segment_length + silence.size])
    second_rms = _rms(enhanced.samples[-segment_length:])

    assert first_rms == pytest.approx(second_rms, rel=0.2)
    assert first_rms == pytest.approx(0.1, rel=0.2)
    assert silence_rms < 1e-4


def test_audio_fixture_hashes_match_baseline() -> None:
    mismatches = list(validate_audio_fixture_hashes())
    assert not mismatches, f"Audio fixtures drifted: {mismatches}"
