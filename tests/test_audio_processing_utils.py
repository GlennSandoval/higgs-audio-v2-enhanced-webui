import numpy as np
import pytest

from audio_processing_utils import (adaptive_volume_normalization,
                                    detect_speaker_segments,
                                    enhance_multi_speaker_audio,
                                    normalize_audio_volume)


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    return float(np.sqrt(np.mean(x ** 2)) if x.size > 0 else 0.0)


def test_normalize_audio_volume_basic_mono():
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Quiet sine wave (RMS ~ 0.035)
    wave = 0.05 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    target = 0.1
    out = normalize_audio_volume(wave, target_rms=target, sample_rate=sr)

    assert isinstance(out, np.ndarray)
    assert out.shape == wave.shape
    # RMS should be close to target within reasonable tolerance
    assert pytest.approx(rms(out), rel=0.15, abs=0.02) == target
    # No clipping
    assert np.max(np.abs(out)) <= 1.01


def test_normalize_audio_volume_handles_zero_signal():
    sr = 24000
    silence = np.zeros(sr, dtype=np.float32)
    out = normalize_audio_volume(silence, target_rms=0.1, sample_rate=sr)
    # Should return input unchanged (avoid division by zero)
    assert np.allclose(out, silence)


def test_adaptive_volume_normalization_shape_and_levels():
    sr = 24000
    # Concatenate quiet then loud segments
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    quiet = 0.02 * np.sin(2 * np.pi * 220 * t[:sr]).astype(np.float32)
    loud = 0.4 * np.sin(2 * np.pi * 220 * t[:sr]).astype(np.float32)
    x = np.concatenate([quiet, loud])

    out = adaptive_volume_normalization(x, window_size=0.5, overlap=0.5, target_rms=0.1, sample_rate=sr)

    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape
    # RMS should be bounded and not clip
    assert np.max(np.abs(out)) <= 1.01


def test_detect_speaker_segments_on_silence_and_tone():
    sr = 16000
    # 0.6s silence, 1.0s tone, 0.6s silence
    silence = np.zeros(int(0.6 * sr), dtype=np.float32)
    t = np.linspace(0, 1.0, int(1.0 * sr), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 300 * t).astype(np.float32)
    x = np.concatenate([silence, tone, silence])

    segs = detect_speaker_segments(x, min_segment_length=0.4, energy_threshold=1e-4, sample_rate=sr)
    # Expect exactly one segment roughly around the tone region
    assert len(segs) == 1
    start, end = segs[0]
    assert start >= 0.4 and end <= 1.8 and (end - start) >= 0.8


essential_methods = ["simple", "adaptive", "segment-based"]
@pytest.mark.parametrize("method", essential_methods)
def test_enhance_multi_speaker_audio_methods(method):
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    x = 0.05 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    out = enhance_multi_speaker_audio(x, sample_rate=sr, normalization_method=method, target_rms=0.1)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape
    assert np.max(np.abs(out)) <= 1.01
