"""Audio post-processing helpers with typed interfaces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import numpy as np

from app.audio.types import AudioBuffer

logger = logging.getLogger(__name__)


def _coerce_buffer(audio: Any, sample_rate: int | None) -> AudioBuffer:
    """Resolve supported inputs into an :class:`AudioBuffer`."""
    return AudioBuffer.coerce(audio, sample_rate)


@dataclass(frozen=True)
class SpeakerSegment:
    """Represents a detected speaker segment in the timeline."""

    start_time: float
    end_time: float
    speaker_id: str = ""

    def to_tuple(self) -> tuple[float, float, str]:
        return (self.start_time, self.end_time, self.speaker_id)


def _coerce_segments(segments: Iterable[SpeakerSegment | Sequence[float]]) -> list[SpeakerSegment]:
    """Normalize segment definitions into :class:`SpeakerSegment` instances."""
    normalized: list[SpeakerSegment] = []
    for entry in segments:
        if isinstance(entry, SpeakerSegment):
            normalized.append(entry)
            continue

        if len(entry) == 3:
            start, end, speaker = entry  # type: ignore[misc]
        elif len(entry) == 2:
            start, end = entry  # type: ignore[misc]
            speaker = ""
        else:  # pragma: no cover - defensive guard
            raise ValueError("Speaker segments must be length 2 or 3")

        normalized.append(SpeakerSegment(float(start), float(end), str(speaker)))
    return normalized


def normalize_audio_volume(
    audio: Any,
    *,
    target_rms: float = 0.1,
    max_gain: float = 10.0,
    sample_rate: int | None = None,
) -> AudioBuffer:
    """Normalize audio volume to the requested RMS target."""
    buffer = _coerce_buffer(audio, sample_rate)
    data = buffer.to_numpy(copy=True)

    mono_reference = data if data.ndim == 1 else np.mean(data, axis=0)
    current_rms = np.sqrt(np.mean(mono_reference ** 2))

    if current_rms == 0:
        logger.warning("Audio has zero RMS - skipping normalization")
        return buffer

    gain = min(target_rms / max(current_rms, 1e-9), max_gain)
    normalized = data * gain

    max_val = float(np.max(np.abs(normalized))) if normalized.size else 0.0
    if max_val > 1.0:
        normalized = normalized / max_val * 0.99

    logger.info(
        "Audio normalization complete: RMS %.3f → %.3f (gain %.2fx)",
        current_rms,
        float(np.sqrt(np.mean((normalized if normalized.ndim == 1 else np.mean(normalized, axis=0)) ** 2))),
        gain,
    )

    return buffer.with_samples(normalized)


def normalize_multi_speaker_segments(
    audio: Any,
    speaker_segments: Sequence[SpeakerSegment | Sequence[float]],
    *,
    target_rms: float = 0.1,
    sample_rate: int | None = None,
) -> AudioBuffer:
    """Normalize specific speaker segments independently."""
    buffer = _coerce_buffer(audio, sample_rate)
    samples = buffer.to_numpy(copy=True)
    sr = buffer.sample_rate

    normalized_segments = _coerce_segments(speaker_segments)
    if not normalized_segments:
        return buffer

    for segment in normalized_segments:
        start_sample = max(0, int(segment.start_time * sr))
        end_sample = min(samples.shape[-1], int(segment.end_time * sr))
        if end_sample <= start_sample:
            continue

        sub_slice = samples[..., start_sample:end_sample]
        sub_buffer = AudioBuffer(sub_slice, sr)
        normalized_segment = normalize_audio_volume(
            sub_buffer, target_rms=target_rms
        ).samples
        samples[..., start_sample:end_sample] = normalized_segment

    return buffer.with_samples(samples)


def _normalize_channel_windows(
    channel_data: np.ndarray,
    *,
    window_size: float,
    overlap: float,
    target_rms: float,
    sample_rate: int,
) -> np.ndarray:
    """Apply adaptive normalization to a single audio channel."""
    window_samples = max(1, int(window_size * sample_rate))
    hop_samples = max(1, int(window_samples * (1 - overlap)))

    output = np.zeros_like(channel_data)
    weight_sum = np.zeros_like(channel_data)

    for start in range(0, max(1, channel_data.size - window_samples + 1), hop_samples):
        end = start + window_samples
        window = channel_data[start:end]
        normalized_window = normalize_audio_volume(
            AudioBuffer(window, sample_rate), target_rms=target_rms
        ).samples
        hann_window = np.hanning(len(normalized_window))
        normalized_window = normalized_window * hann_window

        output[start:end] += normalized_window
        weight_sum[start:end] += hann_window

    with np.errstate(divide="ignore", invalid="ignore"):
        combined = np.divide(
            output,
            weight_sum,
            out=np.zeros_like(output),
            where=weight_sum != 0,
        )

    if weight_sum[0] == 0:
        combined[:hop_samples] = normalize_audio_volume(
            AudioBuffer(channel_data[:hop_samples], sample_rate), target_rms=target_rms
        ).samples
    if weight_sum[-1] == 0:
        combined[-hop_samples:] = normalize_audio_volume(
            AudioBuffer(channel_data[-hop_samples:], sample_rate), target_rms=target_rms
        ).samples

    return combined


def adaptive_volume_normalization(
    audio: Any,
    *,
    window_size: float = 2.0,
    overlap: float = 0.5,
    target_rms: float = 0.1,
    sample_rate: int | None = None,
) -> AudioBuffer:
    """Apply adaptive normalization using sliding windows."""
    buffer = _coerce_buffer(audio, sample_rate)
    samples_2d = buffer.ensure_2d().copy()

    normalized_channels = []
    for channel_index in range(samples_2d.shape[0]):
        channel = samples_2d[channel_index]
        normalized_channel = _normalize_channel_windows(
            channel,
            window_size=window_size,
            overlap=overlap,
            target_rms=target_rms,
            sample_rate=buffer.sample_rate,
        )
        normalized_channels.append(normalized_channel)

    normalized_array = np.vstack(normalized_channels)
    if buffer.samples.ndim == 1 or normalized_array.shape[0] == 1:
        normalized_array = normalized_array[0]

    logger.info("Applied adaptive normalization with %.1fs windows", window_size)
    return buffer.with_samples(normalized_array)


def detect_speaker_segments(
    audio: Any,
    *,
    min_segment_length: float = 0.5,
    energy_threshold: float = 0.01,
    sample_rate: int | None = None,
) -> list[SpeakerSegment]:
    """Detect potential speaker segments based on energy levels."""
    buffer = _coerce_buffer(audio, sample_rate).ensure_mono()
    data = buffer.to_numpy(copy=False)
    sr = buffer.sample_rate

    frame_length = int(0.025 * sr)
    hop_length = max(1, int(0.010 * sr))

    energies: list[float] = []
    for start in range(0, max(1, data.size - frame_length + 1), hop_length):
        frame = data[start : start + frame_length]
        energies.append(float(np.mean(frame ** 2)))

    speech_frames = np.array(energies) > energy_threshold

    segments: list[SpeakerSegment] = []
    in_segment = False
    segment_start = 0.0

    for index, is_speech in enumerate(speech_frames):
        current_time = index * hop_length / sr
        if is_speech and not in_segment:
            segment_start = current_time
            in_segment = True
        elif not is_speech and in_segment:
            duration = current_time - segment_start
            if duration >= min_segment_length:
                segments.append(SpeakerSegment(segment_start, current_time))
            in_segment = False

    if in_segment:
        final_time = data.size / sr
        duration = final_time - segment_start
        if duration >= min_segment_length:
            segments.append(SpeakerSegment(segment_start, final_time))

    logger.info("Detected %d speaker segments", len(segments))
    return segments


def enhance_multi_speaker_audio(
    audio: Any,
    *,
    normalization_method: str = "adaptive",
    target_rms: float = 0.1,
    sample_rate: int | None = None,
) -> AudioBuffer:
    """Enhance multi-speaker audio using the chosen normalization strategy."""
    buffer = _coerce_buffer(audio, sample_rate)

    if normalization_method == "simple":
        return normalize_audio_volume(buffer, target_rms=target_rms)

    if normalization_method == "adaptive":
        return adaptive_volume_normalization(
            buffer,
            target_rms=target_rms,
        )

    if normalization_method == "segment-based":
        segments = detect_speaker_segments(buffer, sample_rate=buffer.sample_rate)
        return normalize_multi_speaker_segments(
            buffer,
            segments,
            target_rms=target_rms,
        )

    raise ValueError(f"Unknown normalization method: {normalization_method}")
