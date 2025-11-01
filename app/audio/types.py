"""Audio data wrappers used throughout the Higgs Audio application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - used for static type checking only
    from pydub import AudioSegment as _AudioSegment
else:  # pragma: no cover - runtime fallback when type checking disabled
    _AudioSegment = Any

try:  # Optional dependency; keep import lazy to support minimal setups
    from pydub import AudioSegment  # type: ignore
except ImportError:  # pragma: no cover - exercised when pydub optional dep missing
    AudioSegment = None  # type: ignore


def _is_torch_tensor(value: Any) -> bool:
    """Return ``True`` if *value* looks like a torch Tensor without importing torch eagerly."""
    try:  # Local import to avoid torch dependency at module import time
        import torch  # type: ignore
    except ImportError:  # pragma: no cover - torch is required at runtime but not for tests
        return False

    return isinstance(value, torch.Tensor)


def _ensure_channels_first(array: np.ndarray) -> np.ndarray:
    """Ensure the provided array uses channels-first convention."""
    if array.ndim == 1:
        return array

    if array.ndim != 2:
        raise ValueError("AudioBuffer expects 1D mono or 2D (channels, samples) arrays")

    channels, samples = array.shape
    if channels > samples and samples <= 8:
        array = array.T
    return array


@dataclass(frozen=True)
class AudioBuffer:
    """Wrapper around a numpy audio array and its associated metadata."""

    samples: np.ndarray
    sample_rate: int
    source_segment: Optional["_AudioSegment"] = None

    def __post_init__(self) -> None:
        array = np.asarray(self.samples, dtype=np.float32)
        array = np.ascontiguousarray(_ensure_channels_first(array))
        object.__setattr__(self, "samples", array)

        if array.ndim not in (1, 2):  # pragma: no cover - safeguarded by _ensure_channels_first
            raise ValueError("AudioBuffer only supports mono or multi-channel arrays")

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_numpy(cls, samples: np.ndarray, sample_rate: int) -> "AudioBuffer":
        return cls(samples=samples, sample_rate=sample_rate)

    @classmethod
    def from_tensor(cls, tensor: Any, sample_rate: int) -> "AudioBuffer":
        if not _is_torch_tensor(tensor):
            raise TypeError("Expected a torch.Tensor-like object")

        import torch  # Local import to avoid hard dependency at module import time

        ndarray = tensor.detach().cpu().float().numpy()
        return cls(samples=ndarray, sample_rate=sample_rate)

    @classmethod
    def from_audio_segment(cls, segment: "_AudioSegment") -> "AudioBuffer":
        if AudioSegment is None:
            raise RuntimeError("pydub AudioSegment is not available")
        if not isinstance(segment, AudioSegment):  # pragma: no cover - defensive
            raise TypeError("segment must be an instance of pydub.AudioSegment")

        raw_samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        if segment.sample_width == 1:
            raw_samples = raw_samples / 128.0
        elif segment.sample_width == 2:
            raw_samples = raw_samples / 32768.0
        elif segment.sample_width == 4:
            raw_samples = raw_samples / 2147483648.0
        else:
            max_val = np.max(np.abs(raw_samples)) or 1.0
            raw_samples = raw_samples / max_val

        if segment.channels > 1:
            raw_samples = raw_samples.reshape((-1, segment.channels)).T

        return cls(samples=raw_samples, sample_rate=segment.frame_rate, source_segment=segment)

    @classmethod
    def coerce(
        cls,
        audio: Any,
        sample_rate: Optional[int] = None,
    ) -> "AudioBuffer":
        """Coerce supported audio inputs into an :class:`AudioBuffer`."""
        if isinstance(audio, AudioBuffer):
            return audio

        if AudioSegment is not None and isinstance(audio, AudioSegment):
            return cls.from_audio_segment(audio)

        if _is_torch_tensor(audio):
            if sample_rate is None:
                raise ValueError("sample_rate is required when coercing torch tensors")
            return cls.from_tensor(audio, sample_rate)

        if isinstance(audio, np.ndarray):
            if sample_rate is None:
                raise ValueError("sample_rate is required when coercing numpy arrays")
            return cls.from_numpy(audio, sample_rate)

        raise TypeError(
            "Unsupported audio input type. Provide an AudioBuffer, numpy array, torch tensor, or AudioSegment."
        )

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    @property
    def channels(self) -> int:
        return 1 if self.samples.ndim == 1 else self.samples.shape[0]

    @property
    def num_samples(self) -> int:
        return self.samples.shape[-1]

    def ensure_mono(self) -> "AudioBuffer":
        if self.samples.ndim == 1 or self.channels == 1:
            return self
        mono = np.mean(self.samples, axis=0)
        return AudioBuffer(samples=mono, sample_rate=self.sample_rate)

    def ensure_channels_first(self) -> np.ndarray:
        return self.samples if self.samples.ndim == 1 else self.samples

    def ensure_2d(self) -> np.ndarray:
        if self.samples.ndim == 2:
            return self.samples
        return self.samples[np.newaxis, :]

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def to_numpy(self, copy: bool = True) -> np.ndarray:
        return self.samples.copy() if copy else self.samples

    def to_tensor(self) -> Any:
        import torch

        return torch.from_numpy(self.ensure_2d()).float()

    def to_audio_segment(self) -> "_AudioSegment":
        if AudioSegment is None:
            raise RuntimeError("pydub AudioSegment is not available")

        mono_or_channels_first = self.ensure_2d()
        clipped = np.clip(mono_or_channels_first, -1.0, 1.0)
        int_samples = (clipped * 32767.0).astype(np.int16)

        if int_samples.shape[0] == 1:
            raw_bytes = int_samples.flatten().tobytes()
            channels = 1
        else:
            interleaved = int_samples.transpose(1, 0).reshape(-1)
            raw_bytes = interleaved.tobytes()
            channels = int_samples.shape[0]

        return AudioSegment(
            data=raw_bytes,
            sample_width=2,
            frame_rate=self.sample_rate,
            channels=channels,
        )

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def with_samples(self, samples: np.ndarray, sample_rate: Optional[int] = None) -> "AudioBuffer":
        return AudioBuffer(samples=samples, sample_rate=sample_rate or self.sample_rate)

    def scaled(self, gain: float) -> "AudioBuffer":
        return self.with_samples(self.samples * gain)

    def clone(self) -> "AudioBuffer":
        return AudioBuffer(samples=self.samples.copy(), sample_rate=self.sample_rate, source_segment=self.source_segment)


SpeakerTimestamp = Tuple[float, float, str]
"""Type alias used when interoperability with legacy tuple structures is required."""
