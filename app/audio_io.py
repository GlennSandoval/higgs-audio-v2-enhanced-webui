"""Audio input/output helpers for the Higgs Audio Gradio application."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from pydub import AudioSegment

from app import config


__all__ = [
    "convert_audio_to_standard_format",
    "load_audio_file_robust",
    "process_uploaded_audio",
    "save_temp_audio",
    "save_temp_audio_robust",
    "enhanced_save_temp_audio_fixed",
    "safe_audio_processing",
    "get_output_path",
    "save_transcript_if_enabled",
    "save_audio_reference_if_enabled",
    "robust_file_cleanup",
    "save_temp_audio",
]


def convert_audio_to_standard_format(
    audio_path: str,
    target_sample_rate: int = config.DEFAULT_SAMPLE_RATE,
    force_mono: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Load audio from ``audio_path`` and return a numpy array plus the sample rate.

    The loader tries torchaudio first, then pydub, and finally scipy as a fallback.
    Stereo channel structure is preserved unless ``force_mono`` is True.
    Raises:
        ValueError: if none of the loaders succeeds.
    """
    print(f"🔄 Converting audio file: {audio_path}")

    try:
        waveform, sample_rate = torchaudio.load(audio_path)

        if force_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("🔄 Converted stereo to mono")

        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate

        if waveform.shape[0] == 1:
            audio_data = waveform.squeeze().numpy()
        else:
            audio_data = waveform.numpy()

        channels = waveform.shape[0]
        samples = waveform.shape[1]
        print(
            f"✅ Loaded with torchaudio: "
            f"{'stereo' if channels == 2 else 'mono'} - {samples} samples at {sample_rate}Hz"
        )
        return audio_data, sample_rate

    except Exception as exc:
        print(f"⚠️ Torchaudio failed: {exc}")

    try:
        if audio_path.lower().endswith(".mp3"):
            audio = AudioSegment.from_mp3(audio_path)
        elif audio_path.lower().endswith(".wav"):
            audio = AudioSegment.from_wav(audio_path)
        else:
            audio = AudioSegment.from_file(audio_path)

        if force_mono and audio.channels > 1:
            audio = audio.set_channels(1)
            print("🔄 Converted stereo to mono")

        audio = audio.set_frame_rate(target_sample_rate)

        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)

        if audio.sample_width == 1:
            audio_data = audio_data / 128.0
        elif audio.sample_width == 2:
            audio_data = audio_data / 32768.0
        elif audio.sample_width == 4:
            audio_data = audio_data / 2147483648.0
        else:
            max_val = np.max(np.abs(audio_data))
            if max_val > 1:
                audio_data = audio_data / max_val

        if audio.channels == 2 and not force_mono:
            audio_data = audio_data.reshape(-1, 2).T

        channel_info = "stereo" if audio.channels == 2 and not force_mono else "mono"
        print(f"✅ Loaded with pydub: {channel_info} - {len(audio_data)} samples at {target_sample_rate}Hz")
        return audio_data, target_sample_rate

    except Exception as exc:
        print(f"⚠️ Pydub failed: {exc}")

    try:
        from scipy.io import wavfile

        sample_rate, audio_data = wavfile.read(audio_path)

        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
        else:
            audio_data = audio_data.astype(np.float32)

        if len(audio_data.shape) > 1:
            if force_mono:
                audio_data = np.mean(audio_data, axis=1)
                print("🔄 Converted stereo to mono")
            else:
                audio_data = audio_data.T

        if sample_rate != target_sample_rate:
            ratio = target_sample_rate / sample_rate
            if audio_data.ndim == 1:
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data,
                )
            else:
                new_length = int(audio_data.shape[1] * ratio)
                resampled = np.zeros((audio_data.shape[0], new_length))
                for channel in range(audio_data.shape[0]):
                    resampled[channel] = np.interp(
                        np.linspace(0, audio_data.shape[1], new_length),
                        np.arange(audio_data.shape[1]),
                        audio_data[channel],
                    )
                audio_data = resampled
            sample_rate = target_sample_rate

        channel_info = "stereo" if audio_data.ndim > 1 else "mono"
        samples = audio_data.shape[1] if audio_data.ndim > 1 else len(audio_data)
        print(f"✅ Loaded with scipy: {channel_info} - {samples} samples at {sample_rate}Hz")
        return audio_data, sample_rate

    except Exception as exc:
        print(f"⚠️ Scipy failed: {exc}")

    raise ValueError(
        f"❌ Could not load audio file: {audio_path}. Tried torchaudio, pydub, and scipy."
    )


def save_temp_audio_robust(
    audio_data: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    force_mono: bool = False,
) -> str:
    """
    Persist in-memory audio to a temporary WAV file and return the path.

    Handles numpy arrays or torch tensors, enforces float32 data, and preserves
    stereo channels unless ``force_mono`` is requested.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    try:
        if isinstance(audio_data, torch.Tensor):
            audio_np = audio_data.cpu().numpy()
        elif isinstance(audio_data, np.ndarray):
            audio_np = audio_data
        else:
            audio_np = np.array(audio_data)

        if audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32)

        if audio_np.ndim == 1:
            audio_np = audio_np if force_mono else np.expand_dims(audio_np, axis=0)
        elif audio_np.ndim == 2:
            if audio_np.shape[0] > audio_np.shape[1]:
                audio_np = audio_np.T
            if force_mono and audio_np.shape[0] > 1:
                audio_np = np.mean(audio_np, axis=0, keepdims=True)

        max_val = np.max(np.abs(audio_np))
        if max_val > 1.0:
            audio_np = audio_np / max_val

        waveform = torch.from_numpy(audio_np).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        print(
            f"Saving audio: shape={waveform.shape}, dtype={waveform.dtype}, "
            f"max={waveform.max()}, min={waveform.min()}"
        )

        torchaudio.save(temp_path, waveform, sample_rate)
        print(f"✅ Saved audio to: {temp_path}")
        return temp_path

    except Exception as exc:
        print(f"❌ Error saving audio: {exc}")
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception:
                pass
        raise


def process_uploaded_audio(
    uploaded_audio: Tuple[int, Union[np.ndarray, torch.Tensor]],
    force_mono: bool = False,
) -> Tuple[np.ndarray, int]:
    """
    Normalize audio uploaded through Gradio widgets.

    Args:
        uploaded_audio: Tuple of (sample_rate, audio_data).
        force_mono: Whether to collapse audio to a single channel.
    Returns:
        The processed audio data and sample rate.
    """
    if uploaded_audio is None:
        raise ValueError("No audio uploaded")

    if isinstance(uploaded_audio, tuple) and len(uploaded_audio) == 2:
        sample_rate, audio_data = uploaded_audio
    else:
        raise ValueError("Invalid uploaded audio format - expected (sample_rate, audio_data) tuple")

    if isinstance(audio_data, torch.Tensor):
        audio_np = audio_data.cpu().numpy()
    elif isinstance(audio_data, np.ndarray):
        audio_np = audio_data
    else:
        audio_np = np.array(audio_data)

    if audio_np.dtype != np.float32:
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0
        elif audio_np.dtype == np.int32:
            audio_np = audio_np.astype(np.float32) / 2147483648.0
        else:
            audio_np = audio_np.astype(np.float32)

    if audio_np.ndim > 1:
        if force_mono:
            audio_np = np.mean(audio_np, axis=1)
            print("🔄 Converted stereo to mono")
        else:
            if audio_np.shape[1] < audio_np.shape[0]:
                audio_np = audio_np.T

    max_val = np.max(np.abs(audio_np))
    if max_val > 1.0:
        audio_np = audio_np / max_val

    if audio_np.ndim > 1:
        channel_info = f"stereo ({audio_np.shape[0]} channels)"
        samples = audio_np.shape[1]
    else:
        channel_info = "mono"
        samples = len(audio_np)

    print(f"✅ Processed uploaded audio: {channel_info} - {samples} samples at {sample_rate}Hz")
    return audio_np, sample_rate


def enhanced_save_temp_audio_fixed(
    uploaded_voice: Optional[Tuple[int, Union[np.ndarray, torch.Tensor]]],
    force_mono: bool = False,
) -> str:
    """
    Process an uploaded voice sample and save it to a temporary WAV file.

    Args:
        uploaded_voice: Tuple of (sample_rate, audio_data) from Gradio.
        force_mono: Whether to collapse audio to a single channel.
    Returns:
        Path to the temporary file on disk.
    """
    if uploaded_voice is None:
        raise ValueError("No uploaded voice provided")

    if isinstance(uploaded_voice, tuple) and len(uploaded_voice) == 2:
        processed_audio, processed_rate = process_uploaded_audio(uploaded_voice, force_mono)
        return save_temp_audio_robust(processed_audio, processed_rate, force_mono)

    raise ValueError("Invalid uploaded voice format - expected (sample_rate, audio_data) tuple")


def load_audio_file_robust(
    file_path: str, target_sample_rate: int = config.DEFAULT_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """Load an arbitrary audio file from disk using ``convert_audio_to_standard_format``."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    return convert_audio_to_standard_format(file_path, target_sample_rate)


def safe_audio_processing(
    uploaded_voice: Optional[Tuple[int, Union[np.ndarray, torch.Tensor]]],
    operation_name: str,
) -> str:
    """
    Wrap ``enhanced_save_temp_audio_fixed`` with user-friendly error handling.

    Args:
        uploaded_voice: Tuple provided by Gradio.
        operation_name: Human-readable name of the action for error messaging.
    Returns:
        Path to the processed temporary file.
    """
    try:
        return enhanced_save_temp_audio_fixed(uploaded_voice)
    except Exception as exc:
        error_msg = f"❌ Error processing audio for {operation_name}: {str(exc)}\n"
        error_msg += "💡 Try these solutions:\n"
        error_msg += "  • Ensure your audio file is a valid WAV or MP3\n"
        error_msg += "  • Try converting your file using a different audio editor\n"
        error_msg += "  • Make sure the file isn't corrupted\n"
        error_msg += "  • Install additional dependencies: pip install pydub scipy"
        raise ValueError(error_msg)


def get_output_path(category: str, filename_base: str, extension: str = ".wav") -> str:
    """
    Build a timestamped output path under ``config.OUTPUT_BASE_DIR`` for the given category.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename_base}{extension}"
    output_path = os.path.join(config.OUTPUT_BASE_DIR, category, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path


def save_transcript_if_enabled(transcript: str, category: str, filename_base: str) -> None:
    """Placeholder for future transcript persistence; currently disabled."""
    return None


def save_audio_reference_if_enabled(audio_path: str, category: str, filename_base: str) -> None:
    """Placeholder for future audio-reference persistence; currently disabled."""
    return None


def robust_file_cleanup(files: Union[str, Iterable[Optional[str]]]) -> None:
    """
    Delete one or more files if they exist, ignoring missing files and errors.
    """
    if not files:
        return

    if isinstance(files, str):
        files = [files]

    for file_path in files:
        if file_path and isinstance(file_path, str) and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass


def save_temp_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Save numpy audio data directly to a temporary WAV file.
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()

    if isinstance(audio_data, np.ndarray):
        waveform = torch.from_numpy(audio_data).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(temp_path, waveform, sample_rate)

    return temp_path
