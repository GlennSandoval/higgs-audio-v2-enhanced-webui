import os

import numpy as np
import torch
from scipy.io import wavfile

from app import audio_io, config


def generate_test_waveform(channels: int, sample_rate: int, duration_seconds: float = 0.2) -> torch.Tensor:
    """Create a simple sine waveform for testing."""
    num_samples = int(sample_rate * duration_seconds)
    t = torch.linspace(0, duration_seconds, num_samples, dtype=torch.float32, requires_grad=False)
    base_wave = torch.sin(2 * torch.pi * 440 * t)
    if channels == 1:
        return base_wave.unsqueeze(0)
    return torch.stack([base_wave, base_wave * 0.5], dim=0)


def test_convert_audio_to_standard_format_resamples_and_preserves_channels(tmp_path):
    src_sr = 16_000
    target_sr = config.DEFAULT_SAMPLE_RATE
    waveform = generate_test_waveform(channels=2, sample_rate=src_sr)
    src_path = tmp_path / "stereo.wav"
    wavfile.write(src_path.as_posix(), src_sr, waveform.transpose(0, 1).numpy())

    audio_data, sample_rate = audio_io.convert_audio_to_standard_format(
        src_path.as_posix(), target_sample_rate=target_sr, force_mono=False
    )

    assert isinstance(audio_data, np.ndarray)
    assert audio_data.shape[0] == 2
    expected_len = int(waveform.shape[1] * target_sr / src_sr)
    assert abs(audio_data.shape[1] - expected_len) <= 1
    assert sample_rate == target_sr


def test_convert_audio_to_standard_format_force_mono(tmp_path):
    src_sr = config.DEFAULT_SAMPLE_RATE
    waveform = generate_test_waveform(channels=2, sample_rate=src_sr)
    src_path = tmp_path / "force_mono.wav"
    wavfile.write(src_path.as_posix(), src_sr, waveform.transpose(0, 1).numpy())

    audio_data, sample_rate = audio_io.convert_audio_to_standard_format(
        src_path.as_posix(), target_sample_rate=src_sr, force_mono=True
    )

    assert isinstance(audio_data, np.ndarray)
    assert audio_data.ndim == 1
    assert len(audio_data) == waveform.shape[1]
    assert sample_rate == src_sr


def test_process_uploaded_audio_normalizes_and_transposes():
    sr = config.DEFAULT_SAMPLE_RATE
    samples = np.arange(8, dtype=np.int16).reshape(4, 2)  # shape (4, 2) -> samples, channels
    audio_np = samples.copy()

    processed, processed_sr = audio_io.process_uploaded_audio((sr, audio_np), force_mono=False)

    assert processed_sr == sr
    assert processed.dtype == np.float32
    assert processed.shape == (2, 4)
    assert np.max(np.abs(processed)) <= 1.0


def test_save_temp_audio_robust_creates_file_and_is_readable():
    sr = config.DEFAULT_SAMPLE_RATE
    waveform = generate_test_waveform(channels=1, sample_rate=sr).squeeze(0).numpy()

    def fake_save(path, tensor_waveform, sample_rate):
        data = tensor_waveform.detach().cpu().numpy()
        if data.ndim == 2:
            data = data.transpose(1, 0)
        wavfile.write(path, sample_rate, data.astype(np.float32))

    torchaudio_save = audio_io.torchaudio.save
    audio_io.torchaudio.save = fake_save

    try:
        temp_path = audio_io.save_temp_audio_robust(waveform, sr)
    finally:
        audio_io.torchaudio.save = torchaudio_save

    assert os.path.exists(temp_path)

    loaded_sr, loaded_waveform = wavfile.read(temp_path)
    assert loaded_sr == sr
    if loaded_waveform.ndim == 2:
        loaded_waveform = loaded_waveform[:, 0]
    assert loaded_waveform.shape[0] == waveform.shape[0]

    os.remove(temp_path)


def test_robust_file_cleanup_handles_missing_and_existing(tmp_path):
    temp_file = tmp_path / "cleanup.wav"
    temp_file.write_bytes(b"1234")

    # Should remove existing file
    audio_io.robust_file_cleanup(temp_file.as_posix())
    assert not temp_file.exists()

    # Should ignore missing file gracefully
    audio_io.robust_file_cleanup(temp_file.as_posix())
