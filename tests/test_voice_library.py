import json

import numpy as np

from app import config
from app.services import voice_service


def create_voice_library(tmp_path):
    voice_dir = tmp_path / "library"
    prompts_dir = tmp_path / "prompts"
    voice_dir.mkdir()
    prompts_dir.mkdir()

    def fake_save_audio(audio_data, sample_rate):
        temp_file = tmp_path / "temp.wav"
        temp_file.write_bytes(b"RIFF")
        return temp_file.as_posix()

    def fake_transcribe(path):
        return "fake transcript"

    library = voice_service.VoiceLibrary(
        transcribe_fn=fake_transcribe,
        save_temp_audio_fn=fake_save_audio,
        voice_directory=voice_dir.as_posix(),
        voice_prompts_dir=prompts_dir.as_posix(),
    )
    return library, voice_dir, prompts_dir


def test_save_voice_creates_audio_transcript_and_config(tmp_path):
    library, voice_dir, _ = create_voice_library(tmp_path)

    audio = np.zeros(16, dtype=np.float32)
    result = library.save_voice(audio, config.DEFAULT_SAMPLE_RATE, "Test Voice")

    assert "✅ Voice 'Test_Voice' saved to library" in result

    audio_path = voice_dir / f"Test_Voice{config.VOICE_LIBRARY_AUDIO_EXTENSION}"
    transcript_path = voice_dir / f"Test_Voice{config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION}"
    config_path = voice_dir / f"Test_Voice{config.VOICE_LIBRARY_CONFIG_SUFFIX}"

    assert audio_path.exists()
    assert transcript_path.exists()
    assert config_path.exists()
    assert transcript_path.read_text(encoding="utf-8") == "fake transcript"

    loaded_config = library.load_voice_config("Test_Voice")
    assert loaded_config == config.DEFAULT_VOICE_CONFIG

    duplicate_result = library.save_voice(audio, config.DEFAULT_SAMPLE_RATE, "Test Voice")
    assert "already exists in library" in duplicate_result


def test_delete_voice_removes_files(tmp_path):
    library, voice_dir, _ = create_voice_library(tmp_path)

    audio_path = voice_dir / f"alpha{config.VOICE_LIBRARY_AUDIO_EXTENSION}"
    transcript_path = voice_dir / f"alpha{config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION}"
    config_path = voice_dir / f"alpha{config.VOICE_LIBRARY_CONFIG_SUFFIX}"

    audio_path.write_bytes(b"RIFF")
    transcript_path.write_text("hello", encoding="utf-8")
    config_path.write_text(json.dumps({"temp": 1}), encoding="utf-8")

    message = library.delete_voice("alpha")
    assert "deleted from library" in message
    assert not audio_path.exists()
    assert not transcript_path.exists()
    assert not config_path.exists()


def test_list_all_available_voices_includes_library_and_predefined(tmp_path):
    library, voice_dir, prompts_dir = create_voice_library(tmp_path)

    (voice_dir / f"beta{config.VOICE_LIBRARY_AUDIO_EXTENSION}").write_bytes(b"RIFF")
    (prompts_dir / "sample.wav").write_bytes(b"RIFF")

    voices = library.list_all_available_voices()

    assert voices[0] == config.SMART_VOICE_LABEL
    assert f"{config.PREDEFINED_VOICE_PREFIX}sample.wav" in voices
    assert f"{config.LIBRARY_VOICE_PREFIX}beta" in voices


def test_robust_txt_path_creation_handles_uppercase_extension():
    path = voice_service.robust_txt_path_creation("Sample.WAV")
    expected = "Sample" + config.VOICE_LIBRARY_TRANSCRIPT_EXTENSION
    assert path == expected
