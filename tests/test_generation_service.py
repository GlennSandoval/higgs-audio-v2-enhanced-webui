"""Tests for GenerationService behavior using mocked engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from app import config
from app.services.generation_service import GenerationService
from boson_multimodal.data_types import AudioContent, Message
from boson_multimodal.serve.serve_engine import HiggsAudioResponse


class StubVoiceLibrary:
    def list_all_available_voices(self) -> list[str]:
        return [config.SMART_VOICE_LABEL]

    def get_voice_path(self, voice_selection: str | None) -> str | None:  # pragma: no cover - unused in tests
        return None

    def create_voice_reference_txt(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - unused in tests
        return "voice.txt"

    def robust_txt_path_creation(self, *args: Any, **kwargs: Any) -> str:  # pragma: no cover - unused in tests
        return "voice.txt"


@dataclass
class FakeEngineCall:
    kwargs: dict[str, Any]
    response: HiggsAudioResponse


class FakeEngine:
    def __init__(self, segment_length: int = 100, sample_rate: int = 1_000) -> None:
        self._segment_length = segment_length
        self._sample_rate = sample_rate
        self.calls: list[FakeEngineCall] = []

    def generate(self, **kwargs: Any) -> HiggsAudioResponse:
        index = len(self.calls) + 1
        audio = np.full(self._segment_length, index, dtype=np.float32)
        response = HiggsAudioResponse(audio=audio, sampling_rate=self._sample_rate)
        self.calls.append(FakeEngineCall(kwargs=kwargs, response=response))
        return response


def test_generate_with_cache_uses_cached_result(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    service = GenerationService(
        device=torch.device("cpu"),
        voice_library_service=StubVoiceLibrary(),
    )

    fake_engine = FakeEngine(segment_length=50, sample_rate=24_000)
    service._serve_engine = fake_engine

    messages = (
        Message(role="system", content="Generate"),
        Message(role="user", content="Hello"),
    )

    first = service._generate_with_cache(
        messages,
        max_new_tokens=32,
        temperature=0.5,
        use_cache=True,
    )

    second = service._generate_with_cache(
        messages,
        max_new_tokens=32,
        temperature=0.5,
        use_cache=True,
    )

    assert first is second
    assert len(service._cache.audio) == 1
    assert len(fake_engine.calls) == 1


def test_generate_multi_speaker_inserts_pauses_and_reuses_smart_voice(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchaudio")

    service = GenerationService(
        device=torch.device("cpu"),
        voice_library_service=StubVoiceLibrary(),
    )

    fake_engine = FakeEngine(segment_length=80, sample_rate=2_000)
    service._serve_engine = fake_engine

    captured: dict[str, Any] = {}

    working_dir = tmp_path_factory.mktemp("multi_speaker")
    monkeypatch.chdir(working_dir)

    def fake_get_output_path(subdir: str, prefix: str) -> str:
        return (working_dir / f"{prefix}.wav").as_posix()

    def fake_save(path: str, tensor, sample_rate: int) -> None:  # type: ignore[override]
        captured["path"] = path
        captured["tensor"] = tensor.clone()
        captured["sample_rate"] = sample_rate

    monkeypatch.setattr("app.services.generation_service.audio_io.get_output_path", fake_get_output_path)
    monkeypatch.setattr("app.services.generation_service.torchaudio.save", fake_save)
    monkeypatch.setattr("app.services.generation_service.audio_io.robust_file_cleanup", lambda paths: None)
    monkeypatch.setattr("app.services.generation_service.transcribe_audio", lambda path: "transcribed")

    transcript = "\n".join(
        [
            "[SPEAKER0] Hello there!",
            "[SPEAKER1] Doing well, thanks!",
            "[SPEAKER0] Glad to hear it.",
            "[SPEAKER1] Catch you soon!",
        ]
    )

    output_path = service.generate_multi_speaker(
        transcript,
        voice_method="Smart Voice",
        uploaded_voices=[],
        predefined_voices=[],
        temperature=0.6,
        max_new_tokens=64,
        seed=123,
        scene_description="",
    auto_format=False,
        speaker_pause_duration=0.25,
    )

    assert "tensor" in captured, "Expected torchaudio.save to be invoked"
    assert output_path == captured["path"]

    tensor = captured["tensor"]
    sample_rate = captured["sample_rate"]
    audio_array = tensor.squeeze(0).numpy()

    segment_length = fake_engine.calls[0].response.audio.size
    expected_pause = int(0.25 * sample_rate)

    cursor = 0
    saw_pause = False
    for index, call in enumerate(fake_engine.calls):
        segment = audio_array[cursor : cursor + segment_length]
        expected_segment = call.response.audio
        assert np.array_equal(segment, expected_segment)
        cursor += segment_length

        pause_slice = audio_array[cursor : cursor + expected_pause]
        if pause_slice.size == expected_pause and np.all(pause_slice == 0):
            saw_pause = True
            cursor += expected_pause

    assert saw_pause, "Expected at least one pause between speakers"

    # Smart Voice mode should reuse the first speaker's reference on later appearances
    third_call_messages = fake_engine.calls[2].kwargs["chat_ml_sample"].messages
    assert any(
        isinstance(message.content, AudioContent)
        for message in third_call_messages
        if hasattr(message, "content")
    )
