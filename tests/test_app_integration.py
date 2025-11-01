"""Integration tests ensuring the Gradio app wiring stays intact."""

from __future__ import annotations

import gradio as gr
import pytest

from app import config
from gradio_app import main


def _dependency_report_stub() -> dict[str, bool]:
    return {"torchaudio": True, "pydub": True, "scipy": True, "ffmpeg": True}


class StubVoiceLibrary:
    def __init__(self) -> None:
        self.voices = ["alpha"]
        self.list_all_calls = 0

    def list_all_available_voices(self) -> list[str]:
        self.list_all_calls += 1
        return [config.SMART_VOICE_LABEL, f"{config.LIBRARY_VOICE_PREFIX}alpha"]

    def list_voice_library_voices(self) -> list[str]:
        return self.voices

    def save_voice(self, *args, **kwargs):  # pragma: no cover - callback stub
        return "saved"

    def save_voice_config(self, *args, **kwargs) -> bool:  # pragma: no cover
        return True

    def load_voice_config(self, voice_name: str) -> dict:
        return dict(config.DEFAULT_VOICE_CONFIG)

    def get_voice_path(self, selection: str | None) -> str | None:  # pragma: no cover
        return None

    def robust_txt_path_creation(self, *args, **kwargs) -> str:  # pragma: no cover
        return "voice.txt"

    def delete_voice(self, voice_name: str) -> str:  # pragma: no cover
        return "deleted"


class StubGenerationService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def _record(self, name: str) -> str:
        self.calls.append(name)
        return f"/{name}.wav"

    def generate_basic(self, *args, **kwargs) -> str:
        return self._record("basic")

    def generate_voice_clone(self, *args, **kwargs) -> str:
        return self._record("voice_clone")

    def generate_voice_clone_alternative(self, *args, **kwargs) -> str:
        return self._record("voice_clone_alternative")

    def generate_longform(self, *args, **kwargs) -> str:
        return self._record("longform")

    def generate_multi_speaker(self, *args, **kwargs) -> str:
        return self._record("multi_speaker")

    def generate_dynamic_multi_speaker(self, *args, **kwargs) -> str:
        return self._record("dynamic_multi_speaker")


def test_create_app_wires_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")

    stub_voice_library = StubVoiceLibrary()
    stub_generation = StubGenerationService()

    monkeypatch.setattr("app.startup.configure_environment", lambda: None)
    monkeypatch.setattr("app.startup.select_device", lambda: torch.device("cpu"))
    monkeypatch.setattr("app.startup.ensure_output_directories", lambda: None)
    monkeypatch.setattr("app.startup.check_audio_dependencies", lambda: _dependency_report_stub())
    monkeypatch.setattr(
        "app.audio.capabilities.describe_missing_dependencies",
        lambda report: [],
    )
    monkeypatch.setattr("app.app.create_default_voice_library", lambda: stub_voice_library)
    monkeypatch.setattr(
        "app.app.create_generation_service",
        lambda device, voice_library_service: stub_generation,
    )

    blocks = main.create_app()
    assert isinstance(blocks, gr.Blocks)

    registered_functions = [entry.fn for entry in blocks.fns.values()]

    basic_callback = next(
        (
            fn
            for fn in registered_functions
            if getattr(fn, "__self__", None) is stub_generation
            and fn.__name__ == "generate_basic"
        ),
        None,
    )
    assert basic_callback is not None

    voice_clone_wrapper = next(
        (fn for fn in registered_functions if fn.__name__ == "_handle_voice_clone_generation"),
        None,
    )
    assert voice_clone_wrapper is not None
    assert any(
        getattr(cell, "cell_contents", None) is stub_generation
        for cell in (voice_clone_wrapper.__closure__ or [])
    )

    multi_speaker_wrapper = next(
        (fn for fn in registered_functions if fn.__name__ == "generate_dynamic_multi_speaker"),
        None,
    )
    assert multi_speaker_wrapper is not None
    assert any(
        getattr(cell, "cell_contents", None) is stub_generation
        for cell in (multi_speaker_wrapper.__closure__ or [])
    )

    assert stub_voice_library.list_all_calls >= 1
