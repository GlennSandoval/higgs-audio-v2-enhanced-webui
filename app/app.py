"""Central application wiring for the Higgs Audio Gradio WebUI."""

from __future__ import annotations

from dataclasses import dataclass

import gradio as gr
import torch

from app import generation, startup, voice_library
from app.ui import build_demo


@dataclass(frozen=True)
class AppContext:
    """Container for the core application services and constructed demo."""

    device: torch.device
    whisper_available: bool
    voice_library_service: voice_library.VoiceLibrary
    generation_service: generation.GenerationService
    audio_dependency_report: dict[str, bool]
    demo: gr.Blocks


def create_app() -> AppContext:
    """
    Configure the environment, initialize services, and build the Gradio demo.

    Returns:
        An ``AppContext`` with the resolved device, initialized services, dependency
        report, and ready-to-launch Gradio ``Blocks`` instance.
    """
    startup.configure_environment()
    device = startup.select_device()
    startup.ensure_output_directories()
    dependency_report = startup.check_audio_dependencies()

    voice_library_service = voice_library.create_default_voice_library()
    generation_service = generation.create_generation_service(
        device=device,
        voice_library_service=voice_library_service,
    )

    demo = build_demo(
        generation_service=generation_service,
        voice_library_service=voice_library_service,
        whisper_available=voice_library.WHISPER_AVAILABLE,
    )

    return AppContext(
        device=device,
        whisper_available=voice_library.WHISPER_AVAILABLE,
        voice_library_service=voice_library_service,
        generation_service=generation_service,
        audio_dependency_report=dependency_report,
        demo=demo,
    )


def create_demo() -> gr.Blocks:
    """Convenience helper that returns a ready-to-launch Gradio ``Blocks``."""
    return create_app().demo
