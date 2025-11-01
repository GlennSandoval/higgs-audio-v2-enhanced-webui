"""Central application wiring for the Higgs Audio Gradio WebUI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gradio as gr
import torch

from app import startup
from app.services import (GenerationService, VoiceLibrary,
                          create_default_voice_library,
                          create_generation_service)
from app.services.voice_service import WHISPER_AVAILABLE
from app.ui import build_demo

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from gradio_app.config import BootstrapConfig


@dataclass(frozen=True)
class AppContext:
    """Container for the core application services and constructed demo."""

    device: torch.device
    whisper_available: bool
    voice_library_service: VoiceLibrary
    generation_service: GenerationService
    audio_dependency_report: dict[str, bool]
    demo: gr.Blocks
    bootstrap_config: "BootstrapConfig"


def create_app_context(
    bootstrap_config: "BootstrapConfig" | None = None,
) -> AppContext:
    """
    Configure the environment, initialize services, and build the Gradio demo.

    Returns:
        An ``AppContext`` with the resolved device, initialized services, dependency
        report, and ready-to-launch Gradio ``Blocks`` instance.
    """
    if bootstrap_config is None:
        from gradio_app.config import BootstrapConfig as _BootstrapConfig

        resolved_config = _BootstrapConfig.from_environment()
    else:
        resolved_config = bootstrap_config

    startup.configure_environment()
    if resolved_config.hf_token and not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = resolved_config.hf_token
    device = startup.select_device()
    startup.ensure_output_directories()
    dependency_report = startup.check_audio_dependencies()

    voice_library_service = create_default_voice_library()
    generation_service = create_generation_service(
        device=device,
        voice_library_service=voice_library_service,
    )

    demo = build_demo(
        generation_service=generation_service,
        voice_library_service=voice_library_service,
        whisper_available=WHISPER_AVAILABLE,
    )

    return AppContext(
        device=device,
        whisper_available=WHISPER_AVAILABLE,
        voice_library_service=voice_library_service,
        generation_service=generation_service,
        audio_dependency_report=dependency_report,
        demo=demo,
        bootstrap_config=resolved_config,
    )


def create_demo() -> gr.Blocks:
    """Convenience helper that returns a ready-to-launch Gradio ``Blocks``."""
    return create_app_context().demo
