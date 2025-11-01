"""Primary Gradio application factory functions."""

from __future__ import annotations

import os
from typing import Iterable

import gradio as gr

from gradio_app.config import BootstrapConfig
from gradio_app.controllers import AppController


def _format_dependency_report(report: Iterable[tuple[str, bool]]) -> str:
    lines = []
    for name, available in report:
        status = "✅" if available else "⚠️"
        lines.append(f"   {status} {name}")
    return "\n".join(lines)


def build_controller(config: BootstrapConfig | None = None) -> AppController:
    """Instantiate an :class:`AppController` using the provided configuration."""
    resolved_config = config or BootstrapConfig.from_environment()
    _apply_bootstrap_environment(resolved_config)
    return AppController.bootstrap(resolved_config)


def create_app(config: BootstrapConfig | None = None) -> gr.Blocks:
    """Return the fully wired Gradio ``Blocks`` instance for the UI."""
    controller = build_controller(config)
    return controller.demo


def format_startup_banner(controller: AppController) -> str:
    """Render a startup banner summarizing key runtime information."""
    lines = [
        "🚀 Starting Higgs Audio v2 Generator...",
        "✨ Features: Voice Cloning, Multi-Speaker, Caching, Auto-Transcription, Enhanced Audio Processing",
    ]
    dependency_report = _format_dependency_report(
        sorted(controller.dependency_report.items())
    )
    if dependency_report:
        lines.append("🔧 Dependency check:")
        lines.append(dependency_report)
    return "\n".join(lines)


def _apply_bootstrap_environment(config: BootstrapConfig) -> None:
    if config.hf_token and not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.hf_token