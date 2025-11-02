"""Primary Gradio application factory functions."""

from __future__ import annotations

import os
from typing import Iterable

import gradio as gr

from app.gradio.config import BootstrapConfig
from app.gradio.controllers import AppController


def build_controller(config: BootstrapConfig | None = None) -> AppController:
    """Instantiate an :class:`AppController` using the provided configuration."""
    resolved_config = config or BootstrapConfig.from_environment()
    _apply_bootstrap_environment(resolved_config)
    return AppController.bootstrap(resolved_config)


def create_app(config: BootstrapConfig | None = None) -> gr.Blocks:
    """Return the fully wired Gradio ``Blocks`` instance for the UI."""
    controller = build_controller(config)
    return controller.ui_blocks


def _apply_bootstrap_environment(config: BootstrapConfig) -> None:
    if config.hf_token and not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = config.hf_token
