"""Controller layer that wires application services to the Gradio UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

from app.app import AppContext, create_app_context
from gradio_app.config import BootstrapConfig


@dataclass(frozen=True)
class AppController:
    """Coordinates Gradio demo creation and exposes dependency metadata."""

    context: AppContext

    @classmethod
    def bootstrap(cls, config: BootstrapConfig | None = None) -> "AppController":
        """Build the full application stack using the provided bootstrap config."""
        app_context = create_app_context(config)
        return cls(context=app_context)

    @property
    def demo(self) -> gr.Blocks:
        """Return the configured Gradio ``Blocks`` instance."""
        return self.context.demo

    @property
    def dependency_report(self) -> dict[str, bool]:
        """Expose the audio dependency availability report."""
        return dict(self.context.audio_dependency_report)

    @property
    def whisper_available(self) -> bool:
        """Indicate whether Whisper transcription is available."""
        return self.context.whisper_available

    def launch(self, *, share: bool, server_name: str, server_port: int) -> None:
        """Launch the underlying Gradio demo with the given server options."""
        self.demo.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_error=True,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return a metadata snapshot useful for debugging and tests."""
        return {
            "device": str(self.context.device),
            "whisper_available": self.whisper_available,
            "dependency_report": self.dependency_report,
        }
