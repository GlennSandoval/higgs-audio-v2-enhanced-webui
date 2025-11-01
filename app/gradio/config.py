"""Bootstrap configuration for the Gradio application wiring."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app import config as app_config


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration values required to bootstrap the Gradio UI."""

    hf_token: str | None = None
    default_temperature: float = app_config.DEFAULT_TEMPERATURE
    cache_root: Path = Path(".cache")
    audio_cache_dir: Path | None = None
    token_cache_dir: Path | None = None

    @classmethod
    def from_environment(cls) -> "BootstrapConfig":
        """Create a config instance by inspecting the current environment."""
        token = (
            os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            or os.environ.get("HF_TOKEN")
            or os.environ.get("HF_API_TOKEN")
        )
        return cls(hf_token=token)

    def resolved_audio_cache_dir(self) -> Path:
        """Return the directory where audio cache artefacts should be stored."""
        return (self.audio_cache_dir or self.cache_root / "audio").expanduser()

    def resolved_token_cache_dir(self) -> Path:
        """Return the directory where token cache artefacts should be stored."""
        return (self.token_cache_dir or self.cache_root / "tokens").expanduser()
