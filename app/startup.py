"""Startup utilities for the Higgs Audio Gradio app."""

from __future__ import annotations

import os

import torch

from app import config


def configure_environment() -> None:
    """Set required environment variables before the app starts."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = config.HF_HUB_ENABLE_HF_TRANSFER


def select_device() -> torch.device:
    """Determine the torch device to use for inference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def initialize_caches() -> tuple[dict, dict]:
    """Create the audio and token cache structures."""
    return {}, {}


def ensure_output_directories() -> None:
    """Create output directories if they do not exist yet."""
    for dir_path in config.OUTPUT_DIRECTORIES:
        os.makedirs(dir_path, exist_ok=True)


def install_ffmpeg_if_needed() -> bool:
    """Check if ffmpeg is available and provide installation guidance."""
    from shutil import which

    if which("ffmpeg") is None:
        print("⚠️ FFmpeg not found. For full audio format support, install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   macOS: brew install ffmpeg")
        print("   Linux: sudo apt install ffmpeg")
        return False

    return True


def check_audio_dependencies() -> dict[str, bool]:
    """Inspect optional audio dependencies and report their availability."""
    print("🔍 Checking audio processing dependencies...")

    dependencies = {
        "torchaudio": True,
        "pydub": False,
        "scipy": False,
        "ffmpeg": False,
    }

    try:
        import pydub  # noqa: F401

        dependencies["pydub"] = True
        print("✅ pydub available")
    except ImportError:
        print("⚠️ pydub not available - install with: pip install pydub")

    try:
        import scipy.io  # noqa: F401

        dependencies["scipy"] = True
        print("✅ scipy available")
    except ImportError:
        print("⚠️ scipy not available - install with: pip install scipy")

    dependencies["ffmpeg"] = install_ffmpeg_if_needed()

    return dependencies
