"""Startup utilities for the Higgs Audio Gradio app."""

from __future__ import annotations

import os

import torch

from app import config
from app.audio.capabilities import dependency_report as build_dependency_report
from app.audio.capabilities import describe_missing_dependencies, has_ffmpeg


def configure_environment() -> None:
    """Set required environment variables before the app starts."""
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = config.HF_HUB_ENABLE_HF_TRANSFER
    # Disable tokenizers parallelism to avoid fork-related warnings when Gradio
    # launches with multiprocessing after tokenizer initialization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    if has_ffmpeg():
        return True

    print("⚠️ FFmpeg not found. For full audio format support, install FFmpeg:")
    print("   Windows: Download from https://ffmpeg.org/download.html")
    print("   macOS: brew install ffmpeg")
    print("   Linux: sudo apt install ffmpeg")
    return False


def check_audio_dependencies() -> dict[str, bool]:
    """Inspect optional audio dependencies and report their availability."""
    print("🔍 Checking audio processing dependencies...")

    report = build_dependency_report()

    if report.get("torchaudio", False):
        print("✅ torchaudio available")
    else:  # pragma: no cover - torchaudio is required but guard for completeness
        print("❌ torchaudio missing - install it to run the web UI")

    if report.get("pydub", False):
        print("✅ pydub available")
    else:
        print("⚠️ pydub not available - install with: pip install pydub")

    if report.get("scipy", False):
        print("✅ scipy available")
    else:
        print("⚠️ scipy not available - install with: pip install scipy")

    if report.get("ffmpeg", False):
        print("✅ FFmpeg available")
    else:
        print("⚠️ FFmpeg not found. For full audio format support, install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   macOS: brew install ffmpeg")
        print("   Linux: sudo apt install ffmpeg")

    for message in describe_missing_dependencies(report):
        print(f"ℹ️ {message}")

    return report
