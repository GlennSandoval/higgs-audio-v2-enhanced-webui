"""Entry point for the Higgs Audio v2 Gradio WebUI."""

from __future__ import annotations

import argparse

from app import generation, startup, voice_library as voice_lib
from app.ui import build_demo

startup.configure_environment()
device = startup.select_device()
startup.ensure_output_directories()
startup.check_audio_dependencies()

voice_library_service = voice_lib.create_default_voice_library()
generation_service = generation.create_generation_service(
    device, voice_library_service
)
WHISPER_AVAILABLE = voice_lib.WHISPER_AVAILABLE

demo = build_demo(
    generation_service=generation_service,
    voice_library_service=voice_library_service,
    whisper_available=WHISPER_AVAILABLE,
)


def main() -> None:
    """Parse CLI arguments and launch the Gradio demo."""
    parser = argparse.ArgumentParser(description="Higgs Audio v2 Generator WebUI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link via Hugging Face",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server host address",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=7860,
        help="Server port",
    )
    args = parser.parse_args()

    print("🚀 Starting Higgs Audio v2 Generator...")
    print(
        "✨ Features: Voice Cloning, Multi-Speaker, Caching, Auto-Transcription, Enhanced Audio Processing"
    )

    if args.share:
        print("🌐 Creating public shareable link via Hugging Face...")
        print(
            "⚠️  Warning: Your interface will be publicly accessible to anyone with the link!"
        )

    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
    )


if __name__ == "__main__":
    main()
