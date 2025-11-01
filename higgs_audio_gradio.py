"""Entry point for the Higgs Audio v2 Gradio WebUI."""

from __future__ import annotations

import argparse

from app import AppContext, create_app


def _format_dependency_report(report: dict[str, bool]) -> str:
    """Return a human-friendly dependency availability summary."""
    lines = []
    for name, available in sorted(report.items()):
        status = "✅" if available else "⚠️"
        lines.append(f"   {status} {name}")
    return "\n".join(lines)


def main() -> None:
    """Parse CLI arguments, build the application, and launch the Gradio demo."""
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

    app_context: AppContext = create_app()
    demo = app_context.demo
    dependency_report = _format_dependency_report(app_context.audio_dependency_report)

    print("🚀 Starting Higgs Audio v2 Generator...")
    print(
        "✨ Features: Voice Cloning, Multi-Speaker, Caching, Auto-Transcription, Enhanced Audio Processing"
    )
    if dependency_report:
        print("🔧 Dependency check:")
        print(dependency_report)

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
