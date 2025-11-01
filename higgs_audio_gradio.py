"""Entry point for the Higgs Audio v2 Gradio WebUI."""

from __future__ import annotations

import argparse

from app import config as app_config
from app.gradio import BootstrapConfig
from app.gradio.main import build_controller, format_startup_banner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Higgs Audio v2 Generator WebUI")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link via Hugging Face",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default=app_config.DEFAULT_SERVER_NAME,
        help="Server host address",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=app_config.DEFAULT_SERVER_PORT,
        help="Server port",
    )
    return parser.parse_args()


def main() -> None:
    """Parse CLI arguments, build the application, and launch the Gradio demo."""
    args = _parse_args()

    controller = build_controller(BootstrapConfig.from_environment())
    print(format_startup_banner(controller))

    if args.share:
        print("🌐 Creating public shareable link via Hugging Face...")
        print(
            "⚠️  Warning: Your interface will be publicly accessible to anyone with the link!"
        )

    controller.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
    )


if __name__ == "__main__":
    main()
