"""Gradio application bootstrap utilities for the Higgs Audio WebUI."""

from .config import BootstrapConfig

__all__ = [
	"BootstrapConfig",
	"AppController",
	"build_controller",
	"create_app",
	"format_startup_banner",
]


def __getattr__(name: str):  # pragma: no cover - thin import shim
	if name == "AppController":
		from .controllers import AppController

		return AppController
	if name in {"build_controller", "create_app", "format_startup_banner"}:
		from . import main as _main

		return getattr(_main, name)
	raise AttributeError(f"module 'gradio_app' has no attribute {name!r}")
