"""Application package for the Higgs Audio Gradio UI."""

from app.app import AppContext, create_app_context, create_demo

# Backwards compatibility shim for older imports
create_app = create_app_context


__all__ = ["AppContext", "create_app_context", "create_demo", "create_app"]
