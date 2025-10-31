#!/bin/bash
# Launch script for Higgs Audio v2 with MacPorts FFmpeg support
# This script sets the necessary environment variables for torchcodec to find FFmpeg libraries

# Add MacPorts library path for FFmpeg
export DYLD_LIBRARY_PATH="/opt/local/lib:$DYLD_LIBRARY_PATH"

# Enable faster Hugging Face downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

# Run the Gradio application
uv run python higgs_audio_gradio.py
