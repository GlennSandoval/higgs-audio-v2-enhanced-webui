# Higgs Audio WebUI Setup Instructions

## Setup
Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment and install all dependencies:
```bash
uv sync
```

This will:
- Create a `.venv` virtual environment 
- Install all dependencies from `pyproject.toml`
- Generate a `uv.lock` file for reproducible installations

Verify the setup (optional):
```bash
uv run python verify_setup.py
```

Run the application:
```bash
uv run python higgs_audio_gradio.py
```

### Alternative: Activate environment manually
```bash
source .venv/bin/activate  # On macOS/Linux
python higgs_audio_gradio.py
```

## Features

- **Basic Generation**: Generate speech from text with predefined voices
- **Voice Cloning**: Clone any voice by uploading a sample
- **Long-form Generation**: Generate long audio content with smart chunking
- **Multi-Speaker Generation**: Create conversations with multiple speakers
- **Voice Library**: Save and manage your voice samples
