# Higgs Audio WebUI Setup Instructions

## Quick Start

### Option 1: Automated Setup (Recommended)
1. Double-click `setup_venv.bat` to automatically set up the virtual environment and install all dependencies
2. The script will automatically install CUDA-enabled PyTorch and verify GPU support
3. Once setup is complete, double-click `run_gui.bat` to launch the application

### Option 2: Manual Setup
If you prefer to set up manually:

```bash
# Create virtual environment
python -m venv higgs_audio_env

# Activate virtual environment (Windows)
higgs_audio_env\Scripts\activate

# Install dependencies in the correct order
python -m pip install --upgrade pip

# Remove any existing PyTorch
python -m pip uninstall torch torchvision torchaudio -y

# Install CUDA-enabled PyTorch
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Install other dependencies
python -m pip install "transformers>=4.45.1,<4.47.0"
python -m pip install -r requirements.txt
python -m pip install gradio faster-whisper

# Run the application
python higgs_audio_gradio.py
```

## Requirements

- Python 3.10 or higher
- **NVIDIA GPU with CUDA support** (RTX 20/30/40 series recommended)
- CUDA 12.1 compatible drivers
- Windows 10/11
- At least 8GB GPU VRAM (16GB+ recommended for best performance)

## Performance Notes

- **With CUDA (GPU)**: Fast inference, real-time generation possible
- **Without CUDA (CPU)**: Much slower, suitable for testing only
- **Your RTX 3090**: Excellent performance with 24GB VRAM - perfect for this application!

## Features

- **Basic Generation**: Generate speech from text with predefined voices
- **Voice Cloning**: Clone any voice by uploading a sample
- **Long-form Generation**: Generate long audio content with smart chunking
- **Multi-Speaker Generation**: Create conversations with multiple speakers
- **Voice Library**: Save and manage your voice samples

## Troubleshooting

### Common Issues

1. **Import Error**: If you get import errors, make sure you're using the virtual environment:
   ```bash
   higgs_audio_env\Scripts\activate
   python higgs_audio_gradio.py
   ```

2. **CUDA Issues**: Verify CUDA is working:
   ```bash
   higgs_audio_env\Scripts\activate
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
   ```

3. **Wrong PyTorch Version**: If you have CPU-only PyTorch, reinstall with CUDA:
   ```bash
   higgs_audio_env\Scripts\activate
   python -m pip uninstall torch torchvision torchaudio -y
   python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Missing Dependencies**: Run the setup script again:
   ```bash
   setup_venv.bat
   ```

### GPU Memory Issues

If you encounter out-of-memory errors:
- Close other GPU-intensive applications
- Reduce batch sizes in the application
- Use shorter text inputs for generation

### Getting Help

- Check the console output for detailed error messages
- Make sure all dependencies are installed in the virtual environment
- Ensure you have sufficient disk space (several GB for models)
- Verify CUDA drivers are up to date

## Usage

1. Launch the application using `run_gui.bat`
2. Open your browser to the displayed URL (usually `http://127.0.0.1:7860`)
3. Start with the "Basic Generation" tab to test the setup
4. Upload voice samples in the "Voice Cloning" tab for personalized speech
5. Use the "Voice Library" to save frequently used voices

The application will automatically download required models on first use and utilize your RTX 3090 for fast inference. 