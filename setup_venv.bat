@echo off
REM Setup script for Higgs Audio WebUI virtual environment

echo [INFO] Setting up Higgs Audio WebUI virtual environment...
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Check if Python is available
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Python is not available in PATH.
    echo Please install Python and add it to your PATH.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "higgs_audio_env" (
    echo [INFO] Creating virtual environment...
    python -m venv higgs_audio_env
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [INFO] Virtual environment created successfully.
) else (
    echo [INFO] Virtual environment already exists.
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call higgs_audio_env\Scripts\activate.bat

REM Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM Uninstall any existing PyTorch to ensure clean CUDA installation
echo [INFO] Removing any existing PyTorch installations...
python -m pip uninstall torch torchvision torchaudio -y >nul 2>&1

REM Install specific transformers version first
echo [INFO] Installing compatible transformers version...
python -m pip install "transformers>=4.45.1,<4.47.0"

REM Install PyTorch with CUDA support (CUDA 12.1)
echo [INFO] Installing PyTorch with CUDA 12.1 support...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

REM Verify CUDA installation
echo [INFO] Verifying CUDA installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch version:', torch.__version__)"

REM Install requirements
echo [INFO] Installing requirements...
python -m pip install -r requirements.txt

REM Install additional dependencies
echo [INFO] Installing Gradio and Whisper...
python -m pip install gradio faster-whisper

echo.
echo [SUCCESS] Setup complete!
echo [INFO] CUDA Status:
python -c "import torch; print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU Count: {torch.cuda.device_count()}'); print(f'  GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.
echo [INFO] You can now run the application using run_gui.bat
echo.
pause 