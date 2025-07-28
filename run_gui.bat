@echo off
REM Batch script to launch Higgs Audio WebUI with virtual environment

REM Change to the script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "higgs_audio_env\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please run the following commands to set up the environment:
    echo   python -m venv higgs_audio_env
    echo   higgs_audio_env\Scripts\activate
    echo   python -m pip install -r requirements.txt
    echo   python -m pip install gradio faster-whisper
    pause
    exit /b 1
)

echo [INFO] Setting up cache environment...
REM Set consistent cache directories to prevent redownloading models
set HF_HOME=%~dp0cache\huggingface
set HF_HUB_CACHE=%HF_HOME%\hub
set TRANSFORMERS_CACHE=%HF_HUB_CACHE%
set HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%

echo [INFO] Activating virtual environment...
call higgs_audio_env\Scripts\activate.bat

REM Check if gradio is installed in the virtual environment
python -c "import gradio" >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] Gradio is not installed in the virtual environment.
    echo Installing required dependencies...
    python -m pip install -r requirements.txt
    python -m pip install gradio faster-whisper
    IF ERRORLEVEL 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
)

echo [INFO] Starting Higgs Audio WebUI...
echo [INFO] The web interface will open in your browser shortly.
echo [INFO] Press Ctrl+C to stop the application.
echo.

REM Run the Gradio app
python higgs_audio_gradio.py

REM Keep the window open after execution
echo.
echo [INFO] The app has exited. If you see errors above, please review them.
pause