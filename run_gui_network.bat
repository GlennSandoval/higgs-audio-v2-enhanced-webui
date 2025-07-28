@echo off
echo ========================================
echo   Higgs Audio WebUI - Network Launcher
echo ========================================
echo.
echo This will launch the Higgs Audio WebUI accessible on your local network.
echo Other devices on your network will be able to access it via your IP address.
echo This does NOT create a public internet link.
echo.

echo [INFO] Setting up cache environment...
REM Set consistent cache directories to prevent redownloading models
set HF_HOME=%~dp0cache\huggingface
set HF_HUB_CACHE=%HF_HOME%\hub
set TRANSFORMERS_CACHE=%HF_HUB_CACHE%
set HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%

echo [INFO] Activating virtual environment...
call higgs_audio_env\Scripts\activate.bat

echo.
echo [INFO] Starting Higgs Audio WebUI for local network access...
echo [INFO] The interface will be accessible to devices on your local network.
echo [INFO] Look for the "Running on" URLs in the output below.
echo [INFO] Press Ctrl+C to stop the application.
echo.

python higgs_audio_gradio.py --server-name 0.0.0.0 --server-port 7860

echo.
echo [INFO] Network session ended.
pause 