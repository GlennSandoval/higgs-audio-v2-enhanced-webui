@echo off
echo [INFO] Launching Higgs Audio WebUI with Public Share...
echo [INFO] This will create a public URL that anyone can access.
echo [INFO] The interface will be available both locally and via a shareable link.
echo.

echo [INFO] Setting up cache environment...
REM Set consistent cache directories to prevent redownloading models
set HF_HOME=%~dp0cache\huggingface
set HF_HUB_CACHE=%HF_HOME%\hub
set TRANSFORMERS_CACHE=%HF_HUB_CACHE%
set HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%

echo [INFO] Activating virtual environment...
call higgs_audio_env\Scripts\activate.bat

echo [INFO] Starting Higgs Audio WebUI with Hugging Face Share...
echo [INFO] A public shareable link will be generated shortly.
echo [INFO] Press Ctrl+C to stop the application.
echo.

python higgs_audio_gradio.py --share

echo.
echo [INFO] Application stopped.
pause 