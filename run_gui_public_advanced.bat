@echo off
echo ========================================
echo    Higgs Audio WebUI - Public Launcher
echo ========================================
echo.
echo This will launch the Higgs Audio WebUI with a PUBLIC shareable link.
echo Anyone with the link will be able to access your interface!
echo.
echo [WARNING] Make sure you understand the security implications:
echo - Your interface will be accessible from anywhere on the internet
echo - People can generate audio using your computational resources
echo - The link will be active until you stop the application
echo.

set /p confirm="Are you sure you want to proceed? (y/N): "
if /i not "%confirm%"=="y" (
    echo Operation cancelled.
    pause
    exit /b
)

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
echo [INFO] Starting Higgs Audio WebUI with Public Share...
echo [INFO] Generating public shareable link via Hugging Face...
echo [INFO] This may take a moment...
echo.
echo [TIP] You can customize the launch with these options:
echo   --share                  Enable public sharing
echo   --server-port 7860       Set custom port
echo   --server-name 0.0.0.0    Allow external connections
echo.

python higgs_audio_gradio.py --share --server-name 0.0.0.0

echo.
echo [INFO] Public session ended.
echo [INFO] The shareable link is no longer active.
pause 