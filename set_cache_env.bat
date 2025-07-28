@echo off
echo ========================================
echo   Higgs Audio - Cache Environment Setup
echo ========================================
echo.
echo This script helps configure Hugging Face cache directories
echo to prevent redownloading models that are already cached.
echo.

REM Set cache directory to local project folder for consistency
set HF_HOME=%~dp0cache\huggingface
set HF_HUB_CACHE=%HF_HOME%\hub
set TRANSFORMERS_CACHE=%HF_HUB_CACHE%
set HUGGINGFACE_HUB_CACHE=%HF_HUB_CACHE%

echo [INFO] Setting Hugging Face cache directories:
echo   HF_HOME=%HF_HOME%
echo   HF_HUB_CACHE=%HF_HUB_CACHE%
echo   TRANSFORMERS_CACHE=%TRANSFORMERS_CACHE%
echo.

REM Create cache directories if they don't exist
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%HF_HUB_CACHE%" mkdir "%HF_HUB_CACHE%"

echo [INFO] Cache directories created/verified.
echo [INFO] Models will be cached locally to avoid redownloading.
echo.

REM Check if models already exist in common cache locations
set "DEFAULT_CACHE=%USERPROFILE%\.cache\huggingface\hub"
set "MODEL_NAME=models--bosonai--higgs-audio-v2-generation-3B-base"

if exist "%DEFAULT_CACHE%\%MODEL_NAME%" (
    echo [FOUND] Models found in default cache: %DEFAULT_CACHE%
    echo [TIP] You can copy models from default cache to local cache:
    echo   xcopy "%DEFAULT_CACHE%\%MODEL_NAME%" "%HF_HUB_CACHE%\%MODEL_NAME%" /E /I /H
    echo.
)

if exist "%HF_HUB_CACHE%\%MODEL_NAME%" (
    echo [FOUND] Models already exist in local cache: %HF_HUB_CACHE%
    echo [INFO] No redownloading should be needed.
) else (
    echo [NOTICE] Models not found in local cache.
    echo [INFO] Models will be downloaded on first run.
)

echo.
echo Environment variables set for current session.
echo Add these to your system environment variables for permanent effect.
pause 