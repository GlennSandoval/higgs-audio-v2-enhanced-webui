@echo off
echo ========================================
echo   Higgs Audio - Cache Migration Tool
echo ========================================
echo.
echo This script will copy existing Hugging Face cached models
echo to the local project cache directory to avoid redownloading.
echo.

REM Define cache locations
set "LOCAL_CACHE=%~dp0cache\huggingface\hub"
set "DEFAULT_CACHE=%USERPROFILE%\.cache\huggingface\hub"
set "ALT_CACHE1=E:\HF_HOME\hub"
set "ALT_CACHE2=%USERPROFILE%\.cache\torch\transformers"

REM Model names to look for
set "MODEL1=models--bosonai--higgs-audio-v2-generation-3B-base"
set "MODEL2=models--bosonai--higgs-audio-v2-tokenizer"

echo [INFO] Local cache directory: %LOCAL_CACHE%
echo [INFO] Searching for existing models...
echo.

REM Create local cache directory
if not exist "%LOCAL_CACHE%" (
    echo [INFO] Creating local cache directory...
    mkdir "%LOCAL_CACHE%" 2>nul
)

REM Function to copy models from a source to local cache
call :copy_models "%DEFAULT_CACHE%" "default cache"
call :copy_models "%ALT_CACHE1%" "E: drive cache"
call :copy_models "%ALT_CACHE2%" "legacy torch cache"

echo.
echo [INFO] Cache migration complete!
echo [INFO] Run your launch script normally - models should not redownload.
pause
goto :eof

:copy_models
set "SOURCE_DIR=%~1"
set "SOURCE_NAME=%~2"

if exist "%SOURCE_DIR%\%MODEL1%" (
    echo [FOUND] %MODEL1% in %SOURCE_NAME%
    if not exist "%LOCAL_CACHE%\%MODEL1%" (
        echo [COPY] Copying %MODEL1%...
        xcopy "%SOURCE_DIR%\%MODEL1%" "%LOCAL_CACHE%\%MODEL1%" /E /I /H /Q
        if %ERRORLEVEL% equ 0 (
            echo [SUCCESS] %MODEL1% copied successfully
        ) else (
            echo [ERROR] Failed to copy %MODEL1%
        )
    ) else (
        echo [SKIP] %MODEL1% already exists in local cache
    )
    echo.
)

if exist "%SOURCE_DIR%\%MODEL2%" (
    echo [FOUND] %MODEL2% in %SOURCE_NAME%
    if not exist "%LOCAL_CACHE%\%MODEL2%" (
        echo [COPY] Copying %MODEL2%...
        xcopy "%SOURCE_DIR%\%MODEL2%" "%LOCAL_CACHE%\%MODEL2%" /E /I /H /Q
        if %ERRORLEVEL% equ 0 (
            echo [SUCCESS] %MODEL2% copied successfully
        ) else (
            echo [ERROR] Failed to copy %MODEL2%
        )
    ) else (
        echo [SKIP] %MODEL2% already exists in local cache
    )
    echo.
)
goto :eof 