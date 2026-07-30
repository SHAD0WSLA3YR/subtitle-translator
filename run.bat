@echo off
title Real-Time Subtitle Translator
cd /d "%~dp0"

REM Activate virtual environment — create if missing
if not exist ".venv\Scripts\activate.bat" (
    echo [SETUP] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Make sure Python is installed.
        pause
        exit /b 1
    )
    call .venv\Scripts\activate.bat
    echo [SETUP] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo [SETUP] Setup complete.
) else (
    call .venv\Scripts\activate.bat
)

echo Starting Real-Time Japanese to English Subtitle Translator...
echo.
echo NOTE: First startup takes ~30 seconds (model loading). Please wait.
echo Arguments passed: %*
echo.
echo Press Ctrl+C in this window to exit.
echo.

python run.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] The application exited with code %errorlevel%.
    echo Check the output above for details.
    pause
)
