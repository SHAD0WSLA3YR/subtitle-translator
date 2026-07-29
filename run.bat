@echo off
title Real-Time Subtitle Translator
cd /d "%~dp0"

REM Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found at .venv\
    echo Run: python -m venv .venv
    pause
    exit /b 1
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
