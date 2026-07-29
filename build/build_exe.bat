@echo off
REM Build a portable Windows executable with PyInstaller.
REM Run from the repo root:  build\build_exe.bat
REM
REM Notes:
REM   - First build downloads/caches Whisper weights separately at runtime.
REM   - GPU (CUDA) builds require the matching CUDA toolkit on the target machine.
REM   - For max compatibility, set model.device: cpu in config.yaml before building.

setlocal
cd /d "%~dp0\.."

if not exist "dist" mkdir dist
if not exist "build\work" mkdir build\work

rem PyInstaller resolves --add-data relative to the spec file directory
rem when --specpath is used, so ensure config.yaml is in build/
if not exist "build\config.yaml" copy config.yaml build\

python -m pip install -r requirements-dev.txt
if errorlevel 1 exit /b 1

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name "SubtitleTranslator" ^
  --windowed ^
  --onefile ^
  --icon "assets\icon.ico" ^
  --add-data "config.yaml;." ^
  --hidden-import faster_whisper ^
  --hidden-import ctranslate2 ^
  --hidden-import silero_vad ^
  --hidden-import pyaudiowpatch ^
  --hidden-import onnxruntime ^
  --collect-all faster_whisper ^
  --collect-all silero_vad ^
  --collect-all onnxruntime ^
  --workpath "build\work" ^
  --distpath "dist" ^
  --specpath "build" ^
  run.py

if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo.
echo Built: dist\SubtitleTranslator.exe
echo Copy config.yaml next to the exe if you want to edit settings without rebuilding.
echo First launch will download the Whisper model (~500MB for "small").
endlocal
