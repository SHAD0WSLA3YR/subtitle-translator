# Phase 8-9: Desktop App Polish, Tray Icon & Distribution

## Overview

Transform the CLI-based Python script into a proper desktop application with system tray integration, runtime controls (on/off toggle, language selection), session history, and a distributable package for GitHub releases.

**Current State:** Bare pipeline that starts from terminal, shows an overlay, and runs until Ctrl+C. No runtime controls, no tray icon, no GUI settings access.

**Target State:** Full desktop app with system tray icon, right-click menu, pause/resume, language picker, session history, and a one-click installer `.exe` for GitHub releases.

---

## Architecture Changes

```
[Before]
run.bat → main.py (SubtitleApp) → overlay.show() → app.exec_()

[After]
run.bat → main.py (SubtitleApp) → 
  ├── SystemTrayIcon (QSystemTrayIcon) ← right-click menu
  ├── Overlay (frameless subtitle window) 
  ├── SettingsDialog (runtime language, appearance)
  ├── HistoryManager (SQLite session log)
  └── PipelineController (pause/resume state machine)
```

### New/Modified Files

```
translate/
├── src/
│   ├── main.py                 # MODIFIED: Add tray icon wiring, pause/resume
│   ├── ui/
│   │   ├── overlay.py           # MODIFIED: Accept runtime config changes
│   │   ├── settings.py          # MODIFIED: Add source/target language combos
│   │   ├── tray.py              # NEW: QSystemTrayIcon + context menu
│   │   └── history_dialog.py    # NEW: Session history viewer
│   ├── core/
│   │   ├── pipeline.py          # NEW: Pipeline state machine (pause/resume)
│   │   └── history.py           # NEW: SQLite session history manager
│   ├── audio/
│   │   ├── capture.py           # MODIFIED: Support runtime device change
│   │   └── vad.py               # Keep as-is
│   ├── stt/
│   │   ├── whisper_stt.py       # MODIFIED: Support runtime language change
│   │   └── processor.py         # MODIFIED: Support runtime language change
│   ├── llm/
│   │   ├── refiner.py           # MODIFIED: Support runtime target language
│   │   └── prompts.py           # MODIFIED: Accept target language param
├── assets/
│   ├── icon.ico                 # NEW: App icon for tray + taskbar
│   └── icon.png                 # NEW: Icon source
├── build/
│   └── build_exe.bat            # NEW: PyInstaller build script
├── requirements.txt             # MODIFIED: Add pyinstaller (dev dep)
├── run.py                       # MODIFIED: CLI arg updates
├── run.bat                      # KEEP
├── PLAN.md                      # KEEP (original plan)
└── PLAN-APP.md                  # THIS FILE
```

---

## Phase 8: System Tray, Runtime Controls & History

**Goal:** Add system tray icon with right-click menu (on/off, settings, quit), pipeline pause/resume, source/target language selection, and session history tracking.

### Estimated Effort: Large (~400-600 lines new code)

---

### Task 8.1: Create System Tray Icon + Menu (`src/ui/tray.py`)

**Requirement:** A tray icon in the Windows notification area that persists even when the overlay is hidden. Right-click shows a context menu with all app controls.

**Implementation:**

```python
class AppTrayIcon(QSystemTrayIcon):
    # Signals
    toggle_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    history_requested = pyqtSignal()
    quit_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIcon(QIcon("assets/icon.png"))
        self.setToolTip("Subtitle Translator")
        
        # Context menu
        menu = QMenu()
        self.toggle_action = menu.addAction("Pause Translation")
        self.toggle_action.triggered.connect(self.toggle_requested)
        menu.addSeparator()
        menu.addAction("Settings...").triggered.connect(self.settings_requested)
        menu.addAction("View History...").triggered.connect(self.history_requested)
        menu.addSeparator()
        menu.addAction("Quit").triggered.connect(self.quit_requested)
        self.setContextMenu(menu)
        
        # Double-click → toggle
        self.activated.connect(self._on_activated)
    
    def set_running(self, running: bool):
        self.toggle_action.setText(
            "Pause Translation" if running else "Resume Translation"
        )
    
    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self.toggle_requested.emit()
```

**Details:**
- Icon file: Create a simple microphone/CC icon (16x16 and 32x32). Use a free icon or generate programmatically with QPainter if no asset file.
- Menu behavior:
  - "Pause Translation" / "Resume Translation" — toggles based on pipeline state
  - "Settings..." — opens `SettingsDialog` modally
  - "View History..." — opens `HistoryDialog`
  - "Quit" — clean shutdown of all components
- Tooltip shows current status: e.g., "Subtitle Translator — Capturing" or "Subtitle Translator — Paused"
- Double-click toggles pause/resume (fastest interaction)

**Verification:**
- [ ] Tray icon appears in notification area on app launch
- [ ] Right-click shows menu with all 4 items
- [ ] Double-click toggles pause/resume
- [ ] Quit button shuts down app cleanly
- [ ] Icon shows correct state (running vs paused)

---

### Task 8.2: Pipeline State Machine (`src/core/pipeline.py`)

**Requirement:** A controllable pipeline that can be paused, resumed, and reconfigured at runtime without restarting the app.

**Implementation:**

```python
class PipelineState(enum.Enum):
    STOPPED = "stopped"
    RUNNING = "running" 
    PAUSED = "paused"
    ERROR = "error"

class PipelineController(QObject):
    state_changed = pyqtSignal(PipelineState)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, capture, vad, processor, overlay, refiner=None):
        self._state = PipelineState.STOPPED
        self._capture = capture
        self._vad = vad
        self._processor = processor
        self._overlay = overlay
        self._refiner = refiner
    
    def start(self):
        """Start the full pipeline."""
        self._processor.start(on_translation=self._on_translation)
        self._capture.start(on_audio=self._vad.process_chunk)
        self._state = PipelineState.RUNNING
        self.state_changed.emit(self._state)
    
    def pause(self):
        """Pause capture + processing. Keep overlay visible but show 'Paused'."""
        if self._state != PipelineState.RUNNING:
            return
        self._capture.stop()
        self._overlay.show_subtitle("⏸ Paused")
        self._state = PipelineState.PAUSED
        self.state_changed.emit(self._state)
    
    def resume(self):
        """Resume from pause."""
        if self._state != PipelineState.PAUSED:
            return
        self._capture.start(on_audio=self._vad.process_chunk)
        self._overlay.clear()
        self._state = PipelineState.RUNNING
        self.state_changed.emit(self._state)
    
    def toggle(self):
        """Toggle between pause and resume."""
        if self._state == PipelineState.RUNNING:
            self.pause()
        elif self._state == PipelineState.PAUSED:
            self.resume()
    
    def shutdown(self):
        """Full shutdown."""
        self._state = PipelineState.STOPPED
        self._capture.stop()
        self._processor.stop()
```

**State machine diagram:**
```
STOPPED → (start) → RUNNING → (pause) → PAUSED → (resume) → RUNNING
                      ↓                      ↓
                  (error) → ERROR        (shutdown) → STOPPED
```

**Verification:**
- [ ] Pipeline starts in RUNNING state
- [ ] Pause stops audio capture, overlay shows "Paused"
- [ ] Resume restarts capture from fresh state
- [ ] Toggle works correctly from both RUNNING and PAUSED
- [ ] Shutdown from any state is clean

---

### Task 8.3: Wire Tray Icon + Pipeline Controller into `main.py`

**Requirement:** `SubtitleApp.__init__()` creates the `PipelineController` and `AppTrayIcon`, wires signals together, and manages the full lifecycle.

**Changes to `src/main.py`:**

1. **Imports:**
   ```python
   from src.core.pipeline import PipelineController
   from src.ui.tray import AppTrayIcon
   from src.ui.settings import SettingsDialog
   from src.ui.history_dialog import HistoryDialog
   from src.core.history import HistoryManager
   ```

2. **Modified `__init__`:**
   ```python
   # Create pipeline controller (wraps capture, vad, processor, overlay, refiner)
   self.pipeline = PipelineController(
       capture=self.capture,
       vad=self.vad,
       processor=self.processor,
       overlay=self.overlay,
       refiner=self.refiner,
   )
   
   # History manager (SQLite)
   self.history = HistoryManager()
   
   # Tray icon
   self.tray = AppTrayIcon()
   self.tray.toggle_requested.connect(self.pipeline.toggle)
   self.tray.settings_requested.connect(self._open_settings)
   self.tray.history_requested.connect(self._open_history)
   self.tray.quit_requested.connect(self.shutdown)
   self.pipeline.state_changed.connect(self.tray.set_running)
   self.tray.show()
   ```

3. **New `_open_settings` method:**
   ```python
   def _open_settings(self):
       dialog = SettingsDialog(self.config, parent=self.overlay)
       dialog.accepted.connect(lambda: self._apply_settings(dialog.get_config()))
       dialog.exec_()
   
   def _apply_settings(self, new_config):
       # Apply overlay position/font changes immediately
       overlay_cfg = new_config.get("overlay", {})
       self.overlay.set_position(
           overlay_cfg.get("x", self.overlay.x()),
           overlay_cfg.get("y", self.overlay.y()),
           overlay_cfg.get("width", self.overlay.width()),
           overlay_cfg.get("height", self.overlay.height()),
       )
       # Apply language changes to processor
       model_cfg = new_config.get("model", {})
       if model_cfg.get("language"):
           self.processor.set_language(model_cfg["language"])
       # Apply LLM toggle
       llm_cfg = new_config.get("llm", {})
       if self.refiner:
           self.refiner.enabled = llm_cfg.get("enabled", True)
   ```

4. **Modified `run`:**
   ```python
   def run(self):
       # ... banner ...
       self.pipeline.start()  # starts processor + capture
       self.overlay.show()
       self.overlay.show_subtitle("Waiting for audio...")
       # ... health timer ...
       return self.app.exec_()
   ```

5. **Modified `shutdown`:**
   ```python
   def shutdown(self):
       self.tray.hide()
       self.pipeline.shutdown()
       self.history.close()
       self.overlay.hide()
       # ... rest of cleanup ...
       self.app.quit()
   ```

**Verification:**
- [ ] App starts with tray icon visible
- [ ] Tray menu → Settings opens the settings dialog
- [ ] Settings changes apply at runtime (position, font, language)
- [ ] Tray menu → Pause pauses the pipeline
- [ ] Tray menu → Resume resumes
- [ ] Tray menu → Quit does clean shutdown
- [ ] Overlay still shows subtitles when pipeline is running

---

### Task 8.4: Language Selectors in Settings (`src/ui/settings.py`)

**Requirement:** Settings dialog gains source/target language dropdowns that update the pipeline at runtime.

**Add to `__init__`:**
```python
# --- Language ---
lang_group = QGroupBox("Language")
lang_layout = QFormLayout(lang_group)

self.source_lang = QComboBox()
self.source_lang.addItems(["Japanese", "English", "Chinese", "Korean", "Spanish", "French", "German"])
self.source_lang.setCurrentText("Japanese")
lang_layout.addRow("Source:", self.source_lang)

self.target_lang = QComboBox()
self.target_lang.addItems(["English", "Japanese", "Chinese", "Korean", "Spanish", "French", "German"])
self.target_lang.setCurrentText("English")
lang_layout.addRow("Target:", self.target_lang)

layout.addWidget(lang_group)
```

**Language code mapping (internal):**
```python
LANGUAGE_CODES = {
    "Japanese": "ja",
    "English": "en", 
    "Chinese": "zh",
    "Korean": "ko",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}
```

**Getter in `get_config`:**
```python
"model": {
    "language": LANGUAGE_CODES.get(self.source_lang.currentText(), "ja"),
    "target_language": LANGUAGE_CODES.get(self.target_lang.currentText(), "en"),
}
```

**Validation:** Whisper only supports `task="translate"` for certain source→target pairs. For non-English targets, fall back to `task="transcribe"` + LLM translation. Add a validation message:
```python
def validate_language(self) -> Optional[str]:
    """Returns warning message if language combo may not work well, or None."""
    source = self.source_lang.currentText()
    target = self.target_lang.currentText()
    if source != "Japanese" and target != "English":
        return "Non-Japanese to non-English may have reduced accuracy."
    if source == target:
        return "Source and target languages are the same."
    return None
```

**Verification:**
- [ ] Settings dialog shows language dropdowns
- [ ] Selecting a different language updates the Whisper model config
- [ ] Language change takes effect on next transcription (no restart needed)
- [ ] Warning shown for unsupported combinations

---

### Task 8.5: Session History (`src/core/history.py`)

**Requirement:** Log translation sessions to a local SQLite database with timestamps, source language, and subtitle count. Viewable from tray menu.

**Implementation:**

```python
import sqlite3
import datetime
from pathlib import Path

class HistoryManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.home() / ".subtitle_translator" / "history.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._session_id = None
    
    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                source_lang TEXT DEFAULT 'ja',
                target_lang TEXT DEFAULT 'en',
                subtitle_count INTEGER DEFAULT 0,
                total_duration_sec REAL DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS subtitles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                raw_text TEXT,
                refined_text TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        self._conn.commit()
    
    def start_session(self, source: str = "ja", target: str = "en"):
        self._conn.execute(
            "INSERT INTO sessions (start_time, source_lang, target_lang) VALUES (?, ?, ?)",
            (datetime.datetime.now().isoformat(), source, target),
        )
        self._session_id = self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._conn.commit()
    
    def log_subtitle(self, raw: str, refined: str = None):
        if self._session_id is None:
            return
        self._conn.execute(
            "INSERT INTO subtitles (session_id, timestamp, raw_text, refined_text) VALUES (?, ?, ?, ?)",
            (self._session_id, datetime.datetime.now().isoformat(), raw, refined),
        )
        self._conn.execute(
            "UPDATE sessions SET subtitle_count = subtitle_count + 1 WHERE id = ?",
            (self._session_id,),
        )
        self._conn.commit()
    
    def end_session(self):
        if self._session_id is None:
            return
        self._conn.execute(
            "UPDATE sessions SET end_time = ? WHERE id = ?",
            (datetime.datetime.now().isoformat(), self._session_id),
        )
        self._conn.commit()
        self._session_id = None
    
    def get_recent_sessions(self, limit: int = 20):
        return self._conn.execute(
            "SELECT * FROM sessions ORDER BY start_time DESC LIMIT ?", (limit,)
        ).fetchall()
    
    def get_session_subtitles(self, session_id: int):
        return self._conn.execute(
            "SELECT * FROM subtitles WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
    
    def close(self):
        self.end_session()
        self._conn.close()
```

**History Dialog (`src/ui/history_dialog.py`):**
- QDialog with QTableWidget showing recent sessions (date, duration, count, source→target)
- Click a session to expand subtitles in a details panel
- "Export Session" button → exports selected session as SRT
- "Clear History" button with confirmation

**Verification:**
- [ ] History.db created in `~/.subtitle_translator/`
- [ ] Session starts on app launch, ends on quit
- [ ] Each subtitle is logged with raw + refined text
- [ ] History dialog shows recent sessions
- [ ] Clicking a session shows individual subtitles
- [ ] Export produces valid SRT file
- [ ] Clear history deletes all data (with confirmation)

---

### Task 8.6: Override `shutdown()` for Clean Exit

**Requirement:** When "Quit" is clicked in tray menu, or window is closed, all components shut down in the correct order.

**Modified `shutdown` flow:**
```python
def shutdown(self):
    logger.info("Shutting down...")
    self._health_timer.stop()
    self.tray.hide()
    self.history.end_session()
    self.history.close()
    self.pipeline.shutdown()
    self.overlay.clear()
    self.overlay.hide()
    if self._srt_file:
        self._srt_file.close()
    self.app.quit()
    logger.info("Shutdown complete")
```

**Also add:** `QApplication.aboutToQuit.connect(self.shutdown)` in `__init__` to catch all exit paths.

**Verification:**
- [ ] Tray → Quit closes the app cleanly
- [ ] Task manager → End task does not leave orphan threads
- [ ] Session is logged as ended in history

---

## Phase 9: Packaging & Distribution

**Goal:** Package the app as a standalone `.exe` for GitHub releases, with auto-update support and proper installer.

**Estimated Effort: Medium (~150-250 lines new code, plus config files)**

---

### Task 9.1: App Icon Asset

**Requirement:** A proper `.ico` file for the Windows app icon (tray, taskbar, task manager).

**Options:**
1. Use a free icon from [icons8](https://icons8.com/icons/set/microphone), [flaticon](https://www.flaticon.com/), or similar (CC attribution)
2. Generate one programmatically with Python + Pillow
3. Use a simple SVG-to-ICO converter

**Recommended:** Create `assets/icon.svg` (simple microphone + "CC" text), convert to `icon.ico` (256x256, 64x64, 32x32, 16x16 layers) using a free online converter or `python -m pip install pypng` script.

Place in `translate/assets/icon.ico` and `translate/assets/icon.png`.

---

### Task 9.2: PyInstaller Build Script (`build/build_exe.bat`)

**Requirement:** One-click build script that produces a standalone `.exe` with all dependencies bundled.

**Implementation:**

```batch
@echo off
cd /d "%~dp0.."
echo Building Subtitle Translator executable...
echo.

REM Activate venv
call .venv\Scripts\activate.bat

REM Install PyInstaller if not present
pip install pyinstaller

REM Clean previous builds
if exist "dist\SubtitleTranslator" rmdir /s /q "dist\SubtitleTranslator"
if exist "build\SubtitleTranslator" rmdir /s /q "build\SubtitleTranslator"

REM Build
pyinstaller ^
    --name "SubtitleTranslator" ^
    --onefile ^
    --windowed ^
    --icon "assets\icon.ico" ^
    --add-data "assets;assets" ^
    --add-data "config.yaml;." ^
    --hidden-import faster_whisper ^
    --hidden-import silero_vad ^
    --hidden-import PyQt5.sip ^
    --hidden-import pyaudiowpatch ^
    --collect-all faster_whisper ^
    --collect-all silero_vad ^
    run.py

echo.
echo Build complete: dist\SubtitleTranslator.exe
```

**`--onefile` vs `--onedir` decision:**
- `--onefile`: Single .exe, but slower startup (extracts to temp dir) and harder to debug
- `--onedir`: Directory with .exe + DLLs, faster startup, easier to troubleshoot

**Default:** Use `--onedir` for development builds, `--onefile` for release.

**Excluded from build:**
- `.venv/`, `tests/`, `src/` (code is bundled into .exe)
- `PLAN.md`, `PLAN-APP.md` (not needed at runtime)

**Post-build verification:**
- [ ] `dist/SubtitleTranslator/SubtitleTranslator.exe` exists
- [ ] Runs without Python installed (test on clean machine or VM)
- [ ] Tray icon shows
- [ ] Settings dialog opens
- [ ] Whisper model loads (first run downloads model to `%USERPROFILE%\.cache\`)
- [ ] Audio capture works

---

### Task 9.3: GitHub Repository Setup

**Requirement:** Clean GitHub repo with README, license, .gitignore, and release workflow.

**Files to create:**

**`README.md`:**
```markdown
# Subtitle Translator

Real-time Japanese→English subtitle overlay for Windows. 
Captures system audio, transcribes with Whisper, refines with LLM.

## Features
- Real-time system audio capture (WASAPI loopback)
- Japanese speech recognition with faster-whisper
- LLM refinement (OpenRouter / NVIDIA API)
- Transparent always-on-top subtitle overlay
- SRT export with timestamps
- System tray controls (pause/resume, settings)
- Session history

## Quick Start

### Download (Windows)
1. Go to [Releases](https://github.com/YOUR_USER/subtitle-translator/releases)
2. Download `SubtitleTranslator.zip`
3. Extract and run `SubtitleTranslator.exe`

### From Source
```bash
git clone https://github.com/YOUR_USER/subtitle-translator.git
cd subtitle-translator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

## Configuration
Edit `config.yaml` or use the Settings dialog (right-click tray icon).

## API Keys
- **OpenRouter:** Set `LLM_API_KEY` environment variable
  - Get key at https://openrouter.ai/keys
```

**`.gitignore`:**
```
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Build
dist/
build/
*.spec

# App data
*.db

# IDE
.vscode/
.idea/

# OS
Thumbs.db
.DS_Store
```

**`LICENSE`:**
- MIT License (recommended for open source)

---

### Task 9.4: GitHub Actions Release Workflow (`.github/workflows/release.yml`)

**Requirement:** Automated builds on tag push, publishes release with `.exe` attached.

```yaml
name: Build and Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: windows-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.13'
    
    - name: Install dependencies
      run: |
        python -m venv .venv
        .venv\Scripts\Activate.ps1
        pip install -r requirements.txt
        pip install pyinstaller
    
    - name: Build executable
      run: |
        .venv\Scripts\Activate.ps1
        pyinstaller --name SubtitleTranslator --onefile --windowed --icon assets/icon.ico --add-data "config.yaml;." run.py
    
    - name: Create Release
      uses: softprops/action-gh-release@v2
      with:
        files: dist/SubtitleTranslator.exe
        generate_release_notes: true
```

**How releases work:**
1. Developer runs: `git tag v1.0.0 && git push --tags`
2. GitHub Actions builds the `.exe` on a Windows runner
3. Release is created with the `.exe` attached
4. Users download the `.exe` from the Releases page

---

### Task 9.5: Version Management

**Requirement:** Single source of truth for app version, displayed in tray tooltip and settings title.

**Create `src/version.py`:**
```python
"""App version."""
__version__ = "1.0.0"
__app_name__ = "Subtitle Translator"
__author__ = "Your Name"
```

Use in tray tooltip: `f"{__app_name__} v{__version__} — Running"`

**Verification:**
- [ ] Version displayed in tray tooltip
- [ ] Version in Settings dialog title
- [ ] Version matches git tag

---

## Implementation Order

Phase 8 and Phase 9 should be implemented in this order — each task builds on the previous:

### Phase 8 (App Polish):

| Step | Task | Files | Depends On |
|------|------|-------|------------|
| 8.1 | Create app icon asset | `assets/icon.ico`, `assets/icon.png` | Nothing |
| 8.2 | Session history manager | `src/core/history.py` | Nothing |
| 8.3 | History dialog UI | `src/ui/history_dialog.py` | 8.2 |
| 8.4 | Pipeline state machine | `src/core/pipeline.py` | Nothing |
| 8.5 | System tray icon + menu | `src/ui/tray.py` | 8.1, 8.4 |
| 8.6 | Language selectors in settings | `src/ui/settings.py` | Nothing |
| 8.7 | Wire everything in main.py | `src/main.py` | 8.2, 8.3, 8.4, 8.5, 8.6 |

### Phase 9 (Distribution):

| Step | Task | Files | Depends On |
|------|------|-------|------------|
| 9.1 | PyInstaller build script | `build/build_exe.bat` | Phase 8 complete |
| 9.2 | README, .gitignore, LICENSE | Root directory | Nothing |
| 9.3 | Version module | `src/version.py` | Nothing |
| 9.4 | GitHub Actions workflow | `.github/workflows/release.yml` | 9.1, 9.2 |
| 9.5 | Test build + verify | — | 9.1-9.4 |

---

## Dependencies to Add

```txt
# requirements.txt additions (Phase 8-9)
pyinstaller>=6.0        # Only needed for builds (dev dependency)
```

No new runtime dependencies. All Phase 8 features use PyQt5 built-in widgets (QSystemTrayIcon, QTableWidget, QComboBox) and Python stdlib (sqlite3).

---

## Files Summary (All Changes)

| File | Status | Lines (approx) |
|------|--------|----------------|
| `src/main.py` | MODIFIED | +80 |
| `src/ui/overlay.py` | MODIFIED | +10 |
| `src/ui/settings.py` | MODIFIED | +60 |
| `src/ui/tray.py` | NEW | +60 |
| `src/ui/history_dialog.py` | NEW | +80 |
| `src/core/pipeline.py` | NEW | +80 |
| `src/core/history.py` | NEW | +70 |
| `src/stt/processor.py` | MODIFIED | +15 |
| `src/stt/whisper_stt.py` | MODIFIED | +10 |
| `src/llm/refiner.py` | MODIFIED | +10 |
| `src/llm/prompts.py` | MODIFIED | +5 |
| `src/version.py` | NEW | +5 |
| `assets/icon.ico` | NEW | binary |
| `assets/icon.png` | NEW | binary |
| `build/build_exe.bat` | NEW | +30 |
| `README.md` | NEW | +60 |
| `.gitignore` | NEW | +15 |
| `LICENSE` | NEW | +20 |
| `.github/workflows/release.yml` | NEW | +35 |
| **Total** | | **~625 lines new code** |

---

## Success Criteria

- [ ] System tray icon appears on launch with right-click menu
- [ ] Pause/Resume toggles translation without restart
- [ ] Settings dialog opens from tray menu with language selection
- [ ] Source/target language changes take effect at runtime
- [ ] Session history is logged and viewable
- [ ] Click-through overlay still works correctly
- [ ] Import order fix preserved (faster_whisper before PyQt5)
- [ ] Audio device loss recovery still works
- [ ] Standalone .exe builds and runs without Python installed
- [ ] GitHub release workflow creates distributable .exe
- [ ] App icon shows in taskbar, tray, and task manager
- [ ] Quit from tray menu does clean shutdown
