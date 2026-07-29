# Subtitle Translator — Project Status

## What We Are Building

A **real-time Japanese-to-English subtitle overlay** for Windows. System audio output (YouTube, Netflix, local video players, games, etc.) is captured, transcribed/translated with Whisper, optionally refined by an LLM, and displayed as floating subtitles on screen — no browser extension needed, works with any application.

## Goal Product

A polished desktop app that:

1. **Runs in the system tray** — no taskbar window, just a blue "CC" icon
2. **Captures any system audio** — YouTube browser, media player, games, etc.
3. **Displays subtitles in real-time** as a transparent always-on-top overlay
4. **Pause/Resume** — toggle from tray menu
5. **Language selection** — source (what's spoken) and target (what to show)
6. **LLM refinement** — optional; Whisper translate already produces English offline
7. **Session history** — browse past translations from tray menu
8. **SRT export** — save subtitles to file (`python run.py --srt out.srt`)

**Primary user goal:** watch a long Japanese video with no existing subtitles and get live English captions.

**Distribution goal:** GitHub-hosted, free, local Windows app (source install + optional PyInstaller `.exe`).

---

## Architecture

```
AudioCapture (WASAPI loopback)
  → VADProcessor (Silero VAD ONNX, clause detection)
    → TranslationProcessor (faster-whisper, queue)
      → PipelineController (state machine + async LLM refiner)
        → pyqtSignal (thread-safe bridge)
          → SubtitleApp._on_translation_ui (Qt main thread)
            → SubtitleOverlay (floating transparent window)
            → HistoryManager (SQLite)
            → SRT export
```

---

## Current Status (Phase 9 in progress — app usable for watching)

**Working:**
- System tray icon with right-click menu ✓
- Pause/Resume toggle from tray ✓
- Audio capture via WASAPI loopback ✓
- Silero VAD detecting speech (ONNX) ✓
- Whisper translation to English (CUDA/CPU, small model) ✓
- Settings dialog with language selectors ✓
- History dialog with session browsing ✓
- SRT file export ✓
- Overlay stays visible for auto-hide delay (flash bug fixed) ✓
- Async LLM refinement (does not block Whisper queue) ✓
- LLM off by default — fully free/local ✓
- Latency-tuned VAD + beam_size defaults for live watching ✓
- Quiet audio/VAD console logging ✓
- Playback-speed compensation 0.75x–1.5x (tray + settings, applies live) ✓
- Clause merging so mid-sentence VAD cuts stop wrecking translations ✓
- Subtitle box: drag to move, corner grip to resize, minimize, close ✓
- Live theming: text/background color, font size, text + background opacity ✓
- Reset to defaults (restores stock look and bottom-center position) ✓
- Single Whisper pass by default; source pass only with `--compare` ✓
- Neural VAD skipped on near-silent audio ✓
- README + PyInstaller build script + GitHub Release workflow ✓

---

## Performance profile

Measured with `python scripts/bench.py` (GPU, `small`, 5s clause):

| Stage | Cost |
|---|---|
| VAD, silence | ~0.05% of one core |
| VAD, speech | ~2.4% of one core |
| Whisper, 1 pass | ~0.9–1.1s per 5s clause (RTF ≈ 0.2) |
| Whisper, 2 passes (`--compare`) | ~1.6–1.9s per 5s clause (RTF ≈ 0.35) |

Backpressure: the clause queue drops the oldest audio past 10 items, VAD force-splits speech at 8s, and the processor truncates clauses over 12s.

---

## How to use for a 2-hour Japanese video

1. `pip install -r requirements.txt` (once)
2. Confirm `config.yaml`: `language: ja`, `llm.enabled: false`, `device: cuda` (or `cpu`)
3. `run.bat`
4. Play the video in any player through the default Windows audio device
5. Optional: `python run.py --srt lecture.srt` to keep a transcript

If latency feels high on CPU, switch `model.size` to `base` or `tiny`. On GPU, `small` is the sweet spot.

---

## Known Issues / Tradeoffs

### 1. Translation latency (~1–3 seconds typical on GPU)
**Cause:** VAD waits for a short silence (~450ms) before emitting a clause; Whisper then needs ~0.5–2s.

**Mitigations already applied:** lower silence threshold, `beam_size: 2`, LLM off by default, async LLM when enabled.

**Further options:** `tiny`/`base` model, or accept slightly more fragmentary clauses by lowering `min_silence_duration_ms` toward 300.

### 2. Whisper translate is English-only
Whisper's `task=translate` always outputs English. Other target languages need a separate translation step (optional LLM or another MT model).

### 3. Packaged `.exe` size / CUDA
A one-file build that embeds Torch + CUDA is huge and fragile across machines. Prefer documenting the Python source install for most users; ship a CPU `.exe` via GitHub Actions for convenience.

---

## Issues Solved

### Overlay subtitle flash
- **Problem:** Subtitle appeared then vanished immediately.
- **Cause:** Fade-out connected `QPropertyAnimation.finished` → `_on_fade_out_complete`. That connection survived into the next fade-in, so fade-in completion cleared the overlay.
- **Fix:** Disconnect the fade-out slot before every fade-in / clear; reconnect only for fade-out.

### VAD Not Detecting Speech (PyTorch model)
- Switched to `onnx=True` and 512-sample windows.

### Thread Safety (Timer Warnings)
- Pipeline emits `translation_output` / `translation_refined` signals onto the Qt main thread.

### LLM Refiner Crash
- None-check before `.strip()` on API content.

### CUDA Context Conflict
- `import faster_whisper` before any PyQt5 import in `main.py`.

### LLM blocking Whisper queue
- Refinement moved to a daemon thread; overlay shows Whisper text immediately and swaps in polish via `replace_last_subtitle`.

### Every clause ran Whisper twice, held clauses ran it four times
- **Problem:** `process()` always ran transcribe + translate. A clause held for merging ran both passes, then both again after merging — 4 passes for one line.
- **Fix:** Split `WhisperSTT` into `transcribe_source` / `translate_to_english`. The source pass runs only when comparison logging is on, and a held clause is never translated until it is merged and emitted. Covered by `tests/test_stt_quality.py::PassCountTests`.

### Untranslated Japanese lines appearing in the subtitles
- **Problem:** Occasional lines came through as raw Japanese, usually two or three in a row.
- **Cause:** Whisper's `translate` task sometimes echoes the source. That line was then stored as `_prev_en` and fed back as the next clause's prompt, so the next line came back Japanese too — a cascade.
- **Fix:** `is_untranslated()` measures the kana/kanji ratio (so a quoted Japanese term in an English sentence still passes). Such output is never committed as context, and a poisoned prompt triggers one retry with no prompt. Covered by `ContextPoisoningTests` and `TranslateRetryTests`.

### Overlay window grew past its set height
- `setFixedSize` plus a vertically `Ignored` size policy on the label stops Windows from expanding the translucent window when wrapped text asks for more room.

---

## Next Steps

### Remaining packaging polish
- [ ] Generate `assets/icon.ico` (`python scripts/generate_icon.py`)
- [ ] Tag `v1.0.0` and verify GitHub Actions release artifact
- [ ] Optional: auto-updater later

### Optional UX
- [ ] Hot-switch language and Whisper model without restart
- [ ] Streaming / partial ASR for sub-second captions (larger scope)

---

## File Map

```
translate/
├── run.py                      # Entry point
├── run.bat                     # Windows launcher
├── config.yaml                 # User configuration
├── requirements.txt            # Runtime dependencies
├── requirements-dev.txt        # Packaging extras
├── README.md                   # User-facing docs
├── PLAN-APP.md                 # Full development plan
├── PROJECT-STATUS.md           # This file
├── .github/workflows/release.yml
├── build/build_exe.bat         # PyInstaller
├── scripts/generate_icon.py
├── scripts/bench.py            # VAD + Whisper cost benchmark
├── assets/                     # App icons (generated)
└── src/
    ├── main.py
    ├── version.py
    ├── audio/                  # WASAPI + VAD
    ├── stt/                    # Whisper queue
    ├── llm/                    # Optional refiner
    ├── ui/                     # Overlay, tray, settings, history
    └── core/                   # Pipeline + SQLite history
```
