# Subtitle Translator

Real-time **Japanese → English** subtitle overlay for Windows.

Play any Japanese video (local file, YouTube, Netflix, VLC, games — anything that makes system audio), and English subtitles appear in a floating always-on-top overlay. Everything runs **locally and free**. No browser extension. No paid API required.

Whisper's built-in `translate` task converts speech to English on your machine. Optional LLM polish is off by default.

---

## Quick start (Windows)

### Requirements

- Windows 10/11
- Python 3.10+ (3.11 recommended)
- Speakers or headphones (captures **system audio loopback**)
- Optional but recommended: NVIDIA GPU with CUDA for near-real-time speed

---

### 1. Install

```bat
git clone https://github.com/SHAD0WSLA3YR/subtitle-translator.git
cd subtitle-translator
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install a CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org) that matches your driver, then reinstall `faster-whisper`.

---

### 2. Run

```bat
run.bat
```

> First launch downloads the Whisper model (~500 MB for `small`). Cached after that.

**Useful flags:**

| Flag | What it does |
|---|---|
| `python run.py --srt out.srt` | Also write an SRT file while watching |
| `python run.py --compare` | Log heard Japanese beside English (~2x slower) |
| `python run.py -v` | Verbose logs |
| `python run.py -c my.yaml` | Custom config path |

---

### 3. Watch

1. Start the app — a blue **CC** icon appears in the system tray
2. Play any Japanese video in VLC, browser, MPC-HC, etc.
3. **English subtitles** appear in the floating overlay
4. Right-click the tray icon to **Pause**, adjust **Playback speed**, open **Settings**, browse **History**, or **Quit**

> Audio is captured from whatever your default Windows output device plays — no browser extension or audio cable needed.

**Watching at 1.25× or 1.5×?** Set the same speed in tray → **Playback speed**. The audio is stretched back to normal before Whisper sees it, which noticeably improves accuracy.

---

## The subtitle box

| Action | How |
|---|---|
| Move | Drag anywhere on the box |
| Resize | Drag the ribbed corner at the bottom right |
| Shrink to one line | Click **–** in the top bar (click again to restore) |
| Hide | Click **✕**, then tray → **Show subtitles** to bring it back |
| Restyle | Tray → **Settings** → Appearance |

Appearance settings apply immediately: text color (White / Yellow / Black / Red / Grey / Green / Cyan), background (Black / Grey / White / Blur / None), font size, text opacity, background opacity, auto-hide delay, and how many lines to stack. **Reset to defaults** restores the stock look and re-centers the box at the bottom of the screen.

`Blur` uses the Windows acrylic backdrop and falls back to a plain translucent box if your build does not support it.

---

## Configuration

Edit `config.yaml` (or use **Settings** from the tray).

| Setting | Default | Notes |
|---|---|---|
| `model.size` | `small` | `tiny`/`base` = faster; `small` = good JP accuracy; `medium` = best quality, slower |
| `model.device` | `cuda` | Use `cpu` + `compute_type: int8` if you have no GPU |
| `model.beam_size` | `3` | Lower = faster; raise to `5` for a bit more accuracy |
| `model.cpu_threads` | `4` | Caps how many cores Whisper may take |
| `model.language` | `ja` | Spoken language |
| `vad.min_silence_duration_ms` | `1100` | Lower = snappier clauses, more mid-sentence cuts |
| `vad.silence_floor` | `0.004` | Peak amplitude below which the neural VAD is skipped |
| `audio.playback_speed` | `1.0` | Match your player (0.75–1.5) |
| `llm.enabled` | `false` | Keep off for fully free/local use |
| `overlay.auto_hide_delay` | `6.0` | How long a line stays after the last update |
| `compare.enabled` | `false` | Logs heard Japanese beside the English, at ~2x Whisper cost |

Whisper translation always targets **English**. Other target languages in Settings are for labeling/history; use optional LLM refinement if you need non-English output polishing.

---

## Performance

The pipeline is built so the only thing running continuously is cheap, and Whisper runs once per clause.

Measured on an RTX-class GPU with `small` and a 5s clause:

| Stage | Cost |
|---|---|
| VAD during silence | ~0.05% of one core (neural VAD skipped below `silence_floor`) |
| VAD during speech | ~2.4% of one core |
| Whisper, 1 pass (default) | ~0.9–1.1s per 5s clause (RTF ≈ 0.2) |
| Whisper, 2 passes (`--compare`) | ~1.6–1.9s per 5s clause (RTF ≈ 0.35) |

Reproduce on your own machine:

```bat
python scripts\bench.py
python scripts\bench.py --skip-whisper     # VAD only, no model download
```

Things that cost you the most, in order:

1. **Comparison logging.** It adds a whole second Whisper pass to transcribe the Japanese. Off by default; enable per-run with `python run.py --compare`.
2. **Model size.** `medium` is roughly 3x `small`. Drop to `base` on a weak GPU or CPU-only.
3. **Beam size.** `1`–`2` is the fastest; `5` costs more on long clauses.

If Whisper falls behind, the clause queue drops the oldest audio rather than growing without bound, and clauses longer than 12s are truncated so one long monologue cannot stall the overlay.

---

## Build a standalone `.exe`

```bat
pip install pillow
python scripts\generate_icon.py
build\build_exe.bat
```

Output: `dist\SubtitleTranslator.exe`

Copy `config.yaml` next to the exe if you want editable settings. First run still downloads the Whisper model into the user cache.

> GPU CUDA runtimes are large. For widest distribution, build with `model.device: cpu` in `config.yaml`, or ship the source install instructions above (recommended for free local use).

---

## Architecture

```
WASAPI loopback audio
  → Silero VAD (clause boundaries)
    → faster-whisper (translate → English)
      → optional async LLM polish
        → transparent PyQt5 overlay + tray + SQLite history + SRT
```

---

## Project layout

```
run.py / run.bat          Entry points
config.yaml               User settings
src/main.py               App wiring
src/audio/                Capture + VAD
src/stt/                  Whisper queue
src/llm/                  Optional API refiner
src/ui/                   Overlay, tray, settings, history
src/core/                 Pipeline state machine + history DB
scripts/bench.py          VAD + Whisper cost benchmark
build/build_exe.bat       PyInstaller packaging
```

---

## Privacy

- Audio never leaves your machine unless you explicitly enable LLM refinement and set `LLM_API_KEY`.
- Session history is stored locally in SQLite.
- No accounts, no telemetry.

---

## License

MIT — use it, fork it, ship it.
