# Subtitle Translator v2 — Multilingual Translation

## Goal

Build the real-time subtitle translator into a **multi-language tool**: auto-detect whatever language is being spoken and translate it to **any target language** — all local, all free.

## What We Want to Achieve

| Phase | Goal | Status |
|-------|------|--------|
| v1 | Japanese → English (fixed), system tray, session history, overlay | ✅ Released |
| v2 | Auto-detect source language → English | ✅ Done |
| v3 | Any language → any language translation | ✅ Done (this branch) |
| v4 | Speed optimization / streaming for near-real-time | Future |

## Architecture (Phase 3)

```
config.yaml: language: auto, target_language: <code>
       │
       ▼
WhisperSTT (language=None → auto-detect from _info.language)
       │
       ▼
TranslationProcessor — target routing:
       │   target == en  → Whisper translate task (fast path, 1 pass)
       │   target != en  → Whisper transcribe pass → Argos NMT (offline)
       ▼
PipelineController (translation_output + status_message signals)
       │
       ├──▶ SubtitleOverlay (subtitle + "Japanese → Korean" badge)
       ├──▶ TrayIcon ("Language: Japanese → Korean")
       └──▶ HistoryManager (SQLite + detected_lang column)
```

## Phase 3 — Any-to-Any Translation (Complete)

**Files:** `src/translate/nmt.py` (new), `processor.py`, `pipeline.py`, `main.py`, `settings.py`, `overlay.py`, `tray.py`, `config.yaml`

- **NMT backend**: Argos Translate (CTranslate2-based, fully offline). Language
  packages download automatically on first use (~100 MB per direction) and are
  cached under the user profile. Pairs without a direct package pivot through
  English automatically.
- **Routing**: English targets keep the fast single-pass Whisper `translate`
  path. Non-English targets run one Whisper `transcribe` pass (same cost) and
  translate the text with NMT (~fast, CPU). Source == target just shows the
  transcription.
- **Live target switching**: changing Target Language in Settings applies
  without restart; the en→target model prefetches in the background.
- **Status line**: model downloads show "Downloading translation model
  (ja→ko)..." in the overlay via the new `status_message` pipeline signal.
- **Fallback**: if NMT is unavailable (no package/no internet on first use),
  the source transcription is shown instead so subtitles never go blank.

## UX Cleanup (Complete)

- Overlay controls (minimize, close, resize grip, border) now appear **only on
  hover** — the idle overlay is just subtitle text on a clean rounded backdrop.
- Removed the "⠿ drag" hint; dragging still works anywhere on the box.
- Language badge shows the full pair ("Japanese → English"), flashes ~4 s when
  the detected pair changes, and stays visible while hovering.
- Tighter margins, smaller minimum height (64 px) for a slim one-line bar.
- Auto-hide pauses while the mouse is over the box.

## History Fix (Complete)

- Session rows now resolve by stored session id instead of list position
  (positions could desync when new sessions appeared while the dialog was open).
- Malformed/legacy DB rows are skipped with a log line instead of raising —
  PyQt5 aborts the whole process on unhandled slot exceptions, which was the
  "crash" failure mode.
- The tray → History action is wrapped so any failure shows a message box
  instead of killing the app.
- Regression tests: `tests/test_history_dialog.py`.

## How to Test

1. `python run.py`
2. Settings → Target Language → e.g. Korean. First use downloads the model
   (overlay shows download status); afterwards subtitles appear in Korean.
3. Hover the overlay → minimize/close buttons, resize grip, and the
   "Japanese → Korean" badge appear; move the mouse away → clean text only.
4. Tray menu shows "Language: Japanese → Korean".
5. Tray → History → sessions load and are clickable while translating.

## Phase 4 — Near-Real-Time Latency (Complete on this machine)

**Hardware probed:** NVIDIA GeForce MX450 (2.1 GB) + i5-11300H

**Measured truth on this GPU:**
| Config | 3s clause | RTF |
|--------|-----------|-----|
| base + beam 1 | **0.75s** | **0.25** |
| small + beam 1 | 1.48s | 0.49 |
| base + beam 2 | 8.93s | 2.98 ← death |
| tiny + beam 1 | 4.61s | 1.54 |

**Why subtitles lagged 3–4 sentences:** `beam_size: 3` + `small` + 1100ms silence wait + no live partials. Whisper finished each clause after the video had already moved on, then the queue piled up.

**Optimizations invented / wired:**
1. **Hardware auto-tune** (`src/core/capability.py`) — probes VRAM and forces a realtime-safe profile (this box → `base` + beam 1 + 550ms silence).
2. **Live provisional subtitles** — VAD emits speech snapshots every ~1.4s while still talking; overlay updates one draft line in place; the final clause replaces it (no double-stack).
3. **Constant-cost tail window** — provisional decode only looks at the newest 3.5s of speech, so cost doesn't grow with long monologues.
4. **Lag governor** — tracks rolling RTF; when behind, drops the backlog (keep newest only), skips merge-holds, and pauses partials until caught up.
5. **`without_timestamps=True`** on Whisper — free encoder win we never used for overlay text.
6. **Partial decode without context prompt** — provisional lines don't pay for / poison the previous-English prompt.

Config knobs under `latency:` — set `auto_tune: false` to freeze your own model/beam choices.

## Known Limitations

- NMT quality (Argos/OPUS models) is below Whisper's English output; pairs
  pivot through English, which compounds errors slightly.
- First non-English use needs internet for the one-time model download.
- Live partials are lower quality than finals (beam 1, no context) — by design.
- MX450 still can't run `medium`/`large` in realtime; stick to `base`.
