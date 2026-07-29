# Subtitle Translator v2 — Language Auto-Detection

## Goal

Build the real-time subtitle translator into a **multi-language tool**: auto-detect whatever language is being spoken and translate it to English — without the user having to manually set the source language. Then, extend to **any language → any language** translation, and optimize for **near-real-time speed**.

## What We Want to Achieve

| Phase | Goal | Status |
|-------|------|--------|
| v1 | Japanese → English (fixed), system tray, session history, overlay | ✅ Released |
| **v2** | **Auto-detect source language → English** | **In progress** |
| v3 | Any language → any language translation | Future |
| v4 | Speed optimization / streaming for near-real-time | Future |

## What We Did (v2, Waves 1 + 2)

### Wave 1 — Core Pipeline (Complete)

**Files changed:** `whisper_stt.py`, `processor.py`, `pipeline.py`, `main.py`

- **WhisperSTT**: `language="auto"` now passes `language=None` to the model → auto-detects from `_info.language`. Added 9-language prompt dictionary. `_run()` returns `(text, detected_lang)`. `process()` returns `(heard, translated, detected_lang)`.
- **TranslationProcessor**: `looks_complete()` accepts a `lang` parameter for language-aware clause merging. Detected language is stored in `_last_detected_lang` and `_pending_lang`. Log output shows `[JA]`, `[ZH]`, etc.
- **PipelineController**: `translation_output` signal changed from `(str, str)` to `(str, str, str)` — carries detected language.
- **main.py**: Stores `_current_detected_lang`, forwards to overlay and tray.

### Wave 2 — UI & Settings (Complete)

**Files changed:** `settings.py`, `overlay.py`, `tray.py`, `history.py`, `main.py`, `config.yaml`

- **config.yaml**: `language: auto` by default
- **Settings**: "Auto (Detect)" as the default source language option
- **Overlay**: Language badge in top bar (e.g., "Japanese", "Spanish") auto-hides after 10s
- **Tray**: "Language: --" status item in context menu, updates on each translation
- **History**: `detected_lang` column in subtitles table, logged per entry

## Current Issues

### 1. Language badge + tray not updating (FIXED)
- **Root cause**: `processor.py` `_emit()` was calling `_on_translation(heard, translated)` with only 2 args — never forwarding `detected_lang`.
- **Fix**: Changed to `_on_translation(heard, translated, detected_lang)`.
- **Status**: ✅ Fixed, pending user re-test.

### 2. Known: Whisper detection lag
- First clause after startup takes ~2s to translate (~1s Whisper + overhead). This is acceptable for clause-level translation but noticeable as a delay.
- Addressed in future **Phase 4** (speed optimization).

### 3. Known: No non-English target support
- Whisper's `translate` task only outputs English. French → Spanish, Japanese → Korean, etc. requires Phase 3 (separate NMT/LLM backend).

## How to Test

1. `python run.py`
2. Speak in Japanese → verify overlay shows **"Japanese"** badge + tray says **"Language: Japanese"**
3. Speak in English → verify overlay shows **"English"** badge + tray updates
4. Open Settings → **"Auto (Detect)"** should be the default source language
5. Change source to "Japanese" manually → auto-detect bypassed, badge still shows detected

## Architecture (Updated for v2)

```
config.yaml: language: auto
       │
       ▼
WhisperSTT (language=None → auto-detect from _info.language)
       │
       ▼
TranslationProcessor (language-aware looks_complete, detected_lang tracking)
       │
       ▼
PipelineController (translation_output = pyqtSignal(str, str, str))
       │
       ├──▶ SubtitleOverlay (show_subtitle + language badge)
       ├──▶ TrayIcon (context menu + Language: status)
       └──▶ HistoryManager (SQLite + detected_lang column)
```
