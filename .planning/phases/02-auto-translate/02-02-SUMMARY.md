# Plan 02-02 Summary: UI & Settings for Multilingual Detection

**Phase:** 02-auto-translate  
**Plan:** 02-02 (Wave 2)  
**Status:** ✅ Complete (pending user verification checkpoint)

## Changes Made

### `config.yaml`
- `language: ja` → `language: auto` (auto-detect by default)

### `src/ui/settings.py`
- Added `"Auto (Detect)": "auto"` as first entry in `LANGUAGE_NAMES`
- Source combo sorts "Auto (Detect)" first, defaults to it when config is `auto`
- Target combo excludes "Auto (Detect)" (only makes sense for source)
- `_validate_language` shows descriptive message in auto-detect mode
- `apply_settings` fallback changed from `"ja"` to `"auto"`

### `src/ui/overlay.py`
- Added `_lang_label` QLabel in the top bar (detected language badge)
- Added `_lang_badge_timer` — single-shot 10s timer auto-hides the badge
- Added `set_detected_language(lang_code)` method — shows "Japanese", "Spanish", etc.
- Added `_lang_code_to_name()` static helper for 9 languages + English
- `_reset_hide_timer()` also resets the badge timer

### `src/ui/tray.py`
- Added non-interactive `_lang_action` ("Language: --") in context menu
- Added `set_detected_language(lang_code)` method to update it

### `src/core/history.py`
- `detected_lang TEXT DEFAULT ''` column added to subtitles table
- Migration: `ALTER TABLE subtitles ADD COLUMN` for existing DBs
- `log_subtitle()` accepts `detected_lang` parameter
- `get_session_subtitles()` returns detected_lang in results

### `src/main.py`
- `_on_translation_ui` forwards detected_lang to tray: `self.tray.set_detected_language(detected_lang)`
- `history.log_subtitle()` call updated with detected_lang parameter

## Verification
- ✅ Settings: "Auto (Detect)" in LANGUAGE_NAMES, CODE_TO_NAME round-trips correctly
- ✅ Overlay: `set_detected_language` method exists, imports clean
- ✅ Tray: `set_detected_language` method exists, imports clean
- ✅ History: detected_lang stored and retrieved correctly from DB
- ✅ LSP diagnostics: no new errors (only pre-existing PyQt5 import false positives)
