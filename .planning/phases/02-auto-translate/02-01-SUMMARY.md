# Plan 02-01 Summary: Core Pipeline Auto-Detection

**Phase:** 02-auto-translate  
**Plan:** 02-01 (Wave 1)  
**Status:** ✅ Complete  

## Changes Made

### `src/stt/whisper_stt.py`
- Added `_LANGUAGE_PROMPTS` dict with 9 major language prompts (ja, zh, ko, es, fr, de, pt, ru, it)
- Default `language` param changed from `"ja"` to `"auto"`
- `_run()` now returns `Tuple[str, str]` (text, detected_lang) — passes `language=None` to model when in auto-detect mode
- `transcribe_source()` accepts `lang_hint` for prompt selection, uses `_LANGUAGE_PROMPTS`
- `translate_to_english()` returns `Tuple[str, str]`, uses detected language label in prompt context
- `process()` returns `Tuple[str, str, str]` (heard, translated, detected_lang)
- `is_untranslated` check is now guarded with `detected_language == "ja"` (Japanese-only)
- Added `detected_language` property (getter/setter)
- `transcribe()` updated to unpack new return signatures

### `src/stt/processor.py`
- Replaced `_COMPLETE_ENDINGS` with `_LANGUAGE_SENTENCE_ENDINGS` dict (ja, zh, ko)
- `looks_complete()` accepts optional `lang` parameter for language-aware clause detection
- Added `_last_detected_lang` and `_pending_lang` to processor state
- `_emit()` logs detected language tag `[JA]`, `[ZH]`, etc.
- `_flush_pending()` and `_hold()` carry language state
- `_process_loop()` unpacks 3-tuple from stt methods, passes detected_lang to clause merge decisions
- Added `detected_language` property

### `src/core/pipeline.py`
- `translation_output` signal changed to `pyqtSignal(str, str, str)` (heard, translated, detected_lang)
- `_on_translation()` accepts and emits 3rd parameter

### `src/main.py`
- Added `_current_detected_lang: str` to app state
- `_on_translation_ui()` accepts 3rd `detected_lang` param, stores it, forwards to overlay
- Startup print shows "Auto-detect" when config language is `auto`

## Verification
- ✅ All modules import cleanly
- ✅ `detected_language` property exists on WhisperSTT and TranslationProcessor
- ✅ `_run()` returns `Tuple[str, str]`
- ✅ `process()` returns `Tuple[str, str, str]`
- ✅ `looks_complete()` accepts `lang` parameter
- ✅ LSP diagnostics: no new errors (only pre-existing import resolution false positives)

## Next Up
**Wave 2:** UI/Settings — add "Auto (Detect)" option in settings dialog, language badge on overlay, detected language in tray status, history database column
