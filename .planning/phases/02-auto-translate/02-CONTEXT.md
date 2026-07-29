# Phase 2: Language Auto-Detection + English Translation - Context

**Gathered:** 2026-07-29
**Status:** Ready for planning
**Source:** User requirements from /gsd-plan-phase

<domain>
## Phase Boundary

**Phase 2: Language Auto-Detection + English Translation**

This phase upgrades the translator from hardcoded Japanese→English to auto-detect ANY language and translate to English. It covers:
- Whisper auto-detection of source language
- Translation to English using Whisper's built-in `translate` task (natively supports any→English)
- Major language support (Japanese, Chinese, Korean, Spanish, French, German, Portuguese, Russian, Italian)
- UI updates to show detected language
- Config/settings for auto-detect mode
- Testing accuracy across supported languages

**NOT in scope:** Translation to non-English target languages (Phase 3), speed optimization (Phase 4).

This is the foundation that Phase 3 (any-to-any) and Phase 4 (speed) build upon.
</domain>

<decisions>
## Implementation Decisions

### Language Detection
- D-01: Whisper will auto-detect source language by passing `language=None` to the model
- D-02: The detected language and confidence will be surfaced through the pipeline to the overlay
- D-03: Language detection runs on the first speech clause after startup; user can override manually

### Language Processing
- D-04: Whisper's `translate` task (which only outputs English) will be used for the translation step — this is already proven in v1 and is free/local
- D-05: For `transcribe` mode (source text), language-specific initial prompts improve accuracy — we'll use language-appropriate prompts rather than the current Japanese-only prompt
- D-06: When source language is set to "Auto" (auto-detect), the `transcribe` pass runs first to detect language, then the `translate` pass uses the detected language for context. When user specifies a language, we skip detection (current behavior)
- D-07: The `looks_complete` heuristic in the processor will be made language-aware (not just Japanese endings) to handle all supported languages
- D-08: If Whisper's detected language confidence is below threshold (< 0.5), and no user override is set, fall back to a best-guess language or the last-used language

### Supported Languages (Major)
- D-09: Phase 2 supports: Japanese (ja), Chinese (zh), Korean (ko), Spanish (es), French (fr), German (de), Portuguese (pt), Russian (ru), Italian (it) — these are the languages Whisper has highest accuracy for
- D-10: The settings UI language combo box will add an "Auto (Detect)" option as the default, while keeping manual override available
- D-11: The config.yaml `model.language` field will accept `auto` as a value for auto-detection mode

### UI Changes
- D-12: The overlay will show the detected source language (e.g., "JA → EN" badge or "[JA] subtitle text") so the user knows what was detected
- D-13: Settings dialog language section updated: source language gets "Auto (Detect)" option + per-language prompts toggle
- D-14: Tray icon menu shows current detected language

### Data & History
- D-15: The detected language will be logged in the history SQLite database alongside each session/subtitle entry
- D-16: The source text (transcribe) pass will be considered for accuracy testing across languages (may optionally enable for quality monitoring)

### the agent's Discretion
- Implementation order of major language prompt templates
- Exact format of language indicator in overlay (badge, prefix, separate area)
- Thresholds for auto-detect confidence and fallback behavior
- Integration test strategy for multi-language accuracy

</decisions>

<canonical_refs>
## Canonical References

### Current Implementation
- `src/stt/whisper_stt.py` — Core Whisper wrapper, modify `_run()` to support `language=None`, add detected language return
- `src/stt/processor.py` — Pipeline orchestrator, update `looks_complete()` for multilingual, propagate detected language
- `src/core/pipeline.py` — Signal signatures may need updating to include detected language
- `src/main.py` — App controller, wires config to processor
- `src/ui/settings.py` — Settings dialog, add Auto-Detect to language combo
- `src/ui/overlay.py` — Subtitle display, add language badge/indicator
- `src/ui/tray.py` — System tray menu, show detected language status
- `src/core/history.py` — SQLite session history, add detected_lang column
- `config.yaml` — User config, default language changes to `auto`
- `src/llm/prompts.py` — Prompt templates (for Phase 3, may update system prompt)

</canonical_refs>

<specifics>
## Specific Ideas

The primary changes are:
1. **WhisperSTT auto-detect**: Change `_run()` to accept `language=None`, extract detected language from `_info` response
2. **Language-appropriate prompts**: Create prompt templates for each major language (similar to `_JA_PROMPT` but for ZH, KO, ES, FR, DE, PT, RU, IT)
3. **Pipeline signal update**: `translation_output` signal should carry detected language info
4. **UI indicator**: Show "ZH → EN" or "[Detected: Japanese]" in overlay
5. **Settings**: "Auto (Detect)" as default source language option
6. **History**: Track detected language per session/entry

Whisper natively supports language detection — the `_model.transcribe()` returns `_info` with `language` and `language_probability`. This is free and requires no additional models.

```python
# Current call in whisper_stt.py
segments, _info = self._model.transcribe(audio, **kwargs)
# _info.language — already available! We just need to USE it.
```

</specifics>

<deferred>
## Deferred Ideas

- Any-to-any translation (non-English target) — Phase 3
- Speed optimization / streaming — Phase 4
- Real-time segmented translation (translate while speaking) — Phase 4
- Additional minor languages beyond the 9 major ones — Phase 3
- Custom language code support — future
- Translation memory / glossary — future

</deferred>

---

*Phase: 02-auto-translate*
*Context gathered: 2026-07-29 via /gsd-plan-phase*
