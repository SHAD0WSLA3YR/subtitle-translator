# Subtitle Translator

## What This Is

A real-time subtitle overlay for Windows that captures system audio via WASAPI loopback, transcribes speech using faster-whisper, and displays translated subtitles in a floating always-on-top PyQt5 window. Everything runs locally — no cloud API dependencies, no paid services. Currently translates Japanese-to-English in real-time.

## Core Value

Watch any Japanese video with instant, local, real-time English subtitles — no browser extension, no paid API, no audio cables.

## Requirements

### Validated

- ✓ **CAPTURE-01**: WASAPI loopback captures system audio without browser extensions — Phase 1
- ✓ **STT-01**: Real-time Japanese speech-to-text using faster-whisper (translate task) — Phase 1
- ✓ **VAD-01**: Silero VAD detects speech/clause boundaries in streaming audio — Phase 1
- ✓ **UI-01**: Floating PyQt5 overlay displays translated subtitles — Phase 1
- ✓ **UI-02**: System tray icon with pause/resume, settings, speed, quit — Phase 1
- ✓ **UI-03**: Settings dialog for language, overlay appearance, performance — Phase 1
- ✓ **STORAGE-01**: Session history persisted to local SQLite — Phase 1
- ✓ **LLM-01**: Optional LLM refinement via OpenRouter/NVIDIA (disabled by default) — Phase 1
- ✓ **SPEED-01**: Playback speed compensation (0.75x–1.5x) via audio stretch — Phase 1

### Active

- [ ] **TRANS-01**: Auto-detect source language from speech (not hardcoded to Japanese)
- [ ] **TRANS-02**: Translate detected speech to English for major languages (JA, ZH, KO, ES, FR, DE, PT, RU, IT)
- [ ] **TRANS-03**: Any-to-any language translation (transcribe → translate between any supported pair)
- [ ] **TRANS-04**: Translation engine selection (offline NMT models, API providers, LLM-based)
- [ ] **SPEED-02**: Near-real-time translation latency optimization (streaming approaches, smaller models, pipeline improvements)
- [ ] **UI-04**: Language selection and language indicator in overlay/settings

### Out of Scope

- Mobile/tablet app — Windows desktop only
- Cloud-only translation (all processing local by default; API optional)
- Video recording or streaming — translation-only tool
- Speech-to-speech translation — text overlay only

## Context

### Architecture

```
WASAPI loopback audio
  → Silero VAD (clause boundaries)
    → faster-whisper (transcribe + translate)
      → optional async LLM polish
        → PyQt5 overlay + tray + SQLite history + SRT export
```

### Current Limitations (v1.0.0)

1. **Hardcoded language**: `model.language: ja` — Whisper is told the source is always Japanese
2. **English-only output**: Whisper's `translate` task only targets English — no other target language supported
3. **Single pass per clause**: For speed, the source transcribe pass is skipped unless `--compare` flag is used
4. **Clause-level granularity**: Audio is segmented by VAD silence, translated one clause at a time
5. **No language detection**: The UI has a language combo box, but changing it doesn't auto-detect — it just sets the model parameter

### Key Technical Details

- **STT backend**: faster-whisper (CTranslate2-optimized Whisper) with CUDA/CPU support
- **VAD**: Silero VAD v4 ONNX, 512-sample windows
- **Audio capture**: WASAPI loopback via pyaudiowpatch
- **Overlay**: PyQt5 transparent window with acrylic blur support
- **Model auto-fallback**: GPU → CPU if CUDA unavailable
- **Clause merging**: Incomplete sentences merged across VAD cuts for coherence
- **Context carry-over**: Previous source/translated text fed as prompt to maintain consistency

## Constraints

- **Windows-only**: WASAPI loopback is Windows-specific
- **Local-first**: All processing should work offline; APIs optional
- **GPU recommended**: Real-time performance depends on CUDA GPU
- **Python 3.10+**: Runtime requirement
- **Privacy**: Audio must never leave the machine unless user explicitly enables external translation

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| faster-whisper over openai-whisper | 4x faster via CTranslate2, better for real-time | ✓ Good |
| Silero VAD over WebRTC VAD | Better accuracy, configurable sensitivity | ✓ Good |
| WASAPI loopback over browser extension | No browser dependency, works with any app | ✓ Good |
| PyQt5 for overlay | Native Windows support, acrylic blur, good DX | ✓ Good |
| Feature branches for v2 | v1 stable on master, v2 features isolated for iteration | — Pending |

---
*Last updated: 2026-07-29 after v1.0.0 completion*
