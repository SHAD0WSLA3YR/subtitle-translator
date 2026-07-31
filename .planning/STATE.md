# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-07-29)

**Core value:** Watch any Japanese video with instant, local, real-time English subtitles — no browser extension, no paid API, no audio cables.
**Current focus:** Phase 2 — Language Auto-Detection + Translation

## Current Position

Phase: 3 of 4 (Any-to-Any Translation) — complete, pending user verification
Plan: 2 of 2 in current phase
Status: Phases 2 + 3 implemented on feat/auto-detect-translate
Last activity: 2026-07-29 — any-to-any NMT (Argos), language pair UI, history fix, overlay UX cleanup

Progress: ███████░░░ 70%

## Accumulated Context

### Decisions

- v1.0.0 is stable on `master` branch
- All v2+ work will be done on feature branches branching from master
- Phase 2: Auto-detect source language → translate to English (9 major languages)
- Phase 3: Any-to-any language translation (separate translation engine for non-English targets)
- Phase 4: Near-real-time speed optimization (streaming, model optimization, benchmarks)
- Auto-detect uses Whisper's native `language=None` support (0-cost, already in the library)
- 9 major languages supported: Japanese, Chinese, Korean, Spanish, French, German, Portuguese, Russian, Italian
- Language-specific prompts improve accuracy for transcribe mode
- Detected language propagated through pipeline signals as additional parameter
- Phase 3 NMT backend: Argos Translate (offline, CTranslate2, auto-pivots through English)
- Non-English targets: 1 Whisper transcribe pass + NMT text translation (same Whisper cost as fast path)
- English target keeps Whisper's direct translate task (no NMT involved)
- Overlay controls are hover-only; language pair badge flashes on change

### Pending Todos

None yet.

### Blockers/Concerns

None yet.

## Session Continuity

Last session: 2026-07-29 18:30
Stopped at: Phase 2 planning complete, ready to execute
Resume file: None

---

### Plans Created
| Plan | Objective | Tasks | Files | Wave | Autonomous |
|------|-----------|-------|-------|------|------------|
| 02-01 | Core pipeline — WhisperSTT auto-detection + language-aware processing | 3 | whisper_stt.py, processor.py, pipeline.py, main.py | 1 | yes |
| 02-02 | UI & Settings for multilingual detection | 3 | overlay.py, settings.py, tray.py, history.py, main.py, config.yaml | 2 | no (checkpoint) |

### Next Steps
1. Create feature branch: `git checkout -b feat/auto-detect-translate`
2. Execute Wave 1: `/gsd-execute-phase 02-auto-translate --wave 1`
3. Execute Wave 2: `/gsd-execute-phase 02-auto-translate --wave 2`
