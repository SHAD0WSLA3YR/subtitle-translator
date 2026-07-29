# Roadmap: Subtitle Translator

## Overview

From hardcoded Japanese→English real-time subtitle translator to a fully multilingual, near-real-time any-to-any translation overlay. v1 proven the architecture works; v2 adds language flexibility and speed.

## Milestones

- ✅ **v1.0 Desktop Translator** — Phase 1 (shipped 2026-07-29)
- 🚧 **v2.0 Multilingual Translation** — Phases 2-4 (in progress)

## Phases

<details>
<summary>✅ v1.0 Desktop Translator (Phase 1) - SHIPPED 2026-07-29</summary>

### Phase 1: Foundation
**Goal**: Real-time Japanese→English subtitle overlay for Windows
**Depends on**: Nothing (initial project)
**Plans**: 1 plan (monolithic initial build)

Plans:
- [x] 01-01: Complete end-to-end subtitle translator implementation

</details>

### 🚧 v2.0 Multilingual Translation (In Progress)

**Milestone Goal:** Real-time subtitle translation supporting auto-detected source languages and any-to-any language pairs, with near-real-time performance.

#### Phase 2: Language Auto-Detection + English Translation
**Goal**: Auto-detect ANY source language and translate to English — no longer hardcoded to Japanese
**Depends on**: Phase 1 (v1 foundation)
**Requirements**: TRANS-01, TRANS-02, UI-04
**Success Criteria** (what must be TRUE):
  1. Whisper auto-detects the spoken language without user specifying it
  2. Detected language is displayed in the overlay for user awareness
  3. Translation to English works correctly for all 9 major languages (JA, ZH, KO, ES, FR, DE, PT, RU, IT)
  4. Settings UI has "Auto (Detect)" as the default source language option
  5. Config yaml accepts `auto` for language, defaults to it
  6. Detected language is logged in session history
  7. Changing source language manually in settings still works and overrides auto-detect
**Plans**: 2 plans

Plans:
- [ ] 02-01: Core pipeline — WhisperSTT auto-detection, language-aware processing, signal chain (Wave 1)
- [ ] 02-02: UI & Settings — auto-detect option, overlay badge, tray status, history logging (Wave 2)

#### Phase 3: Any-to-Any Translation
**Goal**: Translate between any supported language pair, not just to English
**Depends on**: Phase 2
**Requirements**: TRANS-03, TRANS-04
**Success Criteria** (what must be TRUE):
  1. User can select ANY target language (not just English)
  2. Translation engine (NMT or LLM) produces fluent output in the target language
  3. At least one offline translation backend works without internet
  4. Config/settings allow switching translation backends
  5. Performance stays within usable range (>0.5 RTF for offline backends)
**Plans**: 2 plans

Plans:
- [ ] 03-01: Translation engine abstraction + offline NMT integration
- [ ] 03-02: UI for any-to-any target selection + backend switching

#### Phase 4: Near-Real-Time Speed Optimization
**Goal**: Achieve near-real-time conversion latency across all language pairs
**Depends on**: Phase 3
**Requirements**: SPEED-02
**Success Criteria** (what must be TRUE):
  1. End-to-end latency from speech to subtitle ≤ 3 seconds for 5s audio clauses
  2. Streaming/overlapping translation works during active speech (no wait-for-silence)
  3. Optional smaller-model path for low-GPU systems
  4. Pipeline benchmark suite measures regression
  5. User-configurable latency/quality tradeoff in settings
**Plans**: 2 plans

Plans:
- [ ] 04-01: Streaming pipeline with overlapping transcription/translation
- [ ] 04-02: Model optimization + benchmark suite

## Progress

**Execution Order:** Phases execute in numeric order: 2 → 3 → 4

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation | v1.0 | 1/1 | Complete | 2026-07-29 |
| 2. Auto-Detect + EN | v2.0 | 0/2 | **Planning** | - |
| 3. Any-to-Any | v2.0 | 0/2 | Not started | - |
| 4. Speed Optimization | v2.0 | 0/2 | Not started | - |
