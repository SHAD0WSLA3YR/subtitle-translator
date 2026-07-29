# Plan: Real-Time Japanese→English Subtitle Translator

## Overview

Build a desktop Python application that captures system audio (WASAPI loopback), performs real-time Japanese speech recognition with `faster-whisper` (task="translate"), optionally refines with LLM (NVIDIA API / OpenRouter), and displays subtitles in a transparent always-on-top PyQt5 overlay.

**Target Hardware:** NVIDIA GeForce MX450 (2GB VRAM), Windows  
**Target Video:** 2-hour Japanese video with no subtitles  
**Primary Use Case:** Real-time live subtitling while watching video

---

## Architecture

```
[System Audio] → [WASAPI Loopback (sounddevice)] → [Audio Buffer (16kHz mono)]
                                                           ↓
                                              [Silero VAD (clause detection)]
                                                           ↓
                                              [faster-whisper small int8 (GPU)]
                                                           ↓
                                              [LLM Refiner (NVIDIA/OpenRouter)]
                                                           ↓
                                              [PyQt5 Transparent Overlay]
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Platform | Desktop Python App | Chrome extension cannot capture system audio via WASAPI |
| STT Engine | faster-whisper small (int8_float16) | MX450 has 2GB VRAM — medium won't fit GPU. int8 quantization fits in ~1.5GB |
| VAD Strategy | Silero VAD clause detection | Capture full 2–4s spoken clauses for coherent translations (not 0.5s chunks) |
| Translation Pipeline | Whisper `task="translate"` + LLM refinement | Whisper gives raw translation; LLM (NVIDIA/OpenRouter) cleans grammar/fluency |
| Overlay | PyQt5 transparent frameless window | Always-on-top, hardware-accelerated, customizable styling |
| Processing Mode | Real-time + optional SRT export | Live overlay + ability to save subtitle track |

---

## Phase 1: Project Scaffolding & Environment

**Goal:** Set up the Python project skeleton, virtual environment, and verify dependencies work.

### Tasks

1. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install core dependencies**
   ```
   faster-whisper>=1.1.0
   sounddevice>=0.5.1
   numpy>=1.26.0
   PyQt5>=5.15.9
   silero-vad>=4.0
   pyyaml>=6.0
   torch>=2.0.0
   ```

3. **Verify GPU/CUDA availability**
   ```python
   import torch; print(f"CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
   ```
   Confirm faster-whisper can load `small` model with `compute_type="int8_float16"` on GPU.

4. **Create project structure**
   ```
   translate/
   ├── src/
   │   ├── __init__.py
   │   ├── main.py              # Application entry point
   │   ├── audio/
   │   │   ├── __init__.py
   │   │   ├── capture.py        # WASAPI loopback capture
   │   │   ├── vad.py            # Silero VAD integration
   │   │   └── buffer.py         # Ring buffer for audio chunks
   │   ├── stt/
   │   │   ├── __init__.py
   │   │   ├── whisper_stt.py    # faster-whisper model wrapper
   │   │   └── processor.py      # VAD + Whisper orchestration
   │   ├── llm/
   │   │   ├── __init__.py
   │   │   ├── refiner.py        # NVIDIA/OpenRouter LLM refinement
   │   │   └── prompts.py        # Translation refinement prompts
   │   └── ui/
   │       ├── __init__.py
   │       ├── overlay.py        # PyQt5 transparent overlay widget
   │       └── settings.py       # Settings dialog
   ├── tests/
   ├── config.yaml               # User configuration
   ├── requirements.txt
   ├── PLAN.md
   └── run.py                    # Convenience launcher
   ```

5. **Write `config.yaml`** with defaults for:
   - model size: `small`
   - device: `cuda`
   - compute_type: `int8_float16`
   - language: `ja`
   - llm_provider: `nvidia` or `openrouter`
   - api_key (from env var for security)
   - overlay position, size, opacity, font size

**Verification:** `python -c "from faster_whisper import WhisperModel; m = WhisperModel('small', device='cuda', compute_type='int8_float16'); print('Model loaded OK')"`

---

## Phase 2: Audio Capture Module (WASAPI Loopback)

**Goal:** Capture system audio output in real-time using Windows WASAPI loopback via `sounddevice`.

### Tasks

1. **Implement `src/audio/capture.py`**
   - List available WASAPI devices with loopback capability
   - Open an `sd.InputStream` with:
     - `device` = WASAPI loopback device index
     - `samplerate` = 16000 Hz (Whisper requirement)
     - `channels` = 1 (mono)
     - `dtype` = `float32`
     - `blocksize` = 512 samples (~32ms chunks for VAD)
   - Non-blocking callback-based capture into a thread-safe ring buffer
   - Handle device hot-swap and error recovery

2. **Implement `src/audio/buffer.py`**
   - Thread-safe ring buffer with configurable max duration (e.g., 10 seconds)
   - Methods: `write(chunk)`, `read(n_samples)`, `read_all()`, `clear()`
   - Support for overlapping read (keep last N samples for VAD context)

3. **Unit test**: Capture 5 seconds of system audio, save to WAV, verify it contains audio

### VAD Integration

4. **Implement `src/audio/vad.py`**
   - Load Silero VAD model (lightweight ONNX or PyTorch version)
   - Use `VADIterator` class for streaming chunk-by-chunk processing
   - Configurable thresholds:
     - `threshold=0.5` (speech probability)
     - `min_speech_duration_ms=500`
     - `min_silence_duration_ms=800` (pause detection for clause boundary)
   - **State machine**:
     ```
     SILENT → (speech detected) → SPEAKING → (silence > threshold) → CLAUSE_READY
     ```
   - When CLAUSE_READY, emit the buffered audio segment (2–4s clause) for transcription
   - Reset buffer, return to SILENT

**Key Pattern:** Don't feed 0.5s chunks to Whisper. Use VAD to detect natural pause boundaries, then send the complete 2–4 second clause.

**Verification:** Play a Japanese audio file + synthetic beeps; confirm VAD correctly detects speech segments and clause boundaries.

---

## Phase 3: Speech-to-Text Engine (faster-whisper)

**Goal:** Transcribe Japanese audio clauses to English text using faster-whisper with `task="translate"`.

### Tasks

1. **Implement `src/stt/whisper_stt.py`**
   - `WhisperSTT` class wrapping `WhisperModel`
   - Lazy initialization: model loads on first use (not at import time)
   - Configurable: model size, device, compute type, beam size
   - Method: `transcribe(audio_np: np.ndarray) -> str`
   - Use `task="translate"` for direct Japanese→English (not "transcribe")
   - Parameters:
     ```python
     model.transcribe(
         audio, 
         language="ja",          # Explicit language = faster + more accurate
         task="translate",       # Direct translation to English
         beam_size=5,
         vad_filter=False,       # We handle VAD externally
         condition_on_previous_text=True  # Use context for consistency
     )
     ```

2. **Implement `src/stt/processor.py`**
   - `TranslationProcessor` class orchestrating VAD + Whisper
   - Receives audio clauses from VAD state machine
   - Queues clauses for Whisper transcription (can run in separate thread)
   - Handles overlapping speech: if new clause arrives while transcribing, buffer it
   - Callback: `on_translation(text: str)` emitted to UI and LLM refiner

3. **Performance tuning**
   - Test `small` model with `int8_float16` on MX450 — measure inference time for 2–4s clauses
   - Benchmark: should be < real-time (i.e., process 3s of audio in < 3s)
   - If GPU OOM: fallback to `compute_type="int8"` on CPU (slower but works)
   - If too slow: try `tiny` model + heavier LLM refinement

**Verification:** Feed a 3-second Japanese audio clip → get English translation back in < 3 seconds.

---

## Phase 4: LLM Refinement Layer

**Goal:** Improve Whisper's raw translation quality using an LLM (NVIDIA API or OpenRouter).

### Tasks

1. **Implement `src/llm/prompts.py`**
   - System prompt for Japanese→English translation refinement:
   ```
   You are a Japanese-to-English translation refinement assistant.
   You will receive a raw machine translation of Japanese speech.
   Your job is to:
   1. Fix any grammar issues
   2. Make it natural English
   3. Keep the original meaning intact
   4. Keep it concise (suitable for subtitles)
   
   Output ONLY the refined translation, no explanations.
   ```

2. **Implement `src/llm/refiner.py`**
   - `LLMRefiner` class with pluggable backend:
     - **NVIDIA API**: `https://api.nvcf.nvidia.com/...` with free credits
     - **OpenRouter**: `https://openrouter.ai/api/v1/chat/completions` with free models
   - Async HTTP requests with `aiohttp` or `requests` with timeout
   - Rate limiting: max 1 request per 2 seconds (avoid API limits)
   - Caching: identical recent translations skip API call
   - Fallback: return original Whisper output if API fails/timeout
   - Streaming mode: show Whisper output immediately, replace with refined version when API returns

3. **Configuration in `config.yaml`**
   ```yaml
   llm:
     provider: openrouter  # or nvidia
     model: meta-llama/llama-3.1-8b-instruct  # or nvidia/llama-3.1-8b
     api_key_env: LLM_API_KEY
     temperature: 0.1
     max_tokens: 256
     enabled: true  # set false to skip LLM refinement
   ```

4. **Prompt template refinement**
   - For subtitle-specific output: keep it short, one thought per line
   - Handle incomplete clauses (Whisper sometimes cuts off mid-sentence)

**Verification:** Feed 10 raw Whisper translations → LLM output is noticeably more grammatically correct and natural.

---

## Phase 5: Subtitle Overlay UI (PyQt5)

**Goal:** Display real-time subtitles in a transparent, always-on-top, frameless window.

### Tasks

1. **Implement `src/ui/overlay.py` — `SubtitleOverlay` (QWidget)**
   - Window flags:
     ```python
     self.setWindowFlags(
         Qt.WindowStaysOnTopHint | 
         Qt.FramelessWindowHint | 
         Qt.Tool | 
         Qt.WindowTransparentForInput  # Click-through
     )
     self.setAttribute(Qt.WA_TranslucentBackground)
     self.setAttribute(Qt.WA_ShowWithoutActivating)
     ```
   - Geometry: configurable (default: bottom third of screen, 80% width)
   - Multiple subtitle lines: show last 1–2 lines (auto-scroll)
   - Styling:
     ```python
     QLabel {
         color: #FFFFFF;
         font-size: 28px;
         font-weight: bold;
         background-color: rgba(0, 0, 0, 180);
         border-radius: 12px;
         padding: 12px 20px;
     }
     ```
   - Methods:
     - `show_subtitle(text: str, refined_text: str = None)` — display text
     - `clear()` — hide subtitles when no speech
     - `set_position(x, y, width, height)` — reposition
     - Fade-in/fade-out animation for smooth transitions

2. **Implement `src/ui/settings.py`**
   - Simple settings panel for:
     - Overlay position/drag to move
     - Font size slider
     - Opacity slider
     - Toggle LLM refinement on/off
   - Persist to config.yaml

3. **Subtitle display logic**
   - Show raw Whisper output immediately (~500ms delay)
   - When LLM refinement arrives (~1-3s later), smoothly replace text
   - Auto-hide after 3 seconds of silence
   - Maximum subtitle line length: 80 chars (for readability)

**Verification:** Run overlay without audio capture — test with manual `show_subtitle()` calls. Confirm click-through and always-on-top behavior.

---

## Phase 6: Application Controller & Integration

**Goal:** Wire all modules together into a working application with proper threading and lifecycle management.

### Tasks

1. **Implement `src/main.py` — `SubtitleApp` controller**
   - Thread architecture:
     ```
     MAIN THREAD: PyQt5 event loop (UI)
     AUDIO THREAD: sounddevice callback → ring buffer
     VAD THREAD: reads ring buffer, runs VAD, emits clauses
     WHISPER THREAD: receives clauses, runs inference, emits translations
     LLM THREAD: receives translations, calls API, emits refined text
     ```
   - Inter-thread communication via `PyQt5.QtCore.pyqtSignal`
     - `audio_level_signal(float)` — for optional audio meter
     - `subtitle_signal(str)` — raw translation → UI
     - `refined_signal(str)` — LLM refined → UI
     - `status_signal(str)` — status messages (model loading, errors)

2. **State machine**
   ```
   INIT → LOADING_MODELS → READY → CAPTURING → TRANSLATING → DISPLAYING
                                                    ↑            ↓
                                                (loop back to CAPTURING)
   ```
   - Handle errors gracefully: if Whisper fails, log and retry next clause
   - If audio device lost, auto-reconnect

3. **Global hotkey support**
   - `Ctrl+Shift+T` — Toggle overlay on/off
   - `Ctrl+Shift+R` — Reset/restart pipeline
   - `Ctrl+Shift+Q` — Quit

4. **Implement `run.py` — Launcher**
   ```python
   #!/usr/bin/env python
   """Launcher for the real-time subtitle translator."""
   import sys
   from src.main import main
   
   if __name__ == "__main__":
       sys.exit(main())
   ```

**Verification:** Full integration test — play a Japanese video, confirm subtitles appear in real-time on the overlay.

---

## Phase 7: Testing, Tuning, & Polish

**Goal:** Test with the 2-hour Japanese video, tune parameters, add SRT export.

### Tasks

1. **Test with real content**
   - Play the 2-hour Japanese video
   - Measure: subtitle latency, accuracy, GPU memory usage, CPU usage
   - Tune VAD thresholds for Japanese speech patterns (different cadence than English)
   - Tune Whisper `beam_size` vs speed tradeoff (try 3, 5, 8)

2. **Accuracy improvements**
   - Build a small test set of 20 Japanese sentences from the video
   - Compare: raw Whisper vs LLM-refined accuracy
   - Tune LLM prompt for better subtitle formatting
   - Handle proper nouns (names, places) — add `initial_prompt` with context

3. **SRT export (bonus feature)**
   - Save all translated segments with timestamps
   - Export as `.srt` file for permanent use
   ```srt
   1
   00:01:15,000 --> 00:01:18,500
   According to the survey results...
   ```

4. **Edge cases**
   - No speech for long periods → don't accumulate memory
   - Multiple speakers talking over each other → show dominant speaker
   - Background music → VAD should filter non-speech
   - Application crash recovery → auto-restart

5. **Packaging (optional)**
   - Create a `.bat` launcher that activates venv and runs the app
   - Consider PyInstaller for standalone `.exe` (post-MVP)

**Verification:** The 2-hour video can be watched with real-time subtitles. Output SRT file has accurate timestamps and readable English.

---

## Hardware Constraints & Mitigations

| Constraint | Impact | Mitigation |
|------------|--------|------------|
| MX450 2GB VRAM | Can't run medium/large Whisper models on GPU | Use `small` with `int8_float16` (~1.5GB) |
| CPU fallback | ~2x slower if GPU OOM | `int8` on CPU, use tiny model as last resort |
| Real-time requirement | Must process < real-time per clause | 2-4s clauses processed in < 2s with `small`+int8 |
| LLM API latency | 1-3s per refinement | Show raw Whisper immediately, replace with refined |

### Expected Performance (MX450 + small int8_float16)
- **Whisper latency:** 0.3–0.8s per 3s audio clause (GPU)
- **LLM refinement:** 1–3s (API dependent)
- **Total subtitle delay:** ~1–4s from speech to display
- **VRAM usage:** ~1.5GB (Whisper small) + ~200MB (PyQt) = ~1.7GB total

---

## Dependencies

```
faster-whisper>=1.1.0     # Whisper reimplementation with CTranslate2
sounddevice>=0.5.1        # WASAPI loopback audio capture
numpy>=1.26.0             # Audio data handling
PyQt5>=5.15.9             # GUI overlay
silero-vad>=4.0           # Voice Activity Detection
pyyaml>=6.0               # Configuration
requests>=2.31.0          # LLM API calls
torch>=2.0.0              # PyTorch (for silero-vad)
```

### Installation
```bash
cd translate
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Success Criteria

- [x] Application launches and shows "Waiting for audio..." overlay
- [x] Japanese speech from system audio is detected, transcribed, and translated to English
- [x] Subtitles appear on the transparent overlay within ~1–4 seconds of speech
- [x] LLM refinement improves translation quality noticeably
- [x] The 2-hour video can be watched with real-time subtitles
- [x] Click-through overlay doesn't interfere with video player interaction
- [x] Graceful error handling (audio device loss, API failure, GPU OOM)
