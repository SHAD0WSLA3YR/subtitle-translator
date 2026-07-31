"""Orchestrator connecting VAD clause detection to Whisper transcription.

Merges incomplete clauses (mid-sentence VAD cuts) before running Whisper,
then emits (heard_source, translated, detected_lang).

Target language routing (streaming default):
  - JA (or source) via Whisper task=transcribe only
  - EN (or other targets) via ContextualTranslator / NMT — never Whisper translate
Legacy non-streaming path may still use Whisper translate when streaming=false.
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from src.translate.nmt import NMTTranslator
from src.translate.contextual import ContextualTranslator
from src.stt.streamer import StreamingSession, TranslateWorker
from .whisper_stt import WhisperSTT

logger = logging.getLogger(__name__)

# Language-specific sentence endings for clause completion detection.
# The processor merges incomplete clauses (mid-sentence VAD cuts) before
# running Whisper. These endings signal that a clause is likely complete.
_LANGUAGE_SENTENCE_ENDINGS = {
    "ja": (
        "。", "！", "？", "!", "?",
        "です", "ます", "でした", "ました", "ません", "でしたね",
        "ですね", "ますね", "ですよ", "ますよ", "ましたよ",
        "ましょう", "ください", "かな", "よね", "んだ", "んです",
        "ますから", "ですから", "ですが", "ますが",
    ),
    "zh": ("。", "！", "？", "!", "?", "了", "的", "是", "在", "有", "吗", "呢", "啊", "吧"),
    "ko": (".", "!", "?", "습니다", "니다", "어요", "아요", "해요", "했어요", "입니다", "있습니다", "없습니다"),
}


def looks_complete(text: str, lang: str = "") -> bool:
    """Heuristic: does this text look like a finished sentence in the given language?

    Args:
        text: The text to check.
        lang: Language code (ja, zh, ko, etc.). If empty, uses common endings.
    """
    if not text:
        return True
    text_stripped = text.strip()
    if not text_stripped:
        return True
    if lang and lang in _LANGUAGE_SENTENCE_ENDINGS:
        endings = _LANGUAGE_SENTENCE_ENDINGS[lang]
    else:
        endings = (".", "!", "?", "。", "！", "？")
    if text_stripped[-1] in endings:
        return True
    return any(text_stripped.endswith(end) for end in endings)


def looks_complete_en(text: str) -> bool:
    """Completeness check for English, used when the source pass is off."""
    if not text:
        return True
    stripped = text.strip().rstrip('"\'”’)')
    if not stripped:
        return True
    return stripped[-1] in ".!?。"


class TranslationProcessor:
    """Orchestrates VAD → (optional merge) → Whisper pipeline.

    Callback signature: on_translation(heard: str, translated: str)
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "ja",
        beam_size: int = 3,
        playback_speed: float = 1.0,
        max_queued: int = 3,
        merge_incomplete: bool = True,
        max_merge_seconds: float = 10.0,
        max_clause_seconds: float = 12.0,
        dual_pass: bool = False,
        cpu_threads: int = 4,
        target_language: str = "en",
        nmt: Optional[NMTTranslator] = None,
        lag_governor: bool = True,
        # Only decode the newest N seconds for live partials — keeps cost
        # constant even when the speaker has been talking for 20s.
        partial_tail_seconds: float = 3.5,
        streaming: bool = False,
        final_beam: int = 3,
        translator: Optional[ContextualTranslator] = None,
        min_final_seconds: float = 1.2,
        streaming_flush_chars: int = 80,
        streaming_flush_timeout_s: float = 6.0,
        max_subtitle_age_s: float = 6.0,
    ):
        self._stt = WhisperSTT(
            model_size=model_size,
            device=device,
            compute_type=compute_type,
            language=language,
            beam_size=beam_size,
            playback_speed=playback_speed,
            cpu_threads=cpu_threads,
        )
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._max_queued = max(1, int(max_queued))
        self._merge_incomplete = merge_incomplete
        self._dual_pass = dual_pass
        self._lag_governor = lag_governor
        self._partial_tail_samples = int(16000 * max(1.5, partial_tail_seconds))
        self._max_merge_samples = int(16000 * max_merge_seconds)
        self._max_clause_samples = int(16000 * max_clause_seconds)
        self._streaming = bool(streaming)
        self._final_beam = max(1, int(final_beam))
        self._min_final_seconds = float(min_final_seconds)
        self._streaming_flush_chars = int(streaming_flush_chars)
        self._streaming_flush_timeout_s = float(streaming_flush_timeout_s)
        self._max_subtitle_age_s = float(max_subtitle_age_s)
        self._translator = translator
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._on_translation: Optional[Callable[[str, str], None]] = None
        self._on_status: Optional[Callable[[str], None]] = None
        self._on_partial: Optional[Callable[[str], None]] = None
        self._target_language = (target_language or "en").lower()
        self._nmt = nmt or NMTTranslator(on_status=self._emit_status)

        # Latest in-progress speech snapshot from VAD (live subtitles).
        # Only one is kept: a newer partial supersedes an unprocessed older one.
        self._partial_lock = threading.Lock()
        self._partial_audio: Optional[np.ndarray] = None

        # Pending incomplete clause (audio already transcribed once as incomplete)
        self._pending_audio: Optional[np.ndarray] = None
        self._pending_heard: str = ""
        self._pending_lang: str = ""

        # Last detected language from Whisper (used for clause merging decisions)
        self._last_detected_lang: str = ""
        self._lang_lock_candidate: str = ""
        self._lang_lock_hits: int = 0
        self._lang_unlock_hits: int = 0

        # Rolling latency stats — used by the lag governor.
        self._recent_rtfs: list[float] = []
        self._behind = False

        self._session: Optional[StreamingSession] = None
        self._translate_worker: Optional[TranslateWorker] = None
        self._silence_audio: Optional[np.ndarray] = None
        self._abs_sample_cursor = 0
        self._session_needs_origin = True

        self.clauses_received = 0
        self.clauses_transcribed = 0
        self.clauses_dropped = 0
        self.clauses_merged = 0
        self.partials_emitted = 0

    def load_model(self) -> None:
        self._stt.load()

    def start(
        self,
        on_translation: Optional[Callable[[str, str], None]] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_partial: Optional[Callable[[str], None]] = None,
    ) -> None:
        if self._running:
            return
        self._on_translation = on_translation
        self._on_status = on_status
        self._on_partial = on_partial
        self._running = True
        self.load_model()
        if self._translator is not None:
            self._emit_status("Warming up translation...")
            try:
                self._translator.warmup()
            except Exception as exc:
                logger.warning("MT warmup failed: %s", exc)
            self._emit_status("")
        if self._streaming:
            self._session = StreamingSession(
                self._stt,
                min_final_seconds=self._min_final_seconds,
                flush_chars=self._streaming_flush_chars,
                flush_timeout_s=self._streaming_flush_timeout_s,
            )
            if self._translator is None:
                self._translator = ContextualTranslator(nmt=self._nmt)
            self._translate_worker = TranslateWorker(
                self._stt,
                target_language=self._target_language,
                final_beam=self._final_beam,
                on_result=self._on_flush_translated,
                nmt=self._nmt,
                translator=self._translator,
                min_final_seconds=self._min_final_seconds,
                audio_clock_s=lambda: self.audio_clock_s,
                max_subtitle_age_s=self._max_subtitle_age_s,
            )
            self._translate_worker.start()
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True, name="stt-processor"
        )
        self._thread.start()
        logger.info(
            "TranslationProcessor started (streaming=%s, merge=%s, dual_pass=%s, "
            "speed=%.2fx, final_beam=%d)",
            self._streaming,
            self._merge_incomplete,
            self._dual_pass,
            self._stt.playback_speed,
            self._final_beam,
        )

    def _on_flush_translated(
        self,
        source: str,
        translated: str,
        lang: str,
        duration: float = 0.0,
        elapsed: float = 0.0,
        event=None,
    ) -> None:
        """Async flush result from TranslateWorker — never blocks ASR."""
        self._last_detected_lang = lang or self._last_detected_lang
        if self._last_detected_lang:
            self._consider_lang_lock(self._last_detected_lang, max(duration, 1.5))
        # Commit context so the next clause benefits from continuity.
        # In streaming mode this is the only place context gets committed
        # (the non-streaming path commits in _finish_clause instead).
        self._stt.commit_context(source, translated)
        self._emit(
            source, translated, duration, elapsed,
            detected_lang=self._last_detected_lang,
            utterance_event=event,
        )

    def add_clause(self, audio: np.ndarray) -> None:
        if not self._running:
            return

        self.clauses_received += 1

        if self._streaming:
            # End-of-utterance (silence / max-speech). Handled on the ASR
            # thread — never runs the old blocking translate path.
            with self._partial_lock:
                self._silence_audio = audio
            return

        # Lag governor: when we fall behind, keep only the newest clause.
        # Watching the *current* sentence beats finishing a 4-sentence backlog.
        while self._queue.qsize() >= self._max_queued:
            try:
                self._queue.get_nowait()
                self.clauses_dropped += 1
                self._behind = True
            except queue.Empty:
                break

        self._queue.put_nowait(audio)

    def add_partial(self, audio: np.ndarray) -> None:
        """Snapshot of speech still in progress (from VAD). Latest wins.

        Streaming mode keeps the full speech buffer (LocalAgreement trims).
        Legacy mode crops to a constant-cost tail and skips when backlogged.
        """
        if not self._running:
            return
        if self._streaming:
            with self._partial_lock:
                self._partial_audio = audio
            return
        if self._queue.qsize() > 0 or self._behind:
            return
        # Constant-cost window: only the newest few seconds of speech.
        if len(audio) > self._partial_tail_samples:
            audio = audio[-self._partial_tail_samples :]
        with self._partial_lock:
            self._partial_audio = audio

    def _take_partial(self) -> Optional[np.ndarray]:
        with self._partial_lock:
            audio, self._partial_audio = self._partial_audio, None
            return audio

    def _take_silence(self) -> Optional[np.ndarray]:
        with self._partial_lock:
            audio, self._silence_audio = self._silence_audio, None
            return audio

    def _emit_partial(self, text: str) -> None:
        # Empty string clears the live draft line on the overlay.
        if self._on_partial is None:
            return
        try:
            self._on_partial(text or "")
            if text:
                self.partials_emitted += 1
        except Exception:
            pass

    def _submit_flush(self, unit) -> None:
        if unit is None or self._translate_worker is None:
            return
        self._translate_worker.submit(unit)

    def _note_rtf(self, duration: float, elapsed: float) -> None:
        if duration <= 0:
            return
        rtf = elapsed / duration
        self._recent_rtfs.append(rtf)
        if len(self._recent_rtfs) > 8:
            self._recent_rtfs = self._recent_rtfs[-8:]
        avg = sum(self._recent_rtfs) / len(self._recent_rtfs)
        # RTF > 0.85 means Whisper is barely keeping up with speech — enter
        # catch-up mode (skip merge, drop partials) until we recover.
        was_behind = self._behind
        self._behind = avg > 0.85 or self._queue.qsize() > 0
        if self._behind and not was_behind:
            logger.info("Lag governor ON (avg RTF=%.2f) — preferring newest speech", avg)
        elif not self._behind and was_behind:
            logger.info("Lag governor OFF (avg RTF=%.2f)", avg)

    def _process_streaming_partial(self, audio: np.ndarray) -> None:
        """LocalAgreement draft update — never waits on translation."""
        if self._session is None:
            return
        if self._session_needs_origin:
            self._session.set_timeline_origin(self._abs_sample_cursor)
            self._session_needs_origin = False
        t0 = time.perf_counter()
        draft, flush = self._session.update(
            audio, lang_hint=self._last_detected_lang
        )
        elapsed = time.perf_counter() - t0
        duration = len(audio) / 16000.0
        self._note_rtf(min(duration, 4.0), elapsed)
        detected = getattr(self._stt, "detected_language", "") or ""
        if detected:
            self._last_detected_lang = detected
            self._consider_lang_lock(detected, min(duration, 4.0))
        if draft:
            self._emit_partial(draft)
        if flush is not None:
            self._submit_flush(flush)

    def _process_streaming_silence(self, audio: np.ndarray) -> None:
        """Utterance ended — flush remaining committed (+ draft), reset session."""
        if self._session is None:
            return
        if self._session_needs_origin:
            self._session.set_timeline_origin(self._abs_sample_cursor)
            self._session_needs_origin = False
        # One last update with the full clause audio so we don't miss the tail.
        _draft, flush = self._session.update(
            audio, lang_hint=self._last_detected_lang
        )
        if flush is not None:
            self._submit_flush(flush)
        # VAD silence threshold is the gate; promote leftover draft and flush.
        unit = self._session.on_silence(600.0, lang_hint=self._last_detected_lang)
        if unit is None:
            unit = self._session.force_flush_remaining()
        self._submit_flush(unit)
        self._abs_sample_cursor += max(len(audio), 0)
        self._session.reset()
        self._session_needs_origin = True
        self._emit_partial("")

    def _process_partial(self, audio: np.ndarray) -> None:
        """Fast, low-quality decode of in-progress speech for a live line.

        Runs only when the queue is idle; beam=1 and no context commit so it
        can never poison or delay the real translation of the final clause.
        """
        if self._streaming:
            self._process_streaming_partial(audio)
            return
        if self._behind or self._queue.qsize() > 0:
            return
        original_beam = getattr(self._stt, "beam_size", None)
        # Don't let the previous-English prompt slow a provisional decode —
        # and never commit the result so it can't poison final context.
        saved_prev_en = getattr(self._stt, "_prev_en", "")
        saved_prev_ja = getattr(self._stt, "_prev_ja", "")
        try:
            if original_beam is not None:
                self._stt.beam_size = 1
            self._stt._prev_en = ""
            self._stt._prev_ja = ""
            if self._target_language == "en":
                text, detected = self._stt.translate_to_english(audio)
            else:
                text = self._stt.transcribe_source(
                    audio, lang_hint=self._last_detected_lang
                )
                detected = self._stt.detected_language
                # Translate the provisional line only if the model for this
                # pair is already installed — never trigger a download here.
                if text and detected and self._nmt.is_ready(detected, self._target_language):
                    translated = self._nmt.translate(text, detected, self._target_language)
                    text = translated or text
            if detected:
                self._last_detected_lang = detected
            self._emit_partial(text)
        except Exception as e:
            logger.debug("Partial decode failed: %s", e)
        finally:
            if original_beam is not None:
                self._stt.beam_size = original_beam
            self._stt._prev_en = saved_prev_en
            self._stt._prev_ja = saved_prev_ja

    def _consider_lang_lock(self, detected: str, duration: float) -> None:
        """Stick to a confident language so short clips stop flipping en↔ja.

        Critical rule for subtitle use (target=en): never sticky-lock to English
        from an intro, then force Japanese audio through an English decoder —
        that produced the "It's the green / Do you like ramen?" garbage. Prefer
        locking to non-English, and unlock English immediately when JA/ZH/KO
        appears with modest confidence.
        """
        if self._stt.language != "auto" or not detected or detected in ("auto", "??"):
            return
        prob = float(getattr(self._stt, "language_probability", 0.0) or 0.0)
        locked = self._stt.locked_language

        # Escape hatch: locked to EN but content is clearly another language.
        if locked == "en" and detected != "en":
            if duration >= 1.0 and prob >= 0.55:
                self._stt.lock_language(detected)
                self._lang_unlock_hits = 0
                self._lang_lock_candidate = detected
                self._lang_lock_hits = 0
            return

        if locked:
            if detected == locked:
                self._lang_unlock_hits = 0
                return
            if duration >= 1.8 and prob >= 0.85:
                self._lang_unlock_hits += 1
                if self._lang_unlock_hits >= 2:
                    self._stt.lock_language(detected)
                    self._lang_unlock_hits = 0
                    self._lang_lock_candidate = detected
                    self._lang_lock_hits = 0
            return

        # Not locked yet.
        if duration < 1.2 or prob < 0.7:
            return

        # Require much stronger evidence to lock English (intros / bilingual hosts).
        if detected == "en":
            if not (prob >= 0.95 and duration >= 3.0):
                return
            # Still need 3 hits — English lock is almost never what we want.
            need_hits = 3
        else:
            need_hits = 2

        if detected == self._lang_lock_candidate:
            self._lang_lock_hits += 1
        else:
            self._lang_lock_candidate = detected
            self._lang_lock_hits = 1

        if self._lang_lock_hits >= need_hits or (
            detected != "en" and prob >= 0.95 and duration >= 3.0 and self._lang_lock_hits >= 1
        ):
            self._stt.lock_language(detected)
            self._lang_lock_hits = 0

    def _emit(
        self,
        heard: str,
        translated: str,
        duration: float,
        elapsed: float,
        detected_lang: str = "",
        utterance_event=None,
    ) -> None:
        self.clauses_transcribed += 1
        if heard or translated:
            lang_tag = detected_lang.upper() if detected_lang else "??"
            out_tag = self._target_language.upper()
            if heard:
                logger.info(
                    "[dur=%.2fs | infer=%.2fs] [%s] %s | %s: %s",
                    duration, elapsed, lang_tag, heard[:60], out_tag, (translated or "")[:60],
                )
            else:
                logger.info(
                    "[dur=%.2fs | infer=%.2fs] [%s] %s: %s",
                    duration, elapsed, lang_tag, out_tag, (translated or "")[:80],
                )
            if self._on_translation:
                self._on_translation(
                    heard or "",
                    translated or "",
                    detected_lang,
                    utterance_event,
                )

    def _emit_status(self, message: str) -> None:
        """Forward one-off status lines (e.g. model downloads) to the UI."""
        if self._on_status and message:
            try:
                self._on_status(message)
            except Exception:
                pass

    def _translate_text(self, heard: str, src: str) -> str:
        """NMT path for non-English targets. Falls back to the source text."""
        if not heard:
            return ""
        if not src or src == self._target_language:
            return heard
        translated = self._nmt.translate(heard, src, self._target_language)
        return translated if translated else heard

    def _finish_clause(self, audio: np.ndarray, heard: str, duration: float, t0: float) -> None:
        """Run the translation step for a clause and emit the result."""
        if self._target_language == "en":
            translated, detected = self._stt.translate_to_english(audio, heard)
            self._last_detected_lang = detected or self._last_detected_lang
            self._stt.commit_context(heard, translated)
        else:
            detected = self._stt.detected_language or self._last_detected_lang
            self._last_detected_lang = detected
            translated = self._translate_text(heard, detected)
            # Keep source context, but never store non-English output as the
            # "previous English" prompt line.
            self._stt.commit_context(heard, "")
        self._emit(
            heard, translated, duration, time.perf_counter() - t0,
            detected_lang=self._last_detected_lang,
        )

    def _flush_pending(self) -> None:
        """Force-process any held incomplete clause."""
        if self._pending_audio is None:
            return
        audio = self._pending_audio
        heard = self._pending_heard
        lang = self._pending_lang
        self._pending_audio = None
        self._pending_heard = ""
        self._pending_lang = ""
        if lang:
            self._last_detected_lang = lang
        self._finish_clause(audio, heard, len(audio) / 16000, time.perf_counter())

    def _hold(self, audio: np.ndarray, heard: str, lang: str = "") -> None:
        self._pending_audio = audio
        self._pending_heard = heard
        self._pending_lang = lang
        logger.debug("Holding incomplete clause for merge (lang=%s)", lang or "?")

    def _process_loop(self) -> None:
        while self._running:
            if self._streaming:
                silence = self._take_silence()
                if silence is not None:
                    try:
                        self._process_streaming_silence(silence)
                    except Exception as e:
                        logger.error("Streaming silence error: %s", e)
                    continue
                partial = self._take_partial()
                if partial is not None:
                    try:
                        self._process_streaming_partial(partial)
                    except Exception as e:
                        logger.error("Streaming partial error: %s", e)
                    continue
                time.sleep(0.05)
                continue

            try:
                audio = self._queue.get(timeout=0.2)
            except queue.Empty:
                partial = self._take_partial()
                if partial is not None:
                    self._process_partial(partial)
                continue

            # A final clause supersedes any snapshot of the same speech.
            self._take_partial()

            try:
                # Behind? Flush any held incomplete clause as-is — waiting for
                # a perfect merge is how we fall 3–4 sentences behind.
                if self._lag_governor and self._behind and self._pending_audio is not None:
                    self._flush_pending()

                if self._pending_audio is not None:
                    combined = np.concatenate([self._pending_audio, audio])
                    if len(combined) > self._max_merge_samples:
                        self._flush_pending()
                    else:
                        audio = combined
                        self._pending_audio = None
                        self._pending_heard = ""
                        self.clauses_merged += 1

                # Hard cap — Whisper on 30s+ audio can take a minute+
                if len(audio) > self._max_clause_samples:
                    logger.warning(
                        "Clause %.1fs exceeds max %.1fs — truncating",
                        len(audio) / 16000,
                        self._max_clause_samples / 16000,
                    )
                    audio = audio[-self._max_clause_samples :]

                duration = len(audio) / 16000
                # Tiny fragments with auto-detect invent languages (pl/en mid-JA).
                # Hold them to merge with the next clause instead of emitting junk.
                if (
                    duration < 0.85
                    and self._merge_incomplete
                    and not self._behind
                    and self._pending_audio is None
                ):
                    self._hold(audio, "", self._last_detected_lang)
                    continue

                # Merge is a quality trade for latency. Disable it whenever
                # the governor says we're behind.
                mergeable = (
                    self._merge_incomplete
                    and not self._behind
                    and len(audio) < self._max_merge_samples
                )
                t0 = time.perf_counter()
                to_english = self._target_language == "en"
                lang_hint = self._stt.locked_language or self._last_detected_lang

                # Non-English targets always need the source text for NMT.
                # For English the source pass costs a full extra Whisper run,
                # so it only runs when someone needs the text (comparison log).
                if self._dual_pass or not to_english:
                    heard = self._stt.transcribe_source(audio, lang_hint=lang_hint)
                    detected_lang = self._stt.detected_language
                    self._last_detected_lang = detected_lang
                    self._consider_lang_lock(detected_lang, duration)
                else:
                    heard = ""
                    detected_lang = self._last_detected_lang

                if heard and mergeable and not looks_complete(heard, detected_lang):
                    # Show something now — the merged translation replaces it.
                    if not to_english:
                        provisional = heard
                        if detected_lang and self._nmt.is_ready(detected_lang, self._target_language):
                            provisional = self._nmt.translate(
                                heard, detected_lang, self._target_language
                            ) or heard
                        self._emit_partial(provisional)
                    self._hold(audio, heard, detected_lang)
                    continue

                if to_english:
                    translated, detected = self._stt.translate_to_english(audio, heard)
                    self._last_detected_lang = detected or detected_lang
                    self._consider_lang_lock(self._last_detected_lang, duration)
                    if (
                        not self._dual_pass
                        and translated
                        and mergeable
                        and not looks_complete_en(translated)
                    ):
                        # Show the partial translation now instead of staying
                        # silent until the next clause completes the sentence.
                        self._emit_partial(translated)
                        self._hold(audio, heard, self._last_detected_lang)
                        continue
                    self._stt.commit_context(heard, translated)
                    elapsed = time.perf_counter() - t0
                    self._emit(heard, translated, duration, elapsed, detected_lang=self._last_detected_lang)
                    self._note_rtf(duration, elapsed)
                else:
                    elapsed_t0 = t0
                    self._finish_clause(audio, heard, duration, elapsed_t0)
                    self._note_rtf(duration, time.perf_counter() - elapsed_t0)

            except Exception as e:
                logger.error("Clause processing error: %s", e)

        # Shutdown flush
        try:
            self._flush_pending()
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        if self._translate_worker is not None:
            self._translate_worker.stop()
            self._translate_worker = None
        self._session = None
        logger.info(
            "TranslationProcessor stopped: %d received, %d transcribed, "
            "%d dropped, %d merged, %d partials",
            self.clauses_received,
            self.clauses_transcribed,
            self.clauses_dropped,
            self.clauses_merged,
            self.partials_emitted,
        )

    def set_target_language(self, code: str) -> None:
        """Change the output language live. Prefetches NMT models if needed."""
        code = (code or "en").lower()
        if code == self._target_language:
            return
        self._target_language = code
        logger.info("Target language set to %s", code)
        if code != "en":
            # Prefetch the en→target leg in the background; the source→en leg
            # installs lazily once the source language is known.
            self._nmt.retry_pair("en", code)
            threading.Thread(
                target=self._nmt.ensure_pair, args=("en", code),
                daemon=True, name="nmt-prefetch",
            ).start()

    @property
    def target_language(self) -> str:
        return self._target_language

    @property
    def audio_clock_s(self) -> float:
        """Seconds of audio received since session start (live capture clock)."""
        return self._abs_sample_cursor / 16000.0

    @property
    def stt(self) -> WhisperSTT:
        return self._stt

    @property
    def detected_language(self) -> str:
        return self._last_detected_lang

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
