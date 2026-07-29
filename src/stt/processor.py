"""Orchestrator connecting VAD clause detection to Whisper transcription.

Merges incomplete Japanese clauses (mid-sentence VAD cuts) before running
Whisper, then emits (heard_source, translated_english).
"""

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

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
        max_queued: int = 10,
        merge_incomplete: bool = True,
        max_merge_seconds: float = 10.0,
        max_clause_seconds: float = 12.0,
        dual_pass: bool = False,
        cpu_threads: int = 4,
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
        self._max_queued = max_queued
        self._merge_incomplete = merge_incomplete
        self._dual_pass = dual_pass
        self._max_merge_samples = int(16000 * max_merge_seconds)
        self._max_clause_samples = int(16000 * max_clause_seconds)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._on_translation: Optional[Callable[[str, str], None]] = None

        # Pending incomplete clause (audio already transcribed once as incomplete)
        self._pending_audio: Optional[np.ndarray] = None
        self._pending_heard: str = ""
        self._pending_lang: str = ""

        # Last detected language from Whisper (used for clause merging decisions)
        self._last_detected_lang: str = ""

        self.clauses_received = 0
        self.clauses_transcribed = 0
        self.clauses_dropped = 0
        self.clauses_merged = 0

    def load_model(self) -> None:
        self._stt.load()

    def start(
        self,
        on_translation: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        if self._running:
            return
        self._on_translation = on_translation
        self._running = True
        self.load_model()
        self._thread = threading.Thread(
            target=self._process_loop, daemon=True, name="stt-processor"
        )
        self._thread.start()
        logger.info(
            "TranslationProcessor started (merge_incomplete=%s, dual_pass=%s, speed=%.2fx)",
            self._merge_incomplete,
            self._dual_pass,
            self._stt.playback_speed,
        )

    def add_clause(self, audio: np.ndarray) -> None:
        if not self._running:
            return

        self.clauses_received += 1

        if self._queue.qsize() >= self._max_queued:
            try:
                self._queue.get_nowait()
                self.clauses_dropped += 1
            except queue.Empty:
                pass

        self._queue.put_nowait(audio)

    def _emit(self, heard: str, translated: str, duration: float, elapsed: float, detected_lang: str = "") -> None:
        self.clauses_transcribed += 1
        if heard or translated:
            lang_tag = detected_lang.upper() if detected_lang else "??"
            if heard:
                logger.info(
                    "[%.2fs→%.2fs] [%s] %s | EN: %s",
                    duration, elapsed, lang_tag, heard[:60], (translated or "")[:60],
                )
            else:
                logger.info(
                    "[%.2fs→%.2fs] [%s] EN: %s", duration, elapsed, lang_tag, (translated or "")[:80]
                )
            if self._on_translation:
                self._on_translation(heard or "", translated or "", detected_lang)

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
        t0 = time.perf_counter()
        translated, detected = self._stt.translate_to_english(audio, heard)
        self._last_detected_lang = detected or lang
        self._stt.commit_context(heard, translated)
        self._emit(heard, translated, len(audio) / 16000, time.perf_counter() - t0, detected_lang=self._last_detected_lang)

    def _hold(self, audio: np.ndarray, heard: str, lang: str = "") -> None:
        self._pending_audio = audio
        self._pending_heard = heard
        self._pending_lang = lang
        logger.debug("Holding incomplete clause for merge (lang=%s)", lang or "?")

    def _process_loop(self) -> None:
        while self._running:
            try:
                audio = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
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
                mergeable = (
                    self._merge_incomplete and len(audio) < self._max_merge_samples
                )
                t0 = time.perf_counter()

                # The source pass costs a full extra Whisper run, so it only
                # runs when someone needs the text (comparison logging).
                if self._dual_pass:
                    heard = self._stt.transcribe_source(audio, lang_hint=self._last_detected_lang)
                    detected_lang = self._stt.detected_language
                    self._last_detected_lang = detected_lang
                else:
                    heard = ""
                    detected_lang = self._last_detected_lang

                if heard and mergeable and not looks_complete(heard, detected_lang):
                    self._hold(audio, heard, detected_lang)
                    continue

                translated, detected = self._stt.translate_to_english(audio, heard)
                self._last_detected_lang = detected or detected_lang
                if (
                    not self._dual_pass
                    and translated
                    and mergeable
                    and not looks_complete_en(translated)
                ):
                    self._hold(audio, heard, self._last_detected_lang)
                    continue

                self._stt.commit_context(heard, translated)
                self._emit(heard, translated, duration, time.perf_counter() - t0, detected_lang=self._last_detected_lang)

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
        logger.info(
            "TranslationProcessor stopped: %d received, %d transcribed, "
            "%d dropped, %d merged",
            self.clauses_received,
            self.clauses_transcribed,
            self.clauses_dropped,
            self.clauses_merged,
        )

    @property
    def stt(self) -> WhisperSTT:
        return self._stt

    @property
    def detected_language(self) -> str:
        return self._last_detected_lang

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
