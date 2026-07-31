"""Whisper-Streaming style session: LocalAgreement drafts + flush-to-translate.

ASR and translation run on independent clocks. Live overlay shows
source-language draft only. English comes from ContextualTranslator on
finalized units — never from Whisper task=translate.
"""

from __future__ import annotations

import itertools
import logging
import threading
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from src.stt.events import (
    UtteranceEvent,
    UtteranceStatus,
    audio_hash,
    source_hash,
    translation_hash,
)
from src.stt.local_agreement import (
    Word,
    join_words,
    local_agreement_n,
    should_flush,
)
from src.translate.contextual import ContextualTranslator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
_UTTERANCE_ID = itertools.count(1)
MIN_FINAL_SECONDS = 1.2

# Backward-compatible alias used in tests.
FlushUnit = UtteranceEvent


class OrderedOverlayDrain:
    """Buffer MT results and emit in non-decreasing utterance_id order."""

    def __init__(self, on_ready: Callable[[UtteranceEvent], None]):
        self._on_ready = on_ready
        self._pending: Dict[int, UtteranceEvent] = {}
        self._next_id: Optional[int] = None
        self._lock = threading.Lock()

    def note_submitted(self, utterance_id: int) -> None:
        """Reserve the lowest submitted id as the drain cursor."""
        with self._lock:
            if self._next_id is None:
                self._next_id = utterance_id
            else:
                self._next_id = min(self._next_id, utterance_id)

    def push(self, event: UtteranceEvent) -> None:
        with self._lock:
            self._pending[event.utterance_id] = event
            self._drain_locked()

    def _drain_locked(self) -> None:
        while self._next_id is not None and self._next_id in self._pending:
            event = self._pending.pop(self._next_id)
            self._next_id += 1
            try:
                self._on_ready(event)
            except Exception as exc:
                logger.error("Overlay drain callback failed: %s", exc)


class StreamingSession:
    """Stateful LocalAgreement buffer over one continuous speech stretch."""

    def __init__(
        self,
        stt,
        *,
        agreement_n: int = 2,
        max_buffer_seconds: float = 12.0,
        flush_silence_ms: float = 600.0,
        flush_chars: int = 80,
        flush_timeout_s: float = 6.0,
        min_final_seconds: float = MIN_FINAL_SECONDS,
    ):
        self._stt = stt
        self._agreement_n = agreement_n
        self._max_buffer_samples = int(SAMPLE_RATE * max_buffer_seconds)
        self._flush_silence_ms = flush_silence_ms
        self._flush_chars = flush_chars
        self._flush_timeout_s = flush_timeout_s
        self._min_final_samples = int(SAMPLE_RATE * max(0.8, min_final_seconds))

        self._confirmed_samples = 0
        self._speech: Optional[np.ndarray] = None
        self._prev_hyp: List[Word] = []
        self._unflushed_committed: List[Word] = []
        self._draft_words: List[Word] = []
        self._unit_started_at = time.monotonic()
        self._last_commit_at = time.monotonic()
        self._detected_lang = ""
        self._draft_busy = False
        self._draft_pending: Optional[np.ndarray] = None
        self._abs_origin_samples = 0
        # Exclusive end of audio already submitted to MT (speech-buffer index).
        self._flushed_samples = 0

    def set_timeline_origin(self, abs_samples: int) -> None:
        """Set absolute sample position corresponding to speech buffer index 0."""
        self._abs_origin_samples = max(0, int(abs_samples))

    def reset(self) -> None:
        self._confirmed_samples = 0
        self._speech = None
        self._prev_hyp = []
        self._unflushed_committed = []
        self._draft_words = []
        self._unit_started_at = time.monotonic()
        self._last_commit_at = time.monotonic()
        self._draft_pending = None
        self._flushed_samples = 0

    @property
    def draft_text(self) -> str:
        return join_words(self._unflushed_committed + self._draft_words)

    @property
    def committed_unflushed_text(self) -> str:
        return join_words(self._unflushed_committed)

    def update(self, full_speech: np.ndarray, lang_hint: str = "") -> tuple[str, Optional[UtteranceEvent]]:
        """One draft inference at a time; newest buffer wins when busy."""
        if len(full_speech) == 0:
            return self.draft_text, None
        if self._draft_busy:
            self._draft_pending = full_speech
            return self.draft_text, None

        draft = self.draft_text
        flush: Optional[UtteranceEvent] = None
        audio = full_speech
        while True:
            self._draft_busy = True
            try:
                draft, flush = self._update_locked(audio, lang_hint)
            finally:
                self._draft_busy = False
            pending = self._draft_pending
            self._draft_pending = None
            if pending is None:
                break
            audio = pending
        return draft, flush

    def _update_locked(
        self, full_speech: np.ndarray, lang_hint: str = ""
    ) -> tuple[str, Optional[UtteranceEvent]]:
        self._speech = full_speech
        if len(full_speech) - self._confirmed_samples > self._max_buffer_samples:
            return self._force_flush(lang_hint)

        unconfirmed = full_speech[self._confirmed_samples :]
        if len(unconfirmed) < int(SAMPLE_RATE * 0.6):
            return self.draft_text, None

        words, _text, detected, _prob = self._stt.transcribe_words(
            unconfirmed,
            lang_hint=lang_hint or self._detected_lang,
            beam_size=1,
            initial_prompt=None,
        )
        if detected:
            self._detected_lang = detected

        newly = local_agreement_n(self._prev_hyp, words, n=self._agreement_n)
        self._prev_hyp = words

        if newly:
            self._unflushed_committed.extend(newly)
            trim_sec = newly[-1].end
            trim_samples = max(0, int(trim_sec * SAMPLE_RATE) - int(0.05 * SAMPLE_RATE))
            self._confirmed_samples = min(
                len(full_speech),
                self._confirmed_samples + trim_samples,
            )
            self._prev_hyp = []
            self._last_commit_at = time.monotonic()

        self._draft_words = list(words[len(newly) :])

        flush = None
        if (
            self._confirmed_samples >= self._min_final_samples
            and should_flush(
                self.committed_unflushed_text,
                silence_ms=0.0,
                silence_threshold_ms=self._flush_silence_ms,
                max_chars=self._flush_chars,
                hard_timeout_s=time.monotonic() - self._unit_started_at,
                hard_timeout_threshold_s=self._flush_timeout_s,
            )
        ):
            flush = self._take_flush_unit(abandon_if_short=False)
        return self.draft_text, flush

    def on_silence(self, silence_ms: float, lang_hint: str = "") -> Optional[UtteranceEvent]:
        if self._draft_words and silence_ms >= self._flush_silence_ms:
            self._unflushed_committed.extend(self._draft_words)
            self._draft_words = []
            self._prev_hyp = []
        if should_flush(
            self.committed_unflushed_text,
            silence_ms=silence_ms,
            silence_threshold_ms=self._flush_silence_ms,
            max_chars=self._flush_chars,
            hard_timeout_s=time.monotonic() - self._unit_started_at,
            hard_timeout_threshold_s=self._flush_timeout_s,
        ):
            return self._take_flush_unit(abandon_if_short=True)
        return None

    def force_flush_remaining(self) -> Optional[UtteranceEvent]:
        if self._draft_words:
            self._unflushed_committed.extend(self._draft_words)
            self._draft_words = []
            self._prev_hyp = []
        return self._take_flush_unit(abandon_if_short=True)

    def _force_flush(self, lang_hint: str = "") -> tuple[str, Optional[UtteranceEvent]]:
        if self._draft_words:
            self._unflushed_committed.extend(self._draft_words)
            self._draft_words = []
        unit = self._take_flush_unit(abandon_if_short=True)
        if self._speech is not None and len(self._speech) > self._max_buffer_samples:
            # Drop already-flushed prefix; keep a tail for ongoing speech.
            drop = max(0, len(self._speech) - self._max_buffer_samples)
            drop = min(drop, self._flushed_samples)
            if drop > 0:
                self._speech = self._speech[drop:]
                self._confirmed_samples = max(0, self._confirmed_samples - drop)
                self._flushed_samples = max(0, self._flushed_samples - drop)
                self._abs_origin_samples += drop
        return self.draft_text, unit

    def _take_flush_unit(self, *, abandon_if_short: bool = False) -> Optional[UtteranceEvent]:
        text = self.committed_unflushed_text.strip()
        if not text or self._speech is None:
            self._unflushed_committed = []
            self._unit_started_at = time.monotonic()
            return None

        if abandon_if_short:
            end = len(self._speech)
        else:
            end = self._confirmed_samples if self._confirmed_samples > 0 else len(self._speech)

        # Delta only — never re-send audio already flushed in this speech stretch.
        start = max(0, min(self._flushed_samples, end))
        if end <= start:
            self._unflushed_committed = []
            self._unit_started_at = time.monotonic()
            return None

        audio = self._speech[start:end].copy()

        if len(audio) < self._min_final_samples:
            logger.debug(
                "Skipping short flush (%.2fs < %.2fs): %s",
                len(audio) / SAMPLE_RATE,
                self._min_final_samples / SAMPLE_RATE,
                text[:40],
            )
            if abandon_if_short:
                self._unflushed_committed = []
                self._unit_started_at = time.monotonic()
            return None

        uid = next(_UTTERANCE_ID)
        start_s = (self._abs_origin_samples + start) / SAMPLE_RATE
        end_s = (self._abs_origin_samples + end) / SAMPLE_RATE
        if end_s <= start_s and len(audio) > 0:
            end_s = start_s + len(audio) / SAMPLE_RATE

        event = UtteranceEvent(
            utterance_id=uid,
            source_text=text,
            audio=audio,
            detected_lang=self._detected_lang or lang_hint_safe(self._stt),
            audio_start_s=start_s,
            audio_end_s=end_s,
            audio_hash=audio_hash(audio),
            source_hash=source_hash(text),
            status=UtteranceStatus.ASR_FINAL,
        )
        self._unflushed_committed = []
        self._flushed_samples = end
        self._unit_started_at = time.monotonic()
        logger.info(event.log_asr_final())
        return event


def lang_hint_safe(stt) -> str:
    return getattr(stt, "locked_language", "") or getattr(stt, "detected_language", "") or ""


class TranslateWorker:
    """Final JA re-transcribe + ContextualTranslator. Never Whisper translate."""

    def __init__(
        self,
        stt,
        *,
        target_language: str = "en",
        final_beam: int = 3,
        on_result: Optional[Callable[..., None]] = None,
        nmt=None,
        translator: Optional[ContextualTranslator] = None,
        min_final_seconds: float = MIN_FINAL_SECONDS,
        audio_clock_s: Optional[Callable[[], float]] = None,
        max_subtitle_age_s: float = 6.0,
    ):
        self._stt = stt
        self._target = (target_language or "en").lower()
        self._final_beam = max(1, int(final_beam))
        self._on_result = on_result
        self._translator = translator or ContextualTranslator(nmt=nmt)
        self._min_final_samples = int(SAMPLE_RATE * max(0.8, min_final_seconds))
        self._audio_clock_s = audio_clock_s
        self._max_subtitle_age_s = float(max_subtitle_age_s)
        self._queue: List[UtteranceEvent] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._drain = OrderedOverlayDrain(self._emit_result)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="translate-worker"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def submit(self, unit: UtteranceEvent) -> None:
        self._drain.note_submitted(unit.utterance_id)
        with self._lock:
            self._queue.append(unit)
        self._wake.set()

    def _loop(self) -> None:
        while self._running:
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._queue:
                        break
                    unit = self._queue.pop(0)
                try:
                    self._translate(unit)
                except Exception as exc:
                    logger.error("Translate worker failed: %s", exc)

    def _translate(self, event: UtteranceEvent) -> None:
        src = event.detected_lang or "ja"
        utterance_id = event.utterance_id
        if len(event.audio) < self._min_final_samples:
            return

        ja = event.source_text
        if src and src != "en":
            self._stt.lock_language(src)

        original = self._stt.beam_size
        try:
            self._stt.beam_size = self._final_beam
            heard = self._stt.transcribe_source(event.audio, lang_hint=src)
            if heard and heard.strip():
                ja = heard.strip()
                event.source_text = ja
                event.source_hash = source_hash(ja)
        except Exception as exc:
            logger.debug("Final re-transcribe skipped: %s", exc)
        finally:
            self._stt.beam_size = original

        text_out, backend = self._translator.translate(
            ja,
            src=src,
            tgt=self._target,
            utterance_id=utterance_id,
            expected_source_hash=event.source_hash,
        )
        event.mt_latency_s = max(0.0, time.monotonic() - event.asr_final_at)
        if backend == "stale":
            event.status = UtteranceStatus.STALE
            logger.warning(
                "[id=%d | MT_STALE | src=%s] discarded",
                utterance_id,
                event.source_hash,
            )
            return

        event.translation_en = text_out or ""
        event.translation_hash = translation_hash(event.translation_en)
        event.mt_backend = backend
        event.status = UtteranceStatus.MT_DONE
        logger.info(event.log_mt_done())

        if text_out:
            self._drain.push(event)

    def _emit_result(self, event: UtteranceEvent) -> None:
        # Mark stale for the UI layer; history still written there.
        if self._max_subtitle_age_s > 0 and self._audio_clock_s is not None:
            try:
                age = float(self._audio_clock_s()) - float(event.audio_end_s)
            except Exception:
                age = 0.0
            if age > self._max_subtitle_age_s:
                event.status = UtteranceStatus.DROPPED_STALE
                logger.info(event.log_dropped_stale(age))
        dur = max(event.audio_end_s - event.audio_start_s, len(event.audio) / SAMPLE_RATE)
        if self._on_result:
            self._on_result(
                event.source_text,
                event.translation_en,
                event.detected_lang,
                dur,
                float(event.mt_latency_s or 0.0),
                event,
            )
