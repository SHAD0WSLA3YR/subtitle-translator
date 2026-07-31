"""Voice Activity Detection using Silero VAD (ONNX).

Detects speech in streaming audio and emits clause-sized chunks aimed at
3–6 seconds (merge short gaps, hold early ends, hard-cap at max speech).
Applies speech_pad as pre/post-roll on emit.
"""

from __future__ import annotations

import logging
import time
import warnings
from collections import deque
from typing import Optional, Callable, Deque, List

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADProcessor:
    """Streaming VAD with 3–6s-oriented clause batching."""

    VAD_WINDOW = 512  # Silero VAD ONNX requires exactly 512 samples at 16kHz

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.3,
        min_speech_duration_ms: int = 500,
        min_silence_duration_ms: int = 400,
        speech_pad_ms: int = 250,
        max_speech_duration_ms: int = 6000,
        silence_floor: float = 0.004,
        merge_silence_ms: int = 400,
        target_min_speech_ms: int = 3000,
        on_clause: Optional[Callable[[np.ndarray], None]] = None,
        on_partial: Optional[Callable[[np.ndarray], None]] = None,
        partial_interval_ms: int = 2000,
        min_partial_ms: int = 1200,
        _vad_model=None,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_floor = max(0.0, float(silence_floor))
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        self.merge_silence_samples = int(sample_rate * max(0, merge_silence_ms) / 1000)
        self.target_min_speech_samples = int(sample_rate * max(0, target_min_speech_ms) / 1000)
        # Force-end after this much silence even if under target_min (2× end silence).
        self.hold_silence_samples = max(
            self.min_silence_samples * 2,
            int(sample_rate * 800 / 1000),
        )
        self.speech_pad = int(sample_rate * speech_pad_ms / 1000)
        self.max_speech_samples = int(sample_rate * max_speech_duration_ms / 1000)
        self.on_clause = on_clause
        self.on_partial = on_partial
        self.partial_interval_samples = int(sample_rate * partial_interval_ms / 1000)
        self.min_partial_samples = int(sample_rate * min_partial_ms / 1000)
        self._samples_since_partial = 0

        if _vad_model is not None:
            self._vad = _vad_model
            logger.info(
                "VAD using injected model (threshold=%s, max_speech=%dms, "
                "merge=%dms, target_min=%dms, pad=%dms)",
                self.threshold,
                max_speech_duration_ms,
                merge_silence_ms,
                target_min_speech_ms,
                speech_pad_ms,
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import silero_vad
                self._vad = silero_vad.load_silero_vad(onnx=True)
                logger.info(
                    "Loaded Silero VAD (ONNX, threshold=%s, max_speech=%dms, "
                    "merge=%dms, target_min=%dms, pad=%dms)",
                    self.threshold,
                    max_speech_duration_ms,
                    merge_silence_ms,
                    target_min_speech_ms,
                    speech_pad_ms,
                )

        self._ring_buf = np.array([], dtype=np.float32)
        self._speech_buffer: List[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speaking = False
        self._last_prob_log = 0.0
        # Rolling pre-roll of recent non-speech windows (for speech_pad).
        self._pre_roll: Deque[np.ndarray] = deque()
        self._pre_roll_samples = 0

    def process_chunk(self, audio_chunk: np.ndarray) -> None:
        """Process a single audio chunk. May trigger on_clause callback."""
        if len(audio_chunk) == 0:
            return

        self._ring_buf = np.concatenate([self._ring_buf, audio_chunk])

        while len(self._ring_buf) >= self.VAD_WINDOW:
            window = self._ring_buf[: self.VAD_WINDOW]
            self._ring_buf = self._ring_buf[self.VAD_WINDOW :]

            if np.max(np.abs(window)) < self.silence_floor:
                is_speech = False
            else:
                tensor = torch.from_numpy(window.copy().astype(np.float32)).unsqueeze(0)
                speech_prob = self._vad(tensor, self.sample_rate).item()
                is_speech = speech_prob > self.threshold

                now = time.monotonic()
                if is_speech and now - self._last_prob_log > 2.0:
                    logger.debug("VAD speech prob=%.3f", speech_prob)
                    self._last_prob_log = now

            if is_speech:
                self._on_speech_window(window)
            else:
                self._on_silence_window(window)

    def _on_speech_window(self, window: np.ndarray) -> None:
        self._silence_samples = 0
        if not self._is_speaking:
            self._is_speaking = True
            self._speech_buffer = []
            self._speech_samples = 0
            self._samples_since_partial = 0
            # Prepend pre-roll pad.
            while self._pre_roll:
                w = self._pre_roll.popleft()
                self._speech_buffer.append(w)
                self._speech_samples += len(w)
            self._pre_roll_samples = 0

        self._speech_buffer.append(window.copy())
        self._speech_samples += self.VAD_WINDOW
        self._samples_since_partial += self.VAD_WINDOW

        if self._speech_samples >= self.max_speech_samples:
            logger.debug(
                "VAD max speech reached (%.1fs) — emitting clause",
                self._speech_samples / self.sample_rate,
            )
            self._emit_clause()
            self._speech_buffer = []
            self._speech_samples = 0
            self._silence_samples = 0
            self._samples_since_partial = 0
            # Stay in speaking mode — continuous speech continues into next clause.
            self._is_speaking = True
        elif (
            self.on_partial is not None
            and self._speech_samples >= self.min_partial_samples
            and self._samples_since_partial >= self.partial_interval_samples
        ):
            self._samples_since_partial = 0
            self.on_partial(np.concatenate(self._speech_buffer))

    def _on_silence_window(self, window: np.ndarray) -> None:
        if not self._is_speaking:
            self._push_pre_roll(window.copy())
            return

        self._silence_samples += self.VAD_WINDOW
        self._speech_buffer.append(window.copy())
        self._speech_samples += self.VAD_WINDOW

        # Gaps shorter than merge threshold: keep buffering, don't end.
        if self._silence_samples < max(self.merge_silence_samples, 1):
            return

        if self._silence_samples < self.min_silence_samples:
            return

        # Enough silence to consider ending — hold if under target_min.
        content = max(0, self._speech_samples - self._silence_samples)
        under_target = (
            self.target_min_speech_samples > 0
            and content < self.target_min_speech_samples
            and self._speech_samples < self.max_speech_samples
        )
        if under_target and self._silence_samples < self.hold_silence_samples:
            return

        self._emit_clause()
        self._is_speaking = False
        self._speech_buffer = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._samples_since_partial = 0

    def _push_pre_roll(self, window: np.ndarray) -> None:
        if self.speech_pad <= 0:
            return
        self._pre_roll.append(window)
        self._pre_roll_samples += len(window)
        while self._pre_roll_samples > self.speech_pad and self._pre_roll:
            dropped = self._pre_roll.popleft()
            self._pre_roll_samples -= len(dropped)

    def _emit_clause(self) -> None:
        """Combine buffered audio (already includes trailing silence / pre-roll)."""
        if not self._speech_buffer:
            return

        clause = np.concatenate(self._speech_buffer)
        if len(clause) < self.min_speech_samples:
            return
        if self.on_clause:
            self.on_clause(clause)

    def reset(self) -> None:
        """Reset VAD state."""
        self._ring_buf = np.array([], dtype=np.float32)
        self._speech_buffer = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speaking = False
        self._samples_since_partial = 0
        self._pre_roll.clear()
        self._pre_roll_samples = 0
        reset = getattr(self._vad, "reset_states", None)
        if callable(reset):
            reset()
