"""Voice Activity Detection using Silero VAD (ONNX).

Detects speech in streaming audio chunks and identifies clause boundaries
(natural pauses between sentences). Emits complete audio clauses for
transcription rather than tiny fragments.

Uses Silero VAD ONNX runtime for reliable detection. Buffers incoming
audio into 512-sample windows that the ONNX model expects.
"""

import logging
import time
import warnings
from typing import Optional, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VADProcessor:
    """Streaming VAD with clause boundary detection.

    Emits a clause on silence, or when speech exceeds max_speech_duration
    (prevents 30s+ mega-clauses that stall Whisper for a minute).
    """

    VAD_WINDOW = 512  # Silero VAD ONNX requires exactly 512 samples at 16kHz

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.3,
        min_speech_duration_ms: int = 500,
        min_silence_duration_ms: int = 800,
        speech_pad_ms: int = 300,
        max_speech_duration_ms: int = 8000,
        silence_floor: float = 0.004,
        on_clause: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.silence_floor = max(0.0, float(silence_floor))
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        self.speech_pad = int(sample_rate * speech_pad_ms / 1000)
        self.max_speech_samples = int(sample_rate * max_speech_duration_ms / 1000)
        self.on_clause = on_clause

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import silero_vad
            self._vad = silero_vad.load_silero_vad(onnx=True)
            logger.info(
                "Loaded Silero VAD (ONNX, threshold=%s, max_speech=%dms)",
                self.threshold,
                max_speech_duration_ms,
            )

        self._ring_buf = np.array([], dtype=np.float32)
        self._speech_buffer: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speaking = False
        self._last_prob_log = 0.0

    def process_chunk(self, audio_chunk: np.ndarray) -> None:
        """Process a single audio chunk. May trigger on_clause callback."""
        if len(audio_chunk) == 0:
            return

        self._ring_buf = np.concatenate([self._ring_buf, audio_chunk])

        while len(self._ring_buf) >= self.VAD_WINDOW:
            window = self._ring_buf[: self.VAD_WINDOW]
            self._ring_buf = self._ring_buf[self.VAD_WINDOW :]

            # Skip the neural VAD on near-silent windows. During a paused or
            # quiet video this is most of the stream, and it is the only part
            # of the pipeline that runs continuously.
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
                self._silence_samples = 0
                if not self._is_speaking:
                    self._is_speaking = True
                    self._speech_buffer = []
                    self._speech_samples = 0
                self._speech_buffer.append(window.copy())
                self._speech_samples += self.VAD_WINDOW

                if self._speech_samples >= self.max_speech_samples:
                    logger.debug(
                        "VAD max speech reached (%.1fs) — emitting clause",
                        self._speech_samples / self.sample_rate,
                    )
                    self._emit_clause()
                    self._speech_buffer = []
                    self._speech_samples = 0
                    self._silence_samples = 0
            else:
                if self._is_speaking:
                    self._silence_samples += self.VAD_WINDOW
                    self._speech_buffer.append(window.copy())
                    self._speech_samples += self.VAD_WINDOW

                    if self._silence_samples >= self.min_silence_samples:
                        self._emit_clause()
                        self._is_speaking = False
                        self._speech_buffer = []
                        self._speech_samples = 0
                        self._silence_samples = 0

    def _emit_clause(self):
        """Combine buffered audio and fire callback."""
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
        self._vad.reset_states()
