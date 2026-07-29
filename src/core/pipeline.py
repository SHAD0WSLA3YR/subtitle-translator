"""Pipeline state machine for pause/resume lifecycle.

Controls the audio capture → VAD → Whisper → LLM pipeline as a
finite state machine with RUNNING, PAUSED, STOPPED, and ERROR states.

All translation output is delivered via pyqtSignal for thread-safe
delivery to the Qt main thread.
"""

import enum
import logging
import threading
from typing import Optional

from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class PipelineState(enum.Enum):
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class PipelineController(QObject):
    """Controls the audio → VAD → Whisper → LLM pipeline lifecycle.

    Signals:
        state_changed: Emitted when state transitions occur.
        error_occurred: Emitted on unrecoverable errors.
        translation_output: Emitted from processor thread with (heard_ja, translated_en).
            Connected slots run on the Qt main thread automatically.
        translation_refined: Emitted after async LLM polish with refined text.
    """

    state_changed = pyqtSignal(PipelineState)
    error_occurred = pyqtSignal(str)
    translation_output = pyqtSignal(str, str)  # heard_ja, translated_en
    translation_refined = pyqtSignal(str)  # refined_text (async update)

    def __init__(self, capture, vad, processor, overlay, refiner=None, parent=None):
        super().__init__(parent)
        self._capture = capture
        self._vad = vad
        self._processor = processor
        self._overlay = overlay
        self._refiner = refiner
        self._state = PipelineState.STOPPED
        self._refine_lock = threading.Lock()

    @property
    def state(self) -> PipelineState:
        return self._state

    def start(self):
        """Start the full pipeline (STOPPED → RUNNING)."""
        if self._state == PipelineState.RUNNING:
            return
        self._processor.start(on_translation=self._on_translation)
        self._capture.start(on_audio=self._vad.process_chunk)
        self._state = PipelineState.RUNNING
        self.state_changed.emit(self._state)
        logger.info("Pipeline started")

    def pause(self):
        """Pause capture and processing (RUNNING → PAUSED).

        Called from the Qt main thread (tray menu action).
        Thread-safe for overlay calls.
        """
        if self._state != PipelineState.RUNNING:
            return
        self._capture.stop()
        self._overlay.show_subtitle("\u23f8 Paused")
        self._state = PipelineState.PAUSED
        self.state_changed.emit(self._state)
        logger.info("Pipeline paused")

    def resume(self):
        """Resume from pause (PAUSED → RUNNING)."""
        if self._state != PipelineState.PAUSED:
            return
        self._capture.start(on_audio=self._vad.process_chunk)
        self._overlay.clear()
        self._state = PipelineState.RUNNING
        self.state_changed.emit(self._state)
        logger.info("Pipeline resumed")

    def toggle(self):
        """Toggle between RUNNING and PAUSED."""
        if self._state == PipelineState.RUNNING:
            self.pause()
        elif self._state == PipelineState.PAUSED:
            self.resume()

    def shutdown(self):
        """Full shutdown to STOPPED state."""
        self._state = PipelineState.STOPPED
        self._capture.stop()
        self._processor.stop()
        self.state_changed.emit(self._state)
        logger.info("Pipeline stopped")

    def _on_translation(self, heard: str, translated: str):
        """Called from the processor (non-Qt) thread.

        Emits Whisper output immediately so the overlay stays responsive, then
        optionally polishes the English line via LLM on a separate worker.
        """
        display = translated or heard
        self.translation_output.emit(heard or "", display)
        if self._refiner and self._refiner.enabled and translated:
            threading.Thread(
                target=self._refine_async,
                args=(translated,),
                daemon=True,
                name="llm-refine",
            ).start()

    def _refine_async(self, text: str):
        """Run LLM refinement off the Whisper queue thread."""
        if self._state == PipelineState.STOPPED:
            return
        with self._refine_lock:
            try:
                refined = self._refiner.refine(text)
            except Exception as e:
                logger.debug("LLM refinement skipped: %s", e)
                return
        if (
            refined
            and isinstance(refined, str)
            and refined.strip()
            and refined.strip() != text
            and self._state != PipelineState.STOPPED
        ):
            self.translation_refined.emit(refined.strip())
