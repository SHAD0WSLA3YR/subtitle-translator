"""Lightweight regression checks for overlay fade / pipeline async behavior.

Run:  python -m unittest tests.test_overlay_flash -v
"""

import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.ui.overlay import SubtitleOverlay
from src.core.pipeline import PipelineController, PipelineState


_app = None


def setUpModule():
    global _app
    _app = QApplication.instance() or QApplication([])


class OverlayFlashTests(unittest.TestCase):
    def test_fade_in_disconnects_clear_slot(self):
        """Regression: fade-out finished→clear must not stay wired for fade-in."""
        overlay = SubtitleOverlay(auto_hide_delay=3.0, width=400, height=80)
        overlay.show()
        overlay.show_subtitle("first")

        overlay._on_auto_hide()
        receivers_during_fade_out = overlay._fade.receivers(overlay._fade.finished)
        self.assertGreaterEqual(receivers_during_fade_out, 1)

        overlay.show_subtitle("second")
        self.assertIn("second", overlay._current_texts)
        self.assertEqual(overlay._fade.receivers(overlay._fade.finished), 0)

        overlay._fade.finished.emit()
        self.assertIn("second", overlay._current_texts)
        overlay.close()

    def test_manual_fade_out_complete_still_clears(self):
        overlay = SubtitleOverlay(auto_hide_delay=3.0, width=400, height=80)
        overlay.show_subtitle("gone soon")
        overlay._on_fade_out_complete()
        self.assertEqual(overlay._current_texts, [])
        overlay.close()

    def test_box_starts_with_painted_background_opacity(self):
        overlay = SubtitleOverlay(width=800, height=160, opacity=200)
        self.assertEqual(overlay._bg_opacity, 200)
        overlay.show_subtitle("line one that should wrap inside a wide box")
        self.assertTrue(overlay._text.text())
        overlay.close()


class PipelineAsyncTests(unittest.TestCase):
    def test_emits_heard_and_translated_without_waiting_for_llm(self):
        capture = MagicMock()
        vad = MagicMock()
        processor = MagicMock()
        overlay = MagicMock()

        class FakeRefiner:
            enabled = True

            def refine(self, text):
                return "polished " + text

        pipe = PipelineController(
            capture, vad, processor, overlay, refiner=FakeRefiner()
        )
        received = []
        pipe.translation_output.connect(
            lambda heard, translated: received.append((heard, translated))
        )
        pipe._state = PipelineState.RUNNING

        pipe._on_translation("こんにちは", "Hello")
        self.assertEqual(received, [("こんにちは", "Hello")])


if __name__ == "__main__":
    unittest.main()
