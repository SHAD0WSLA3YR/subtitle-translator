"""Near-real-time latency path: capability profiles, lag governor, provisional UI.

Run:  python -m unittest tests.test_latency -v
"""

import os
import threading
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.core.capability import (
    CapabilityProfile,
    apply_profile_to_config,
    detect_capability,
)
from src.stt.processor import TranslationProcessor


class CapabilityTests(unittest.TestCase):
    def test_detect_returns_profile(self):
        profile = detect_capability()
        self.assertIsInstance(profile, CapabilityProfile)
        self.assertIn(profile.device, ("cuda", "cpu"))
        self.assertGreaterEqual(profile.beam_size, 1)
        self.assertLessEqual(profile.beam_size, 2)

    def test_low_vram_profile_forces_beam_1_keeps_small(self):
        profile = CapabilityProfile(
            device="cuda", gpu_name="MX450", vram_gb=2.0,
            model_size="small", beam_size=1, compute_type="int8_float16",
            cpu_threads=4, min_silence_ms=400, max_speech_ms=6000,
            partial_interval_ms=700, max_queued=1, reason="test",
        )
        cfg = {
            "model": {"size": "small", "beam_size": 3, "device": "cuda"},
            "vad": {"min_silence_duration_ms": 1100},
            "latency": {"auto_tune": True},
        }
        apply_profile_to_config(cfg, profile)
        self.assertEqual(cfg["model"]["beam_size"], 1)
        self.assertEqual(cfg["model"]["size"], "small")
        self.assertEqual(cfg["vad"]["min_silence_duration_ms"], 400)
        self.assertTrue(cfg["latency"]["live_partials"])

    def test_auto_tune_off_preserves_user_beam(self):
        profile = CapabilityProfile(
            device="cuda", gpu_name="MX450", vram_gb=2.0,
            model_size="base", beam_size=1, compute_type="int8_float16",
            cpu_threads=4, min_silence_ms=950, max_speech_ms=5500,
            partial_interval_ms=1800, max_queued=1, reason="test",
        )
        cfg = {
            "model": {"size": "small", "beam_size": 3},
            "latency": {"auto_tune": False},
        }
        apply_profile_to_config(cfg, profile)
        self.assertEqual(cfg["model"]["beam_size"], 3)
        self.assertEqual(cfg["model"]["size"], "small")


class FakeSTT:
    def __init__(self, translated=(), detected="ja"):
        self._translated = list(translated)
        self.transcribe_calls = 0
        self.translate_calls = 0
        self.playback_speed = 1.0
        self.beam_size = 1
        self.language = "auto"
        self.detected_language = detected
        self.language_probability = 0.95
        self._locked_language = ""
        self._prev_en = ""
        self._prev_ja = ""

    @property
    def locked_language(self):
        return self._locked_language

    def lock_language(self, code):
        self._locked_language = code

    def unlock_language(self):
        self._locked_language = ""

    def transcribe_source(self, audio, lang_hint=""):
        self.transcribe_calls += 1
        return ""

    def translate_to_english(self, audio, heard=""):
        self.translate_calls += 1
        # Simulate a slow decode so RTF climbs above the governor threshold.
        time.sleep(0.05)
        text = self._translated.pop(0) if self._translated else "ok."
        return text, self.detected_language

    def commit_context(self, heard, translated):
        pass


class FakeNMT:
    def translate(self, text, src, tgt):
        return text

    def ensure_pair(self, src, tgt):
        return True

    def retry_pair(self, src, tgt):
        pass

    def is_ready(self, src, tgt):
        return True


def _drive(processor, clauses, expected, timeout=5.0):
    emitted = []
    partials = []
    processor._on_translation = lambda h, t, lang="", _ev=None: emitted.append((h, t))
    processor._on_partial = lambda t: partials.append(t)
    processor._running = True
    thread = threading.Thread(target=processor._process_loop, daemon=True)
    thread.start()
    for c in clauses:
        processor.add_clause(c)
    deadline = time.monotonic() + timeout
    while len(emitted) < expected and time.monotonic() < deadline:
        time.sleep(0.01)
    processor._running = False
    thread.join(timeout=5)
    return emitted, partials


class LagGovernorTests(unittest.TestCase):
    def test_max_queued_1_drops_backlog(self):
        audio = np.zeros(8000, dtype=np.float32)
        fake = FakeSTT(translated=["one.", "two.", "three."])
        proc = TranslationProcessor(max_queued=1, lag_governor=True, merge_incomplete=False)
        proc._stt = fake
        proc._nmt = FakeNMT()

        # Flood the queue faster than FakeSTT can drain it.
        proc._running = True
        proc.add_clause(audio)
        proc.add_clause(audio)
        proc.add_clause(audio)
        self.assertLessEqual(proc._queue.qsize(), 1)
        self.assertGreaterEqual(proc.clauses_dropped, 1)
        proc._running = False

    def test_partial_uses_tail_window(self):
        long_audio = np.zeros(16000 * 10, dtype=np.float32)  # 10s
        proc = TranslationProcessor(partial_tail_seconds=3.5)
        proc._running = True
        proc.add_partial(long_audio)
        snapped = proc._take_partial()
        self.assertIsNotNone(snapped)
        self.assertLessEqual(len(snapped), int(16000 * 3.5) + 1)
        proc._running = False

    def test_behind_skips_partial_decode(self):
        audio = np.zeros(16000, dtype=np.float32)
        fake = FakeSTT(translated=["Hello."])
        proc = TranslationProcessor(lag_governor=True)
        proc._stt = fake
        proc._nmt = FakeNMT()
        proc._behind = True
        proc._running = True
        proc.add_partial(audio)
        # add_partial should ignore while behind
        self.assertIsNone(proc._take_partial())
        proc._running = False


class ProvisionalOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        cls._app = QApplication.instance() or QApplication([])

    def test_provisional_replaced_by_final(self):
        from src.ui.overlay import SubtitleOverlay

        overlay = SubtitleOverlay(width=800, height=180)
        overlay.show_provisional("draft line")
        self.assertTrue(overlay._has_provisional)
        self.assertEqual(overlay._text.text(), "draft line")

        overlay.show_subtitle("final line")
        self.assertFalse(overlay._has_provisional)
        self.assertEqual(overlay._text.text(), "final line")
        # Must be one line, not draft+final stacked.
        self.assertEqual(len(overlay._current_texts), 1)
        overlay.close()

    def test_provisional_updates_in_place(self):
        from src.ui.overlay import SubtitleOverlay

        overlay = SubtitleOverlay(width=800, height=180)
        overlay.show_provisional("a")
        overlay.show_provisional("a b")
        overlay.show_provisional("a b c")
        self.assertEqual(len(overlay._current_texts), 1)
        self.assertEqual(overlay._text.text(), "a b c")
        overlay.close()

    def test_ensure_visible_recovers_from_fade_out(self):
        from src.ui.overlay import SubtitleOverlay

        overlay = SubtitleOverlay(width=800, height=180)
        overlay.show_subtitle("before")
        # Simulate fade-out completion leaving the window hidden.
        overlay.hide()
        overlay.setWindowOpacity(0.0)
        overlay.show_subtitle("after recovery")
        self.assertFalse(overlay.isHidden())
        self.assertGreaterEqual(overlay.windowOpacity(), 0.99)
        self.assertIn("after recovery", overlay._text.text())
        overlay.close()


class LanguageLockTests(unittest.TestCase):
    def test_locks_after_two_confident_detections(self):
        from src.stt.whisper_stt import WhisperSTT

        stt = WhisperSTT(language="auto")
        proc = TranslationProcessor()
        proc._stt = stt
        stt._language_probability = 0.92
        proc._consider_lang_lock("ja", 2.5)
        self.assertEqual(stt.locked_language, "")  # one hit not enough
        proc._consider_lang_lock("ja", 2.5)
        self.assertEqual(stt.locked_language, "ja")

    def test_ignores_short_noisy_detection(self):
        from src.stt.whisper_stt import WhisperSTT

        stt = WhisperSTT(language="auto")
        proc = TranslationProcessor()
        proc._stt = stt
        stt._language_probability = 0.95
        proc._consider_lang_lock("pl", 0.8)  # too short
        proc._consider_lang_lock("pl", 0.8)
        self.assertEqual(stt.locked_language, "")

    def test_does_not_sticky_lock_english_from_intro(self):
        from src.stt.whisper_stt import WhisperSTT

        stt = WhisperSTT(language="auto")
        proc = TranslationProcessor()
        proc._stt = stt
        stt._language_probability = 0.90
        for _ in range(3):
            proc._consider_lang_lock("en", 2.5)
        self.assertEqual(stt.locked_language, "")  # needs 0.95 + 3s + 3 hits

    def test_unlocks_en_when_japanese_appears(self):
        from src.stt.whisper_stt import WhisperSTT

        stt = WhisperSTT(language="auto")
        proc = TranslationProcessor()
        proc._stt = stt
        stt.lock_language("en")
        stt._language_probability = 0.70
        proc._consider_lang_lock("ja", 1.5)
        self.assertEqual(stt.locked_language, "ja")


if __name__ == "__main__":
    unittest.main()
