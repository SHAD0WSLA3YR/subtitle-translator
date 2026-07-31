"""UtteranceEvent hashes and lifecycle log formats."""

import unittest

import numpy as np

from src.stt.events import (
    UtteranceEvent,
    UtteranceStatus,
    audio_hash,
    source_hash,
    translation_hash,
)


class EventHashTests(unittest.TestCase):
    def test_source_hash_stable(self):
        self.assertEqual(source_hash("abc"), source_hash("abc"))
        self.assertNotEqual(source_hash("abc"), source_hash("abd"))

    def test_audio_hash_differs_for_different_audio(self):
        a = np.zeros(1600, dtype=np.float32)
        b = np.ones(1600, dtype=np.float32)
        self.assertNotEqual(audio_hash(a), audio_hash(b))

    def test_translation_hash_matches_source_hash_helper(self):
        self.assertEqual(translation_hash("hello"), source_hash("hello"))

    def test_log_formats(self):
        ev = UtteranceEvent(
            utterance_id=3,
            audio_start_s=1.0,
            audio_end_s=3.5,
            source_text="猫がいる",
            translation_en="There is a cat",
        )
        self.assertIn("id=3", ev.log_asr_final())
        self.assertIn("ASR_FINAL", ev.log_asr_final())
        self.assertIn("src=", ev.log_asr_final())
        self.assertIn("MT_DONE", ev.log_mt_done())
        self.assertIn("en=", ev.log_mt_done())
        self.assertIn("OVERLAY_COMMIT", ev.log_overlay_commit())
        self.assertEqual(ev.status, UtteranceStatus.ASR_FINAL)


if __name__ == "__main__":
    unittest.main()
