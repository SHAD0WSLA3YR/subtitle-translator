"""Unit tests for sentence-complete heuristics, playback stretch, and
the number of Whisper passes the processor runs per clause."""

import threading
import time
import unittest

import numpy as np

from src.stt.processor import TranslationProcessor, looks_complete, looks_complete_en
from src.stt.whisper_stt import WhisperSTT, is_untranslated, stretch_audio


class LooksCompleteTests(unittest.TestCase):
    def test_complete_desu_masu(self):
        self.assertTrue(looks_complete("ここは公民館と言います。"))
        self.assertTrue(looks_complete("会議が終わりました"))
        self.assertTrue(looks_complete("公園の方に行ってみましょう"))

    def test_incomplete_mid_sentence(self):
        self.assertFalse(looks_complete("こっちに行くと、さっきの通り"))
        self.assertFalse(looks_complete("テンスの向こうに"))
        self.assertFalse(looks_complete("と、見に行ってみましょう"[:2]))  # "と、"


class LooksCompleteEnTests(unittest.TestCase):
    def test_complete_sentences(self):
        self.assertTrue(looks_complete_en("This is the entrance."))
        self.assertTrue(looks_complete_en("Is this a shrine?"))
        self.assertTrue(looks_complete_en('He said "let\'s go."'))

    def test_incomplete_sentences(self):
        self.assertFalse(looks_complete_en("If I go this way"))
        self.assertFalse(looks_complete_en("Beyond the fence,"))

    def test_empty_counts_as_complete(self):
        self.assertTrue(looks_complete_en(""))


class UntranslatedDetectionTests(unittest.TestCase):
    def test_detects_raw_japanese(self):
        self.assertTrue(is_untranslated("先がない道を付き当たりと言います。"))
        self.assertTrue(is_untranslated("付き当たりの右は、さっき見た公園です。"))

    def test_plain_english_passes(self):
        self.assertFalse(is_untranslated("Let's go back the way we came."))
        self.assertFalse(is_untranslated(""))

    def test_quoted_japanese_term_is_allowed(self):
        self.assertFalse(
            is_untranslated('A "どひょう" is the ring where sumo wrestling takes place.')
        )


class ContextPoisoningTests(unittest.TestCase):
    """A Japanese line must not become the next clause's English prompt."""

    def test_japanese_output_is_not_stored_as_previous_english(self):
        stt = WhisperSTT()
        stt.commit_context("", "This is the entrance.")
        stt.commit_context("", "先がない道を付き当たりと言います。")
        self.assertEqual(stt._prev_en, "This is the entrance.")

    def test_english_output_replaces_previous(self):
        stt = WhisperSTT()
        stt.commit_context("", "First line.")
        stt.commit_context("", "Second line.")
        self.assertEqual(stt._prev_en, "Second line.")


class TranslateRetryTests(unittest.TestCase):
    def _stt_with_runs(self, outputs):
        stt = WhisperSTT()
        stt._model = object()  # bypass model loading
        stt.load = lambda: None
        calls = []

        def fake_run(audio, task, initial_prompt=None):
            calls.append(initial_prompt)
            return outputs[len(calls) - 1]

        stt._run = fake_run
        return stt, calls

    def test_untranslated_result_retries_without_prompt(self):
        stt, calls = self._stt_with_runs(
            ["先がない道を付き当たりと言います。", "A dead-end road."]
        )
        stt._prev_en = "Let's go back the way we came."
        result = stt.translate_to_english(np.zeros(16000, dtype=np.float32))

        self.assertEqual(result, "A dead-end road.")
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[0])
        self.assertIsNone(calls[1], "retry must drop the poisoned prompt")

    def test_good_result_does_not_retry(self):
        stt, calls = self._stt_with_runs(["A dead-end road."])
        stt._prev_en = "Previous line."
        result = stt.translate_to_english(np.zeros(16000, dtype=np.float32))

        self.assertEqual(result, "A dead-end road.")
        self.assertEqual(len(calls), 1)

    def test_no_prompt_means_no_retry(self):
        stt, calls = self._stt_with_runs(["先がない道を付き当たりと言います。"])
        result = stt.translate_to_english(np.zeros(16000, dtype=np.float32))

        self.assertEqual(len(calls), 1)
        self.assertTrue(is_untranslated(result))


class FakeSTT:
    """Counts passes so we can assert how much Whisper work is requested."""

    def __init__(self, heard=(), translated=()):
        self._heard = list(heard)
        self._translated = list(translated)
        self.transcribe_calls = 0
        self.translate_calls = 0
        self.playback_speed = 1.0

    def transcribe_source(self, audio):
        self.transcribe_calls += 1
        return self._heard.pop(0) if self._heard else ""

    def translate_to_english(self, audio, heard=""):
        self.translate_calls += 1
        return self._translated.pop(0) if self._translated else ""

    def commit_context(self, heard, translated):
        pass


def _run_clauses(processor, clauses, expected_emits):
    """Drive the processor loop directly, bypassing model loading."""
    emitted = []
    processor._on_translation = lambda h, t: emitted.append((h, t))
    processor._running = True
    thread = threading.Thread(target=processor._process_loop, daemon=True)
    thread.start()
    for clause in clauses:
        processor._queue.put(clause)

    deadline = time.monotonic() + 5.0
    while len(emitted) < expected_emits and time.monotonic() < deadline:
        time.sleep(0.01)

    processor._running = False
    thread.join(timeout=5)
    return emitted


class PassCountTests(unittest.TestCase):
    """The source-language pass doubles GPU cost, so it must stay opt-in."""

    def setUp(self):
        self.audio = np.zeros(16000, dtype=np.float32)

    def _processor(self, fake, **kwargs):
        proc = TranslationProcessor(**kwargs)
        proc._stt = fake
        return proc

    def test_single_pass_skips_source_transcription(self):
        fake = FakeSTT(translated=["This is the entrance."])
        proc = self._processor(fake, dual_pass=False)
        emitted = _run_clauses(proc, [self.audio], 1)

        self.assertEqual(fake.transcribe_calls, 0)
        self.assertEqual(fake.translate_calls, 1)
        self.assertEqual(emitted, [("", "This is the entrance.")])

    def test_dual_pass_skips_translation_of_held_clause(self):
        fake = FakeSTT(
            heard=["こっちに行くと", "こっちに行くと、さっきの通りに戻ります。"],
            translated=["If I go this way, I'll be back on the road."],
        )
        proc = self._processor(fake, dual_pass=True)
        emitted = _run_clauses(proc, [self.audio, self.audio], 1)

        self.assertEqual(fake.transcribe_calls, 2)
        self.assertEqual(fake.translate_calls, 1, "held clause must not be translated")
        self.assertEqual(len(emitted), 1)
        self.assertEqual(proc.clauses_merged, 1)

    def test_single_pass_merges_on_incomplete_english(self):
        fake = FakeSTT(
            translated=["Beyond the fence,", "Beyond the fence, there's something."]
        )
        proc = self._processor(fake, dual_pass=False)
        emitted = _run_clauses(proc, [self.audio, self.audio], 1)

        self.assertEqual(fake.transcribe_calls, 0)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0][1], "Beyond the fence, there's something.")


class StretchAudioTests(unittest.TestCase):
    def test_noop_at_one(self):
        audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        out = stretch_audio(audio, 1.0)
        self.assertEqual(len(out), len(audio))

    def test_stretch_1_25(self):
        audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        out = stretch_audio(audio, 1.25)
        self.assertAlmostEqual(len(out) / len(audio), 1.25, places=2)

    def test_compress_0_75(self):
        audio = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        out = stretch_audio(audio, 0.75)
        self.assertAlmostEqual(len(out) / len(audio), 0.75, places=2)

    def test_clamp_range(self):
        from src.stt.whisper_stt import clamp_playback_speed
        self.assertEqual(clamp_playback_speed(0.5), 0.75)
        self.assertEqual(clamp_playback_speed(2.0), 1.5)
        self.assertEqual(clamp_playback_speed(1.25), 1.25)


if __name__ == "__main__":
    unittest.main()
