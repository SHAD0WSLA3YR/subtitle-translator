"""Guard: streaming runtime must never call Whisper task=translate."""

import ast
import os
import threading
import time
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.stt.processor import TranslationProcessor
from src.stt.streamer import StreamingSession, TranslateWorker, FlushUnit
from src.translate.contextual import ContextualTranslator


class _TranslateCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.hits = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "translate_to_english":
            self.hits.append(node.lineno)
        self.generic_visit(node)


class StreamingNoWhisperTranslateTests(unittest.TestCase):
    def test_streamer_module_has_no_translate_to_english_calls(self):
        import inspect
        from src.stt import streamer

        src = inspect.getsource(streamer)
        tree = ast.parse(src)
        visitor = _TranslateCallVisitor()
        visitor.visit(tree)
        self.assertEqual(visitor.hits, [], f"translate_to_english at lines {visitor.hits}")

    def test_streaming_partial_never_calls_whisper_translate(self):
        class GuardSTT:
            beam_size = 1
            language = "ja"
            detected_language = "ja"
            locked_language = "ja"

            def lock_language(self, code):
                pass

            def transcribe_words(self, audio, lang_hint="", beam_size=None, initial_prompt=None):
                return [], "", "ja", 0.9

            def translate_to_english(self, audio, heard=""):
                raise AssertionError("streaming must not call translate_to_english")

        proc = TranslationProcessor(streaming=True)
        proc._stt = GuardSTT()
        proc._session = StreamingSession(proc._stt, min_final_seconds=0.5)
        proc._running = True
        audio = np.zeros(16000 * 2, dtype=np.float32)
        proc._process_streaming_partial(audio)

    def test_translate_worker_uses_transcribe_source_only(self):
        class GuardSTT:
            beam_size = 1
            language = "ja"
            detected_language = "ja"
            locked_language = "ja"
            transcribe_calls = 0
            translate_calls = 0

            def lock_language(self, code):
                pass

            def transcribe_source(self, audio, lang_hint=""):
                self.transcribe_calls += 1
                return "猫"

            def translate_to_english(self, audio, heard=""):
                self.translate_calls += 1
                return "cat", "ja"

        stt = GuardSTT()
        results = []

        class FakeNMT:
            def translate(self, text, src, tgt):
                return f"EN:{text}"

        tr = ContextualTranslator(
            nmt=FakeNMT(),
            api_key_env="__NO_KEY__",
            local_provider="argos",
            use_nvidia=False,
        )
        tr._api_key = None
        worker = TranslateWorker(
            stt,
            translator=tr,
            min_final_seconds=0.5,
            on_result=lambda *a: results.append(a),
        )
        worker.start()
        worker.submit(FlushUnit(
            utterance_id=1,
            audio_start_s=0.0,
            audio_end_s=2.0,
            source_text="猫",
            audio=np.zeros(16000 * 2, dtype=np.float32),
            detected_lang="ja",
            source_hash="deadbeef0001",
        ))
        for _ in range(50):
            if results:
                break
            time.sleep(0.05)
        worker.stop()
        self.assertEqual(stt.translate_calls, 0)
        self.assertGreater(stt.transcribe_calls, 0)
        self.assertTrue(results)


class StaleMatchTests(unittest.TestCase):
    def test_contextual_rejects_hash_mismatch(self):
        class FakeNMT:
            def translate(self, text, src, tgt):
                return "EN"

        tr = ContextualTranslator(
            nmt=FakeNMT(),
            api_key_env="__NO__",
            local_provider="argos",
            use_nvidia=False,
        )
        tr._api_key = None
        out, backend = tr.translate(
            "猫",
            "ja",
            "en",
            utterance_id=5,
            expected_source_hash="wronghash000",
        )
        self.assertEqual(out, "")
        self.assertEqual(backend, "stale")


if __name__ == "__main__":
    unittest.main()
