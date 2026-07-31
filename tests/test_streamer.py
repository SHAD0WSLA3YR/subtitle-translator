"""StreamingSession + TranslateWorker (mocked STT / translator).

Run:  python -m unittest tests.test_streamer -v
"""

import time
import unittest

import numpy as np

from src.stt.local_agreement import Word
from src.stt.streamer import StreamingSession, TranslateWorker, FlushUnit, OrderedOverlayDrain
from src.stt.events import UtteranceEvent, UtteranceStatus
from src.translate.rate_limit import RateBudget
from src.translate.contextual import ContextualTranslator, source_hash


class FakeWordSTT:
    def __init__(self, hyps=None):
        self._hyps = list(hyps or [])
        self.i = 0
        self.beam_size = 1
        self.language = "ja"
        self.detected_language = "ja"
        self._locked_language = "ja"
        self._prev_en = ""
        self._prev_ja = ""
        self.transcribe_calls = 0

    @property
    def locked_language(self):
        return self._locked_language

    def lock_language(self, code):
        self._locked_language = code

    def unlock_language(self):
        self._locked_language = ""

    def commit_context(self, heard, translated):
        pass

    def transcribe_words(self, audio, lang_hint="", beam_size=None, initial_prompt=None):
        if self.i >= len(self._hyps):
            return self._hyps[-1] if self._hyps else ([], "", "ja", 0.0)
        out = self._hyps[self.i]
        self.i += 1
        self.detected_language = out[2]
        return out

    def transcribe_source(self, audio, lang_hint=""):
        self.transcribe_calls += 1
        return "再認識された文"


def _words(*texts):
    t = 0.0
    out = []
    for text in texts:
        out.append(Word(text=text, start=t, end=t + 0.4))
        t += 0.4
    return out


class StreamingSessionTests(unittest.TestCase):
    def test_commits_on_second_agreeing_hypothesis(self):
        w1 = _words("猫", "が", "マット")
        w2 = _words("猫", "が", "マット", "の")
        stt = FakeWordSTT([
            (w1, "猫がマット", "ja", 0.9),
            (w2, "猫がマットの", "ja", 0.9),
        ])
        session = StreamingSession(
            stt, agreement_n=2, flush_chars=999, flush_timeout_s=99,
            min_final_seconds=0.5,
        )
        audio = np.zeros(16000 * 3, dtype=np.float32)
        session.update(audio)
        draft2, flush2 = session.update(audio)
        self.assertIsNone(flush2)
        self.assertEqual(session.committed_unflushed_text, "猫がマット")
        self.assertIn("の", draft2)

    def test_silence_promotes_draft_and_flushes(self):
        # Word end times must span >= min_final_seconds so the flush is kept.
        w1 = [Word(text="こんにちは", start=0.0, end=1.5)]
        w2 = [Word(text="こんにちは", start=0.0, end=1.5)]
        stt = FakeWordSTT([
            (w1, "こんにちは", "ja", 0.9),
            (w2, "こんにちは", "ja", 0.9),
        ])
        session = StreamingSession(
            stt, flush_chars=999, flush_timeout_s=99, min_final_seconds=0.5,
        )
        audio = np.zeros(16000 * 3, dtype=np.float32)
        session.update(audio)
        session.update(audio)
        unit = session.on_silence(700.0)
        if unit is None:
            unit = session.force_flush_remaining()
        self.assertIsNotNone(unit)
        self.assertIn("こんにちは", unit.source_text)
        self.assertGreater(unit.utterance_id, 0)
        self.assertTrue(unit.source_hash)

    def test_short_audio_not_flushed(self):
        w1 = _words("あ")
        w2 = _words("あ")
        stt = FakeWordSTT([(w1, "あ", "ja", 0.9), (w2, "あ", "ja", 0.9)])
        session = StreamingSession(
            stt, flush_chars=1, flush_timeout_s=0.01, min_final_seconds=1.2,
        )
        # Confirmed region will be tiny after LA commit.
        audio = np.zeros(16000 * 2, dtype=np.float32)
        session.update(audio)
        session.update(audio)
        unit = session.force_flush_remaining()
        # Either None (short) or requires enough samples — with tiny words end~0.4s, None.
        if unit is not None:
            self.assertGreaterEqual(len(unit.audio) / 16000, 1.2)

    def test_mid_utt_char_flush_gated_by_min_final(self):
        """Dense JA chars must not flush until committed audio ≥ min_final."""
        long = "あ" * 90
        w1 = [Word(text=long, start=0.0, end=0.5)]
        w2 = [Word(text=long, start=0.0, end=0.5)]
        stt = FakeWordSTT([(w1, long, "ja", 0.9), (w2, long, "ja", 0.9)])
        session = StreamingSession(
            stt,
            flush_chars=80,
            flush_timeout_s=6.0,
            min_final_seconds=1.2,
        )
        audio = np.zeros(16000 * 3, dtype=np.float32)
        session.update(audio)
        draft, flush = session.update(audio)
        self.assertIsNone(flush)
        self.assertTrue(len(session.committed_unflushed_text.replace(" ", "")) >= 80)
        self.assertIn("あ", draft)

    def test_defaults_raised_for_batching(self):
        session = StreamingSession(FakeWordSTT([]))
        self.assertEqual(session._flush_chars, 80)
        self.assertEqual(session._flush_timeout_s, 6.0)

    def test_silence_flush_uses_full_speech_after_partial_commit(self):
        """Silence flush must cover enough audio even after a short LA commit."""
        # First agreeing pair commits only ~0.4s; draft continues.
        early = [Word(text="猫", start=0.0, end=0.4)]
        full = [
            Word(text="猫", start=0.0, end=0.4),
            Word(text="が", start=0.4, end=0.8),
            Word(text="いる", start=0.8, end=1.5),
        ]
        stt = FakeWordSTT([
            (early, "猫", "ja", 0.9),
            (early, "猫", "ja", 0.9),
            (full, "猫がいる", "ja", 0.9),
        ])
        session = StreamingSession(
            stt, flush_chars=999, flush_timeout_s=99, min_final_seconds=1.2,
        )
        audio = np.zeros(16000 * 3, dtype=np.float32)
        session.update(audio)
        session.update(audio)  # commits "猫" with confirmed ~0.4s
        session.update(audio)  # advances draft; confirmed still short
        unit = session.on_silence(700.0)
        if unit is None:
            unit = session.force_flush_remaining()
        self.assertIsNotNone(unit)
        self.assertGreaterEqual(len(unit.audio) / 16000, 1.2)
        self.assertIn("猫", unit.source_text)

    def test_consecutive_flushes_do_not_overlap_audio(self):
        """Mid-utt flush then silence flush must emit deltas, not rewind."""
        # Force an early char flush once confirmed audio is long enough.
        chunk_a = "あ" * 90
        w_a = [Word(text=chunk_a, start=0.0, end=2.0)]
        w_a2 = [Word(text=chunk_a, start=0.0, end=2.0)]
        # After first flush, new committed tail.
        chunk_b = "い" * 20
        w_b = [Word(text=chunk_b, start=0.0, end=1.5)]
        w_b2 = [Word(text=chunk_b, start=0.0, end=1.5)]
        stt = FakeWordSTT([
            (w_a, chunk_a, "ja", 0.9),
            (w_a2, chunk_a, "ja", 0.9),
            (w_b, chunk_b, "ja", 0.9),
            (w_b2, chunk_b, "ja", 0.9),
        ])
        session = StreamingSession(
            stt,
            flush_chars=80,
            flush_timeout_s=99,
            min_final_seconds=1.2,
        )
        audio1 = np.zeros(16000 * 3, dtype=np.float32)
        session.update(audio1)
        _draft, flush1 = session.update(audio1)
        self.assertIsNotNone(flush1)
        end1 = flush1.audio_end_s

        # Continue speech with more samples; silence flushes the delta.
        audio2 = np.zeros(16000 * 5, dtype=np.float32)
        session.update(audio2)
        session.update(audio2)
        flush2 = session.on_silence(700.0) or session.force_flush_remaining()
        self.assertIsNotNone(flush2)
        self.assertGreaterEqual(flush2.audio_start_s, end1 - 1e-3)
        self.assertGreater(flush2.audio_end_s, flush2.audio_start_s)
        # No shared prefix window like the old 8s lookback.
        self.assertNotAlmostEqual(flush1.audio_start_s, flush2.audio_start_s, places=2)

    def test_absolute_timeline_on_flush(self):
        w1 = [Word(text="こんにちは", start=0.0, end=1.5)]
        w2 = [Word(text="こんにちは", start=0.0, end=1.5)]
        stt = FakeWordSTT([(w1, "こんにちは", "ja", 0.9), (w2, "こんにちは", "ja", 0.9)])
        session = StreamingSession(
            stt, flush_chars=999, flush_timeout_s=99, min_final_seconds=0.5,
        )
        session.set_timeline_origin(16000 * 10)  # 10s into the session
        audio = np.zeros(16000 * 3, dtype=np.float32)
        session.update(audio)
        session.update(audio)
        unit = session.on_silence(700.0) or session.force_flush_remaining()
        self.assertIsNotNone(unit)
        self.assertGreaterEqual(unit.audio_start_s, 10.0)
        self.assertGreater(unit.audio_end_s, unit.audio_start_s)
        self.assertGreater(unit.utterance_id, 0)
        self.assertTrue(unit.source_hash)


class RateBudgetTests(unittest.TestCase):
    def test_rpm_exhaustion(self):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "budget.json"
        budget = RateBudget(rpm_limit=2, weekly_limit=100, state_path=path)
        self.assertTrue(budget.try_consume())
        self.assertTrue(budget.try_consume())
        self.assertFalse(budget.try_consume())

    def test_weekly_exhaustion(self):
        import tempfile
        from pathlib import Path

        path = Path(tempfile.mkdtemp()) / "budget2.json"
        budget = RateBudget(rpm_limit=100, weekly_limit=1, state_path=path)
        self.assertTrue(budget.try_consume())
        self.assertFalse(budget.try_consume())


class ContextualTranslatorTests(unittest.TestCase):
    def test_passthrough_without_key(self):
        class FakeNMT:
            def translate(self, text, src, tgt):
                return None

        tr = ContextualTranslator(
            nmt=FakeNMT(),
            api_key_env="__MISSING_NVIDIA_KEY__",
            local_provider="argos",
            use_nvidia=False,
        )
        tr._api_key = None
        out, backend = tr.translate("猫がいる", "ja", "en")
        self.assertEqual(out, "猫がいる")
        self.assertEqual(backend, "passthrough")

    def test_argos_backup(self):
        class FakeNMT:
            def translate(self, text, src, tgt):
                return "There is a cat"

        tr = ContextualTranslator(
            nmt=FakeNMT(),
            api_key_env="__MISSING_NVIDIA_KEY__",
            local_provider="argos",
            use_nvidia=False,
        )
        tr._api_key = None
        out, backend = tr.translate("猫がいる", "ja", "en")
        self.assertEqual(out, "There is a cat")
        self.assertEqual(backend, "argos")

    def test_source_hash_stable(self):
        self.assertEqual(source_hash("abc"), source_hash("abc"))
        self.assertNotEqual(source_hash("abc"), source_hash("abd"))

    def test_rejects_prompt_leak_from_nvidia(self):
        class FakeNMT:
            def translate(self, text, src, tgt):
                return f"ARGOS:{text}"

        class FakeLocal:
            available = True

            def translate(self, text, src, tgt):
                # Must receive raw JA, never the chat template.
                self.last = text
                return None  # force fallthrough so nvidia is tried

        class FakeResp:
            status_code = 200

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Previous pairs:\n(none)\n\n"
                                    "Translate to English:\nHello"
                                )
                            }
                        }
                    ]
                }

        local = FakeLocal()
        tr = ContextualTranslator(
            nmt=FakeNMT(),
            local=local,
            local_provider="nllb",
            use_nvidia=True,
            api_key_env="__TEST__",
        )
        tr._api_key = "k"
        tr._budget.try_consume = lambda: True  # type: ignore

        import src.translate.contextual as ctx_mod

        original = ctx_mod.requests.post
        ctx_mod.requests.post = lambda *_a, **_k: FakeResp()
        try:
            out, backend = tr.translate("ごめん、今忙しいの", "ja", "en", utterance_id=1)
        finally:
            ctx_mod.requests.post = original

        self.assertEqual(local.last, "ごめん、今忙しいの")
        self.assertEqual(backend, "argos")
        self.assertTrue(out.startswith("ARGOS:"))
        self.assertNotIn("Previous pairs", out)

    def test_nllb_path_gets_raw_source_only(self):
        class FakeLocal:
            available = True

            def translate(self, text, src, tgt):
                self.seen = (text, src, tgt)
                return "Sorry, I'm busy right now."

        local = FakeLocal()
        tr = ContextualTranslator(
            nmt=None,
            local=local,
            local_provider="nllb",
            use_nvidia=False,
        )
        out, backend = tr.translate(
            "ごめん、今忙しいの。ちょっと待ってて。", "ja", "en", utterance_id=6
        )
        self.assertEqual(backend, "local-cpu")
        self.assertEqual(local.seen[0], "ごめん、今忙しいの。ちょっと待ってて。")
        self.assertNotIn("Previous pairs", local.seen[0])
        self.assertEqual(out, "Sorry, I'm busy right now.")

    def test_nvidia_401_circuit_breaker(self):
        class FakeNMT:
            def translate(self, text, src, tgt):
                return f"ARGOS:{text}"

        class FakeResp:
            status_code = 401

            def json(self):
                return {}

        tr = ContextualTranslator(
            nmt=FakeNMT(),
            api_key_env="__TEST__",
            local_provider="argos",
            use_nvidia=True,
        )
        tr._api_key = "bad-key"
        tr._budget.try_consume = lambda: True  # type: ignore

        calls = {"n": 0}

        def fake_post(*_a, **_k):
            calls["n"] += 1
            return FakeResp()

        import src.translate.contextual as ctx_mod

        original = ctx_mod.requests.post
        ctx_mod.requests.post = fake_post
        try:
            out1, b1 = tr.translate("一", "ja", "en", utterance_id=1)
            out2, b2 = tr.translate("二", "ja", "en", utterance_id=2)
        finally:
            ctx_mod.requests.post = original

        self.assertEqual(calls["n"], 1)  # second utterance skips NVIDIA
        self.assertTrue(tr._nvidia_hard_failed)
        self.assertFalse(tr.nvidia_enabled)
        self.assertEqual(b1, "argos")
        self.assertEqual(b2, "argos")
        self.assertIn("ARGOS", out1)
        self.assertIn("nvidia(dead)", tr.mt_chain_status())

    def test_mt_done_log_includes_latency(self):
        ev = UtteranceEvent(
            utterance_id=4,
            audio_start_s=10.4,
            audio_end_s=14.1,
            source_text="じゃあゆうじで",
            translation_en="Then Yuji",
            translation_hash="abc",
            mt_latency_s=1.3,
        )
        msg = ev.log_mt_done()
        self.assertIn("latency=1.3s", msg)
        self.assertIn("MT_DONE", msg)


class OrderedDrainTests(unittest.TestCase):
    def test_emits_in_utterance_id_order(self):
        seen = []

        def on_ready(ev):
            seen.append(ev.utterance_id)

        drain = OrderedOverlayDrain(on_ready)
        for uid in (1, 2, 3):
            drain.note_submitted(uid)
        for uid in (2, 1, 3):
            drain.push(UtteranceEvent(
                utterance_id=uid,
                audio_start_s=0.0,
                audio_end_s=1.0,
                source_text=f"t{uid}",
                translation_en=f"e{uid}",
            ))
        self.assertEqual(seen, [1, 2, 3])

    def test_stale_drop_marks_event(self):
        from src.stt.streamer import TranslateWorker
        from src.stt.events import UtteranceStatus

        seen = []

        class FakeSTT:
            beam_size = 1

            def lock_language(self, *_a):
                return None

            def transcribe_source(self, *_a, **_k):
                return "あとで。"

        class FakeNMT:
            def translate(self, text, src, tgt):
                return "Close"

        tr = ContextualTranslator(
            nmt=FakeNMT(),
            api_key_env="__NO__",
            local_provider="argos",
            use_nvidia=False,
        )
        tr._api_key = None
        worker = TranslateWorker(
            FakeSTT(),
            translator=tr,
            min_final_seconds=0.5,
            audio_clock_s=lambda: 40.0,  # live clock far ahead of event end
            max_subtitle_age_s=6.0,
            on_result=lambda *a: seen.append(a),
        )
        worker.start()
        worker.submit(FlushUnit(
            utterance_id=12,
            audio_start_s=30.0,
            audio_end_s=33.0,  # age = 7s > 6s
            source_text="あとで。",
            audio=np.zeros(16000 * 2, dtype=np.float32),
            detected_lang="ja",
        ))
        for _ in range(50):
            if seen:
                break
            time.sleep(0.05)
        worker.stop()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][5].status, UtteranceStatus.DROPPED_STALE)


class TranslateWorkerTests(unittest.TestCase):
    def test_async_translate_does_not_use_whisper_translate(self):
        stt = FakeWordSTT([])
        results = []

        class FakeNMT:
            def translate(self, text, src, tgt):
                return "EN-LOCAL"

        tr = ContextualTranslator(nmt=FakeNMT(), api_key_env="__NO__", local_provider="argos", use_nvidia=False)
        tr._api_key = None
        worker = TranslateWorker(
            stt,
            target_language="en",
            final_beam=3,
            translator=tr,
            min_final_seconds=0.5,
            on_result=lambda *a: results.append(a),
        )
        worker.start()
        worker.submit(FlushUnit(
            utterance_id=7,
            audio_start_s=1.0,
            audio_end_s=3.0,
            source_text="猫がいる",
            audio=np.zeros(16000 * 2, dtype=np.float32),
            detected_lang="ja",
            source_hash="abc",
        ))
        for _ in range(50):
            if results:
                break
            time.sleep(0.05)
        worker.stop()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][1], "EN-LOCAL")


if __name__ == "__main__":
    unittest.main()
