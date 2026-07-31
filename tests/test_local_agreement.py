"""LocalAgreement-n + flush rules (Whisper-Streaming style).

Run:  python -m unittest tests.test_local_agreement -v
"""

import unittest

from src.stt.local_agreement import (
    Word,
    join_words,
    local_agreement_n,
    should_flush,
)


def _w(*parts):
    """Build timed words: ('猫', 0.0, 0.3), ..."""
    out = []
    t = 0.0
    for p in parts:
        if isinstance(p, tuple):
            text, start, end = p
            out.append(Word(text=text, start=start, end=end))
        else:
            out.append(Word(text=p, start=t, end=t + 0.3))
            t += 0.3
    return out


class LocalAgreementTests(unittest.TestCase):
    def test_empty_prev_commits_nothing(self):
        new = _w("猫", "が", "マット")
        self.assertEqual(local_agreement_n([], new), [])

    def test_lcp_commits_stable_prefix(self):
        prev = _w("猫", "が", "マット", "の")
        new = _w("猫", "が", "マット", "の", "上", "に")
        committed = local_agreement_n(prev, new, n=2)
        self.assertEqual(join_words(committed), "猫がマットの")

    def test_disagreement_commits_nothing_past_diverge(self):
        prev = _w("猫", "が", "マット")
        new = _w("犬", "が", "マット")
        self.assertEqual(local_agreement_n(prev, new), [])

    def test_join_cjk_no_spaces(self):
        self.assertEqual(join_words(_w("猫", "が", "マット")), "猫がマット")

    def test_join_latin_spaces(self):
        self.assertEqual(join_words(_w("the", "cat")), "the cat")


class FlushRuleTests(unittest.TestCase):
    def test_sentence_ending(self):
        self.assertTrue(should_flush("終わり。"))
        self.assertTrue(should_flush("Done!"))

    def test_silence(self):
        self.assertTrue(should_flush("まだ", silence_ms=600, silence_threshold_ms=600))
        self.assertFalse(should_flush("まだ", silence_ms=100, silence_threshold_ms=600))

    def test_char_cap(self):
        long_ja = "あ" * 45
        self.assertTrue(should_flush(long_ja, max_chars=45))
        self.assertFalse(should_flush("短い", max_chars=45))

    def test_hard_timeout(self):
        self.assertTrue(
            should_flush("途中", hard_timeout_s=4.0, hard_timeout_threshold_s=4.0)
        )

    def test_empty_never_flushes(self):
        self.assertFalse(should_flush(""))
        self.assertFalse(should_flush("   ", silence_ms=999))


if __name__ == "__main__":
    unittest.main()
