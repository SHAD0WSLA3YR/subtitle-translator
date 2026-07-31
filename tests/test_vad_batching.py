"""VAD 3–6s batching: merge short gaps, max split, pre-roll pad.

Run:  python -m unittest tests.test_vad_batching -v
"""

from __future__ import annotations

import unittest
from typing import List

import numpy as np

from src.audio.vad import VADProcessor

SR = 16000
WIN = 512


class _Prob:
    def __init__(self, value: float):
        self._value = value

    def item(self) -> float:
        return self._value


class _AlwaysSpeechVad:
    """Deterministic stand-in for Silero: any non-floor window is speech."""

    def __call__(self, tensor, sample_rate):
        return _Prob(0.95)

    def reset_states(self):
        pass


def _sine(seconds: float, amp: float = 0.3) -> np.ndarray:
    n = int(SR * seconds)
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def _feed(vad: VADProcessor, audio: np.ndarray) -> None:
    for i in range(0, len(audio), WIN):
        vad.process_chunk(audio[i : i + WIN])


def _make_vad(**kwargs) -> tuple[VADProcessor, List[np.ndarray]]:
    clauses: List[np.ndarray] = []
    defaults = dict(
        threshold=0.3,
        min_speech_duration_ms=200,
        min_silence_duration_ms=400,
        merge_silence_ms=400,
        target_min_speech_ms=3000,
        speech_pad_ms=250,
        max_speech_duration_ms=6000,
        silence_floor=0.004,
        on_clause=clauses.append,
        on_partial=None,
        _vad_model=_AlwaysSpeechVad(),
    )
    defaults.update(kwargs)
    vad = VADProcessor(**defaults)
    return vad, clauses


class VadBatchingTests(unittest.TestCase):
    def test_short_gap_does_not_split(self):
        vad, clauses = _make_vad(target_min_speech_ms=0)
        # 1.5s speech, 200ms gap, 1.5s speech, then long silence to end.
        _feed(vad, _sine(1.5))
        _feed(vad, _silence(0.2))
        _feed(vad, _sine(1.5))
        _feed(vad, _silence(1.0))
        self.assertEqual(len(clauses), 1)
        dur = len(clauses[0]) / SR
        self.assertGreaterEqual(dur, 2.5)
        self.assertLessEqual(dur, 4.5)

    def test_long_gap_ends_clause(self):
        vad, clauses = _make_vad(target_min_speech_ms=0)
        _feed(vad, _sine(1.5))
        _feed(vad, _silence(0.6))
        _feed(vad, _sine(1.0))
        _feed(vad, _silence(1.0))
        self.assertGreaterEqual(len(clauses), 2)
        self.assertLess(len(clauses[0]) / SR, 2.5)

    def test_continuous_speech_splits_near_max(self):
        vad, clauses = _make_vad(target_min_speech_ms=0, max_speech_duration_ms=6000)
        _feed(vad, _sine(7.0))
        _feed(vad, _silence(1.0))
        self.assertGreaterEqual(len(clauses), 2)
        first = len(clauses[0]) / SR
        self.assertGreaterEqual(first, 5.5)
        self.assertLessEqual(first, 6.5)

    def test_target_min_holds_early_silence(self):
        vad, clauses = _make_vad(target_min_speech_ms=3000)
        # ~1.2s speech then 500ms silence — under target, hold until hold_silence (~800ms).
        _feed(vad, _sine(1.2))
        _feed(vad, _silence(0.5))
        self.assertEqual(len(clauses), 0)
        _feed(vad, _silence(0.5))  # total silence ~1.0s ≥ hold
        self.assertEqual(len(clauses), 1)

    def test_pre_roll_pad_applied(self):
        vad, clauses = _make_vad(target_min_speech_ms=0, speech_pad_ms=250)
        _feed(vad, _silence(0.5))  # fill pre-roll
        _feed(vad, _sine(1.0))
        _feed(vad, _silence(0.8))
        self.assertEqual(len(clauses), 1)
        # Emitted audio should exceed pure speech (~1s) by roughly pad + trailing silence.
        self.assertGreaterEqual(len(clauses[0]) / SR, 1.15)


if __name__ == "__main__":
    unittest.main()
