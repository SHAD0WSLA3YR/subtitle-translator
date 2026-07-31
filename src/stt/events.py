"""Utterance lifecycle events for streaming JA→EN observability."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class UtteranceStatus(str, Enum):
    ASR_FINAL = "asr_final"
    MT_DONE = "mt_done"
    OVERLAY_COMMIT = "overlay_commit"
    STALE = "stale"
    DROPPED_STALE = "dropped_stale"


def source_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def audio_hash(audio: np.ndarray) -> str:
    if audio is None or len(audio) == 0:
        return "000000000000"
    # Stable fingerprint without storing full audio in logs.
    payload = audio.tobytes()
    return hashlib.sha1(payload).hexdigest()[:12]


def translation_hash(text: str) -> str:
    return source_hash(text)


@dataclass
class UtteranceEvent:
    """One finalized utterance from VAD flush through overlay commit."""

    utterance_id: int
    audio_start_s: float
    audio_end_s: float
    source_text: str
    detected_lang: str = "ja"
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    audio_hash: str = ""
    source_hash: str = ""
    translation_en: str = ""
    translation_hash: str = ""
    status: UtteranceStatus = UtteranceStatus.ASR_FINAL
    mt_backend: str = ""
    asr_final_at: float = field(default_factory=time.monotonic)
    mt_latency_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.source_hash and self.source_text:
            self.source_hash = source_hash(self.source_text)
        if self.audio is not None and len(self.audio) and not self.audio_hash:
            self.audio_hash = audio_hash(self.audio)

    @property
    def time_range(self) -> str:
        return f"{self.audio_start_s:.2f}s→{self.audio_end_s:.2f}s"

    def log_asr_final(self) -> str:
        return (
            f"[id={self.utterance_id} | {self.time_range} | ASR_FINAL | "
            f"src={self.source_hash}] JA: {self.source_text[:80]}"
        )

    def log_mt_done(self) -> str:
        return (
            f"[id={self.utterance_id} | MT_DONE | src={self.source_hash} | "
            f"en={self.translation_hash} | latency={self.mt_latency_s:.1f}s] "
            f"EN: {(self.translation_en or '')[:80]}"
        )

    def log_overlay_commit(self) -> str:
        return f"[id={self.utterance_id} | OVERLAY_COMMIT | src={self.source_hash}]"

    def log_dropped_stale(self, age_s: float) -> str:
        return (
            f"[id={self.utterance_id} | DROPPED_STALE | age={age_s:.1f}s | "
            f"src={self.source_hash}]"
        )
