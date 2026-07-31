"""LocalAgreement-n for Whisper streaming (Macháček et al. 2023).

Re-decode a growing buffer, compare consecutive hypotheses, and commit the
longest common prefix. Committed text never changes; only the draft tail
flickers — which is fine when it's visually marked as draft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Word:
    """One timed token from Whisper word_timestamps."""

    text: str
    start: float  # seconds relative to the decoded buffer start
    end: float

    def normalized(self) -> str:
        return self.text.strip()


def join_words(words: Sequence[Word]) -> str:
    """Join tokens with language-aware spacing (no spaces for CJK)."""
    if not words:
        return ""
    parts: List[str] = []
    for w in words:
        t = w.text.strip()
        if not t:
            continue
        if not parts:
            parts.append(t)
            continue
        # CJK / no-space scripts: glue. Latin etc.: space.
        prev = parts[-1]
        if _is_cjk(prev[-1]) or _is_cjk(t[0]):
            parts.append(t)
        else:
            parts.append(" " + t)
    return "".join(parts)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF  # kana
        or 0x3400 <= o <= 0x9FFF  # CJK
        or 0xAC00 <= o <= 0xD7AF  # hangul
        or 0xFF66 <= o <= 0xFF9D  # halfwidth kana
    )


def local_agreement_n(
    prev: Sequence[Word],
    new: Sequence[Word],
    n: int = 2,
) -> List[Word]:
    """Return newly-committable words = LCP of ``prev`` and ``new``.

    Classic LocalAgreement-2: when the same prefix appears in two consecutive
    hypotheses, that prefix is stable enough to commit. ``n`` is kept for API
    compatibility with the paper; with two hypotheses it is always pairwise.
    """
    if n < 2:
        n = 2
    if not prev or not new:
        return []
    limit = min(len(prev), len(new))
    k = 0
    while k < limit and prev[k].normalized() == new[k].normalized():
        k += 1
    return list(new[:k])


def words_from_segments(segments: Iterable) -> List[Word]:
    """Flatten faster-whisper segments (with .words) into Word list."""
    out: List[Word] = []
    for seg in segments:
        words = getattr(seg, "words", None) or []
        for w in words:
            text = (getattr(w, "word", None) or getattr(w, "text", "") or "").strip()
            if not text:
                continue
            start = float(getattr(w, "start", 0.0) or 0.0)
            end = float(getattr(w, "end", start) or start)
            out.append(Word(text=text, start=start, end=end))
    return out


# Flush triggers for sending a translation unit (committed source → target).
SENTENCE_ENDINGS = ("。", "！", "？", "…", ".", "!", "?")
CLAUSE_PARTICLES = ("て", "で", "けど", "から", "ので", "が", "を", "に")


def should_flush(
    committed_text: str,
    *,
    silence_ms: float = 0.0,
    silence_threshold_ms: float = 600.0,
    max_chars: int = 80,
    hard_timeout_s: float = 0.0,
    hard_timeout_threshold_s: float = 6.0,
    clause_pause_ms: float = 0.0,
) -> bool:
    """True when committed source text should be sent to translation."""
    text = (committed_text or "").strip()
    if not text:
        return False
    if text.endswith(SENTENCE_ENDINGS):
        return True
    if silence_ms >= silence_threshold_ms:
        return True
    # JA is denser; count chars without spaces.
    compact = text.replace(" ", "")
    if len(compact) >= max_chars:
        return True
    if hard_timeout_s >= hard_timeout_threshold_s:
        return True
    if clause_pause_ms >= 350.0 and any(text.endswith(p) for p in CLAUSE_PARTICLES):
        return True
    return False
