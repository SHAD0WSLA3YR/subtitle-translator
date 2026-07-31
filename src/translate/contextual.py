"""Contextual JA→EN (or any→any) translator chain.
Order (offline-first):
  1. Local CPU NLLB — raw source text only (never chat prompts)
  2. NVIDIA NIM (Riva) — optional, budgeted; circuit-breaks on 401/403
  3. Offline Argos NMT
  4. Passthrough source text (never invent English)
Never uses Whisper task=translate. Context stays here, not in ASR prompts.
NLLB is seq2seq: feed ONLY the utterance text + language codes.
"""
from __future__ import annotations
import logging
import os
import time
from typing import List, Optional, Tuple
import requests
from src.stt.events import source_hash
from src.translate.local_cpu import LocalCPUTranslator
from src.translate.nmt import NMTTranslator
from src.translate.rate_limit import RateBudget
logger = logging.getLogger(__name__)
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_NVIDIA_MODEL = "nvidia/riva-translate-4b-instruct-v2"
# Reject LLM-style prompt leakage from any backend.
_PROMPT_LEAK_MARKERS = (
    "previous pairs:",
    "translate to english:",
    "translate to japanese:",
    "you are a machine translation",
)
class ContextualTranslator:
    """Local NLLB → NVIDIA (optional) → Argos → passthrough."""
    def __init__(
        self,
        *,
        nmt: Optional[NMTTranslator] = None,
        local: Optional[LocalCPUTranslator] = None,
        local_provider: str = "auto",
        api_key_env: str = "NVIDIA_API_KEY",
        model: str = DEFAULT_NVIDIA_MODEL,
        rpm_limit: int = 40,
        weekly_limit: int = 1000,
        temperature: float = 0.0,
        max_tokens: int = 160,
        timeout: float = 12.0,
        context_pairs: int = 0,
        use_nvidia: bool = False,
    ):
        self._nmt = nmt or NMTTranslator()
        self.local_provider = (local_provider or "auto").lower()
        self._local = local
        if self._local is None and self.local_provider in ("auto", "nllb", "nllb-cpu"):
            self._local = LocalCPUTranslator()
        self.api_key_env = api_key_env
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.context_pairs = max(0, int(context_pairs))
        self.use_nvidia = use_nvidia
        self._api_key = self._load_key()
        self._budget = RateBudget(rpm_limit=rpm_limit, weekly_limit=weekly_limit)
        self._history: List[Tuple[str, str]] = []
        self._nvidia_hard_failed = False
    def _load_key(self) -> Optional[str]:
        # Do NOT fall back to LLM_API_KEY — wrong product key caused serial 401s.
        for env in (self.api_key_env, "NVIDIA_API_KEY", "NVAPI_KEY"):
            key = (os.environ.get(env) or "").strip()
            if key:
                if env != self.api_key_env:
                    logger.info("Using NVIDIA API key from %s", env)
                return key
        try:
            from pathlib import Path
            env_path = Path.cwd() / ".env"
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() in (self.api_key_env, "NVIDIA_API_KEY", "NVAPI_KEY"):
                        val = v.strip().strip('"').strip("'")
                        if val:
                            os.environ.setdefault(k.strip(), val)
                            return val
        except Exception:
            pass
        if self.use_nvidia:
            logger.info(
                "No %s / NVIDIA_API_KEY — NVIDIA tier disabled this session",
                self.api_key_env,
            )
        return None
    @property
    def nvidia_enabled(self) -> bool:
        return (
            bool(self._api_key)
            and self.use_nvidia
            and not self._nvidia_hard_failed
        )
    @property
    def local_enabled(self) -> bool:
        if self.local_provider == "argos":
            return False
        return self._local is not None and self._local.available
    def mt_chain_status(self) -> str:
        parts = []
        if self.local_enabled:
            parts.append("local-cpu(ON)")
        elif self.local_provider in ("auto", "nllb", "nllb-cpu"):
            parts.append("local-cpu(off)")
        if self._nvidia_hard_failed:
            parts.append("nvidia(dead)")
        elif self.nvidia_enabled:
            parts.append("nvidia")
        parts.extend(["argos", "pass"])
        return "+".join(parts)
    def budget_status(self) -> str:
        return self._budget.status()
    @staticmethod
    def _looks_like_prompt_leak(text: str) -> bool:
        low = (text or "").lower()
        return any(m in low for m in _PROMPT_LEAK_MARKERS)
    def warmup(self, text: str = "こんにちは") -> float:
        """Pay cold-start cost before live audio (primary MT backend)."""
        t0 = time.perf_counter()
        try:
            out, backend = self.translate(text, "ja", "en", utterance_id=0)
            # Do not keep warmup pairs in the contextual history.
            self._history.clear()
            elapsed = time.perf_counter() - t0
            if out and not self._looks_like_prompt_leak(out):
                logger.info(
                    "MT warmup complete via %s (%.1fs): %s",
                    backend,
                    elapsed,
                    (out or "")[:60],
                )
            else:
                logger.warning(
                    "MT warmup finished backend=%s (%.1fs) out=%r",
                    backend,
                    elapsed,
                    (out or "")[:80],
                )
            return elapsed
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            logger.warning("MT warmup failed (%.1fs): %s", elapsed, exc)
            return elapsed
    def remember(self, source: str, translated: str) -> None:
        source = (source or "").strip()
        translated = (translated or "").strip()
        if not source or not translated or translated == source:
            return
        if self._looks_like_prompt_leak(translated):
            return
        self._history.append((source, translated))
        if len(self._history) > 20:
            self._history = self._history[-20:]
    def translate(
        self,
        text: str,
        src: str = "ja",
        tgt: str = "en",
        *,
        utterance_id: int = 0,
        expected_source_hash: str = "",
    ) -> Tuple[str, str]:
        """Return (translated_text, backend_tag).
        Local NLLB always receives the raw ``text`` utterance — never a chat
        prompt. Context pairs are NVIDIA-only and currently unused (context_pairs=0).
        """
        text = (text or "").strip()
        if not text:
            return "", "empty"
        src = (src or "ja").lower()
        tgt = (tgt or "en").lower()
        if src == tgt:
            return text, "passthrough"
        current_hash = source_hash(text)
        if expected_source_hash and current_hash != expected_source_hash:
            logger.warning(
                "[id=%d | MT_STALE | expected=%s got=%s]",
                utterance_id,
                expected_source_hash,
                current_hash,
            )
            return "", "stale"
        # 1) Local CPU NLLB — raw JA/EN text only.
        if self._local is not None and self.local_provider != "argos":
            local = self._local.translate(text, src, tgt)
            if local and local.strip() and not self._looks_like_prompt_leak(local):
                self.remember(text, local)
                logger.info(
                    "[id=%d | MT local-cpu | hash=%s] EN: %s",
                    utterance_id,
                    current_hash,
                    local[:80],
                )
                return local.strip(), "local-cpu"
        # 2) NVIDIA Riva (optional).
        if self.nvidia_enabled and self._budget.try_consume():
            en = self._call_nvidia(text, src, tgt)
            if en and not self._looks_like_prompt_leak(en):
                self.remember(text, en)
                logger.info(
                    "[id=%d | MT nvidia | hash=%s] EN: %s",
                    utterance_id,
                    current_hash,
                    en[:80],
                )
                return en, "nvidia"
            if en and self._looks_like_prompt_leak(en):
                logger.warning(
                    "[id=%d | MT nvidia prompt-leak rejected] falling back",
                    utterance_id,
                )
        # 3) Argos
        if self._nmt is not None:
            try:
                local = self._nmt.translate(text, src, tgt)
                if local and local.strip() and not self._looks_like_prompt_leak(local):
                    self.remember(text, local)
                    logger.info(
                        "[id=%d | MT argos | hash=%s] EN: %s",
                        utterance_id,
                        source_hash(text),
                        local[:80],
                    )
                    return local.strip(), "argos"
            except Exception as exc:
                logger.debug("Argos failed: %s", exc)
        # 4) Passthrough
        logger.info(
            "[id=%d | MT passthrough | hash=%s] EN: %s",
            utterance_id,
            source_hash(text),
            text[:80],
        )
        return text, "passthrough"
    def _call_nvidia(self, text: str, src: str, tgt: str) -> Optional[str]:
        messages = self._build_messages(text, src, tgt)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        try:
            resp = requests.post(
                NVIDIA_URL, headers=headers, json=payload, timeout=self.timeout
            )
            if resp.status_code in (401, 403):
                self._nvidia_hard_failed = True
                logger.warning(
                    "NVIDIA API HTTP %s — disabling NVIDIA for this session "
                    "(fix NVIDIA_API_KEY or set latency.translator.use_nvidia: false)",
                    resp.status_code,
                )
                return None
            if resp.status_code >= 400:
                logger.warning(
                    "NVIDIA API HTTP %s — falling back",
                    resp.status_code,
                )
                return None
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return (content or "").strip() or None
        except Exception as exc:
            logger.warning("NVIDIA call failed — fallback: %s", exc)
            return None
    def _build_messages(self, text: str, src: str, tgt: str) -> list:
        """Minimal Riva Translate prompt — raw utterance only, no history blob."""
        lang_names = {
            "ja": "Japanese",
            "en": "English",
            "zh": "Chinese",
            "ko": "Korean",
        }
        tgt_name = lang_names.get(tgt, tgt)
        # Single user turn; Riva is an NMT model behind a chat endpoint.
        return [
            {
                "role": "user",
                "content": f"Translate to {tgt_name}:\n{text}",
            }
        ]
