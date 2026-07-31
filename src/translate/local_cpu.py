"""Pluggable local CPU translator (NLLB-600M CT2 when available).

GPU is reserved for Whisper ASR; this runs on CPU only. When the NLLB
model is not installed, ``available`` is False and ContextualTranslator
falls through to Argos.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# NLLB language codes for common pairs
_NLLB_LANG = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
}

_DEFAULT_MODEL_DIR = Path.home() / ".subtitle_translator" / "nllb-600m-ct2"
_HF_TOKENIZER_ID = "facebook/nllb-200-distilled-600M"


class LocalCPUTranslator:
    """Lazy NLLB-200-distilled-600M via CTranslate2 (CPU, int8)."""

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        on_status: Optional[callable] = None,
        tokenizer_id: str = _HF_TOKENIZER_ID,
    ):
        self._model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._device = device
        self._compute_type = compute_type
        self._on_status = on_status
        self._tokenizer_id = tokenizer_id
        self._lock = threading.Lock()
        self._translator = None
        self._tokenizer = None
        self._available: Optional[bool] = None
        self._load_error: Optional[str] = None

    def _status(self, message: str) -> None:
        if self._on_status and message:
            try:
                self._on_status(message)
            except Exception:
                pass

    @property
    def available(self) -> bool:
        if self._available is None:
            self._try_load()
        return bool(self._available)

    @property
    def backend_name(self) -> str:
        return "nllb-cpu" if self.available else "none"

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    def _try_load(self) -> bool:
        with self._lock:
            if self._available is not None:
                return self._available
            try:
                import ctranslate2 as ct2
            except ImportError:
                self._load_error = "ctranslate2 not installed"
                self._available = False
                logger.info("Local CPU NLLB unavailable: %s", self._load_error)
                return False

            if not self._model_dir.is_dir():
                self._load_error = f"model dir missing: {self._model_dir}"
                self._available = False
                logger.info(
                    "Local CPU NLLB unavailable (%s). "
                    "Run: python scripts/install_nllb_ct2.py",
                    self._load_error,
                )
                return False

            try:
                from transformers import AutoTokenizer
            except ImportError:
                self._load_error = "transformers not installed (needed for NLLB tokenizer)"
                self._available = False
                logger.info("Local CPU NLLB unavailable: %s", self._load_error)
                return False

            try:
                self._status("Loading local NLLB translator (CPU)...")
                self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
                self._translator = ct2.Translator(
                    str(self._model_dir),
                    device=self._device,
                    compute_type=self._compute_type,
                    inter_threads=max(1, min(4, (os.cpu_count() or 4) // 2)),
                )
                self._available = True
                self._status("")
                logger.info("Local CPU NLLB ready at %s", self._model_dir)
                return True
            except Exception as exc:
                self._load_error = str(exc)
                self._available = False
                self._translator = None
                self._tokenizer = None
                logger.warning("Local CPU NLLB load failed: %s", exc)
                return False

    def warmup(self, text: str = "こんにちは") -> bool:
        """Force-load model/tokenizer and run one dummy translate."""
        if not self._try_load():
            return False
        out = self.translate(text, "ja", "en")
        return bool(out)

    def translate(self, text: str, src: str, tgt: str) -> Optional[str]:
        text = (text or "").strip()
        if not text or not src or not tgt or src == tgt:
            return text or None
        if not self._try_load() or self._translator is None or self._tokenizer is None:
            return None

        src_tok = _NLLB_LANG.get(src.lower())
        tgt_tok = _NLLB_LANG.get(tgt.lower())
        if not src_tok or not tgt_tok:
            logger.debug("NLLB: unsupported pair %s→%s", src, tgt)
            return None

        # Guard: never feed chat/LLM prompt templates into NLLB.
        lowered = text.lower()
        if (
            "previous pairs:" in lowered
            or "translate to english:" in lowered
            or "translate to japanese:" in lowered
        ):
            logger.warning(
                "NLLB refused chat-style input (%.40r...) — caller bug",
                text[:40],
            )
            return None

        try:
            self._tokenizer.src_lang = src_tok
            source_tokens: List[str] = self._tokenizer.convert_ids_to_tokens(
                self._tokenizer.encode(text)
            )
            # Keep trailing </s>: CT2 config has add_source_eos=false, so the
            # caller must supply EOS. Stripping it causes token-loop garbage.
            if not source_tokens or source_tokens[-1] not in (
                self._tokenizer.eos_token,
                "</s>",
            ):
                source_tokens = list(source_tokens) + ["</s>"]

            results = self._translator.translate_batch(
                [source_tokens],
                target_prefix=[[tgt_tok]],
                max_batch_size=1,
                beam_size=2,
                max_decoding_length=96,
                no_repeat_ngram_size=3,
            )
            if not results or not results[0].hypotheses:
                return None
            hyp = list(results[0].hypotheses[0])
            # Drop target language code / specials before decode.
            if hyp and hyp[0] == tgt_tok:
                hyp = hyp[1:]
            if hyp and hyp[0] in ("</s>", self._tokenizer.eos_token, "<s>"):
                hyp = hyp[1:]
            if hyp and hyp[0] == tgt_tok:
                hyp = hyp[1:]
            ids = self._tokenizer.convert_tokens_to_ids(hyp)
            out = self._tokenizer.decode(ids, skip_special_tokens=True)
            return (out or "").strip() or None
        except Exception as exc:
            logger.debug("NLLB translate failed: %s", exc)
            return None
