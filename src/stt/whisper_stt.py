"""faster-whisper model wrapper for Japanese ASR + English translation.

Two passes are available per clause:
  1) task=transcribe → what was heard (source language)
  2) task=translate  → English subtitle text

The transcribe pass roughly doubles GPU/CPU cost, so callers run it only
when they need the source text (comparison logging or clause-merge checks).

Supports playback_speed compensation when the video is played faster
than 1.0x (loopback audio is sped up; we stretch it back before Whisper).
"""

import logging
import threading
from typing import Optional, Tuple

import numpy as np

from src.stt.local_agreement import join_words, words_from_segments

logger = logging.getLogger(__name__)

# Supported YouTube/player speed range (what we compensate for)
PLAYBACK_SPEED_MIN = 0.75
PLAYBACK_SPEED_MAX = 1.5
PLAYBACK_SPEED_PRESETS = (0.75, 1.0, 1.25, 1.5)

# Biases Japanese ASR toward words that Whisper-small often mishears
# (土俵→投票, フェンス→テンス, etc.)
_JA_PROMPT = (
    "これは日本語の音声です。"
    "公民館、土俵、相撲、フェンス、横断歩道、駐車場、病院、公園、神社。"
)

# Language-specific initial prompts for the 9 major supported languages.
# These bias Whisper toward common words in each language.
_LANGUAGE_PROMPTS = {
    "ja": _JA_PROMPT,
    "zh": "这是中文语音。包含常见词汇：你好、谢谢、但是、因为、所以、问题、电脑、老师、朋友。",
    "ko": "이것은 한국어 음성입니다. 안녕하세요, 감사합니다, 하지만, 왜냐하면, 그리고,입니다,습니다.",
    "es": "Este es audio en español. Palabras comunes: hola, gracias, porque, pero, entonces, también, bueno, casa.",
    "fr": "Ceci est un audio en français. Mots courants : bonjour, merci, parce que, mais, alors, très, bien, maison.",
    "de": "Dies ist eine Audioaufnahme auf Deutsch. Häufige Wörter: hallo, danke, weil, aber, also, schon, doch, Mensch.",
    "pt": "Este é um áudio em português. Palavras comuns: olá, obrigado, porque, mas, então, também, bom, casa.",
    "ru": "Это аудио на русском языке. Распространённые слова: здравствуйте, спасибо, потому что, но, также, хорошо.",
    "it": "Questo è un audio in italiano. Parole comuni: ciao, grazie, perché, ma, allora, molto, bene, casa.",
}


def japanese_ratio(text: str) -> float:
    """Fraction of non-space characters that are kana or kanji."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    jp = sum(
        1
        for c in chars
        if "\u3040" <= c <= "\u30ff" or "\u4e00" <= c <= "\u9fff"
    )
    return jp / len(chars)


def is_untranslated(text: str, threshold: float = 0.3) -> bool:
    """True when a 'translation' is really just the Japanese source.

    A quoted Japanese term inside an English sentence is fine, so this
    checks the proportion rather than the presence of kana/kanji.
    """
    return japanese_ratio(text) >= threshold


def clamp_playback_speed(speed: float) -> float:
    """Clamp to the supported 0.75x–1.5x watching range."""
    try:
        value = float(speed)
    except (TypeError, ValueError):
        return 1.0
    if value <= 0:
        return 1.0
    return max(PLAYBACK_SPEED_MIN, min(PLAYBACK_SPEED_MAX, value))


def stretch_audio(audio: np.ndarray, playback_speed: float) -> np.ndarray:
    """Undo playback speedup/slowdown so Whisper hears ~1.0x speech.

    If the user plays YouTube at 1.25x, loopback captures sped-up audio.
    Stretching by playback_speed restores approximate normal speaking rate.
    At 0.75x, audio is shortened (sped up) for Whisper.
    Supported range: 0.75–1.5.
    """
    speed = clamp_playback_speed(playback_speed)
    if abs(speed - 1.0) < 0.01:
        return audio
    if len(audio) < 2:
        return audio
    new_len = max(2, int(round(len(audio) * speed)))
    x_old = np.linspace(0.0, 1.0, len(audio), dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, new_len, dtype=np.float64)
    return np.interp(x_new, x_old, audio.astype(np.float64)).astype(np.float32)


class WhisperSTT:
    """Lazy-initialized faster-whisper wrapper.

    Usage:
        stt = WhisperSTT(model_size="small", device="cuda", playback_speed=1.25)
        heard, translated = stt.process(audio_array)
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cuda",
        compute_type: str = "int8_float16",
        language: str = "ja",
        beam_size: int = 3,
        playback_speed: float = 1.0,
        cpu_threads: int = 4,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.playback_speed = clamp_playback_speed(playback_speed or 1.0)
        self.cpu_threads = max(1, int(cpu_threads))
        self._model = None
        self._load_error: Optional[str] = None
        self._prev_ja = ""
        self._prev_en = ""
        self._detected_language: str = ""
        self._language_probability: float = 0.0
        # Sticky language for auto mode — set by the processor once detection
        # is confident, so short clauses don't flip en↔ja and trash quality.
        self._locked_language: str = ""
        # Serialize inference: streaming ASR + async final-pass share one model.
        self._infer_lock = threading.RLock()

    def set_playback_speed(self, speed: float) -> float:
        """Update playback speed at runtime (clamped to 0.75–1.5)."""
        self.playback_speed = clamp_playback_speed(speed)
        logger.info("Playback speed set to %.2fx", self.playback_speed)
        return self.playback_speed

    def load(self) -> None:
        """Load the model explicitly. Safe to call multiple times."""
        if self._model is not None or self._load_error is not None:
            return
        self._do_load()

    def _do_load(self):
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel  # type: ignore
            logger.info(
                "Loading WhisperModel(%s, device=%s, compute_type=%s, speed=%.2fx)",
                self.model_size, self.device, self.compute_type, self.playback_speed,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=1,
                cpu_threads=self.cpu_threads,
            )
            logger.info("WhisperModel loaded successfully")
        except Exception as exc:
            msg = str(exc)
            logger.warning("Failed to load model on %s: %s", self.device, msg)

            if self.device == "cuda":
                logger.info("Falling back to CPU with int8 compute type")
                try:
                    from faster_whisper import WhisperModel  # type: ignore
                    self._model = WhisperModel(
                        self.model_size,
                        device="cpu",
                        compute_type="int8",
                        num_workers=1,
                        cpu_threads=self.cpu_threads,
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    logger.info("WhisperModel loaded on CPU (fallback)")
                    return
                except Exception as exc2:
                    self._load_error = (
                        f"GPU failed: {msg}, CPU fallback failed: {exc2}"
                    )
                    logger.error(self._load_error)
                    raise RuntimeError(self._load_error) from exc2
            else:
                self._load_error = msg
                raise

    def _run(
        self,
        audio: np.ndarray,
        task: str,
        initial_prompt: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        """Run a single Whisper pass. task is 'transcribe' or 'translate'.

        When language is None and self.language is "auto", passes language=None
        to the model which triggers auto-detection — unless a locked language
        is set via ``locked_language``.
        Returns:
            Tuple of (text, detected_language_code, language_probability)
        """
        assert self._model is not None
        if language is not None:
            resolved_lang = language
        elif self._locked_language:
            resolved_lang = self._locked_language
        elif self.language == "auto":
            resolved_lang = None
        else:
            resolved_lang = self.language
        kwargs = dict(
            language=resolved_lang,
            task=task,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=True,
            temperature=0.0,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            without_timestamps=True,
        )
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segments, _info = self._model.transcribe(audio, **kwargs)
        text = " ".join(seg.text for seg in segments).strip()
        detected = (
            _info.language if hasattr(_info, "language") and _info.language
            else (resolved_lang or self.language or "??")
        )
        prob = float(getattr(_info, "language_probability", 0.0) or 0.0)
        if resolved_lang and not hasattr(_info, "language"):
            prob = 1.0
        return text, detected, prob

    def transcribe_words(
        self,
        audio: np.ndarray,
        lang_hint: str = "",
        beam_size: Optional[int] = None,
        initial_prompt: Optional[str] = None,
    ):
        """Transcribe with word_timestamps for LocalAgreement streaming.

        Returns:
            (words, full_text, detected_lang, probability)
        """
        with self._infer_lock:
            return self._transcribe_words_locked(
                audio, lang_hint=lang_hint, beam_size=beam_size, initial_prompt=initial_prompt
            )

    def _transcribe_words_locked(
        self,
        audio: np.ndarray,
        lang_hint: str = "",
        beam_size: Optional[int] = None,
        initial_prompt: Optional[str] = None,
    ):
        if len(audio) == 0:
            return [], "", self._detected_language or self.language, 0.0
        self.load()
        assert self._model is not None
        lang = lang_hint or self._locked_language or (
            None if self.language == "auto" else self.language
        )
        original_beam = self.beam_size
        if beam_size is not None:
            self.beam_size = beam_size
        try:
            audio = stretch_audio(audio, self.playback_speed)
            kwargs = dict(
                language=lang,
                task="transcribe",
                beam_size=self.beam_size,
                vad_filter=False,
                condition_on_previous_text=True,
                temperature=0.0,
                word_timestamps=True,
                without_timestamps=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
            )
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt
            segments, _info = self._model.transcribe(audio, **kwargs)
            # Materialize generator before reading info-dependent fields.
            seg_list = list(segments)
            words = words_from_segments(seg_list)
            text = join_words(words) or " ".join(
                (getattr(s, "text", "") or "").strip() for s in seg_list
            ).strip()
            detected = (
                _info.language if hasattr(_info, "language") and _info.language
                else (lang or self.language or "??")
            )
            prob = float(getattr(_info, "language_probability", 0.0) or 0.0)
            self._detected_language = detected
            self._language_probability = prob
            return words, text, detected, prob
        except Exception as exc:
            logger.error("Word-timestamp transcription error: %s", exc)
            return [], "", self._detected_language or self.language, 0.0
        finally:
            self.beam_size = original_beam

    def transcribe_source(self, audio: np.ndarray, lang_hint: str = "") -> str:
        """Single pass: what was heard, in the source language.

        Args:
            audio: Audio array to transcribe.
            lang_hint: Language code hint for prompt selection (e.g. from
                       previous detection). Falls back to self.language.
        """
        with self._infer_lock:
            if len(audio) == 0:
                return ""
            self.load()
            lang_key = lang_hint or self.language
            base_prompt = _LANGUAGE_PROMPTS.get(lang_key, "")
            prompt = base_prompt
            if self._prev_ja:
                prompt = f"{base_prompt} {self._prev_ja}"[-800:] if base_prompt else self._prev_ja[-800:]
            try:
                audio = stretch_audio(audio, self.playback_speed)
                # Prefer an explicit hint, then the sticky lock, else auto-detect.
                lang = lang_hint or self._locked_language or None
                result, detected, prob = self._run(
                    audio, task="transcribe", initial_prompt=prompt, language=lang
                )
                self._detected_language = detected
                self._language_probability = prob
                return result
            except Exception as exc:
                logger.error("Transcription error: %s", exc)
                return ""

    def translate_to_english(
        self,
        audio: np.ndarray,
        heard: str = "",
        beam_size: Optional[int] = None,
    ) -> Tuple[str, str]:
        """Single pass: English subtitle text.

        `heard` (if available) is fed in as vocabulary bias alongside the
        previous English line, which reduces hallucinated names and numbers.

        Returns:
            Tuple of (translated_text, detected_language_code)
        """
        with self._infer_lock:
            if len(audio) == 0:
                return "", self._detected_language or self.language
            self.load()
            original_beam = self.beam_size
            if beam_size is not None:
                self.beam_size = max(1, int(beam_size))
            parts = []
            if self._prev_en:
                parts.append(self._prev_en)
            if heard:
                lang_label = self._detected_language.upper() if self._detected_language else "SOURCE"
                parts.append(f"({lang_label}: {heard})")
            prompt = " ".join(parts)[-800:] or None
            try:
                audio = stretch_audio(audio, self.playback_speed)
                lang = self._locked_language or None
                result, detected, prob = self._run(
                    audio, task="translate", initial_prompt=prompt, language=lang
                )
                self._detected_language = detected
                self._language_probability = prob
                if prompt and detected == "ja" and is_untranslated(result):
                    # Whisper sometimes echoes the source instead of translating.
                    # The prompt is the usual trigger, so retry without context.
                    retry, retry_lang, _ = self._run(
                        audio, task="translate", initial_prompt=None, language=lang
                    )
                    if retry and not is_untranslated(retry):
                        logger.debug("Recovered untranslated line by dropping prompt")
                        return retry, retry_lang or detected
                return result, detected
            except Exception as exc:
                logger.error("Translation error: %s", exc)
                return "", self._detected_language or self.language
            finally:
                self.beam_size = original_beam

    def lock_language(self, code: str) -> None:
        """Pin auto-detect to a language so short clauses stay consistent."""
        code = (code or "").lower()
        if code in ("", "auto", "??"):
            return
        if code != self._locked_language:
            logger.info("Source language locked to %s", code)
        self._locked_language = code

    def unlock_language(self) -> None:
        if self._locked_language:
            logger.info("Source language unlocked (was %s)", self._locked_language)
        self._locked_language = ""

    @property
    def locked_language(self) -> str:
        return self._locked_language

    @property
    def language_probability(self) -> float:
        return getattr(self, "_language_probability", 0.0)

    def commit_context(self, heard: str, translated: str) -> None:
        """Remember an emitted line so the next clause gets it as context."""
        if heard:
            self._prev_ja = heard
        # Never feed Japanese back in as the "previous English" line — that
        # makes the next clause come back untranslated too, and it cascades.
        if translated and not is_untranslated(translated):
            self._prev_en = translated

    def process(self, audio: np.ndarray) -> Tuple[str, str, str]:
        """Both passes: return (heard_source, translated_english, detected_lang)."""
        heard = self.transcribe_source(audio)
        translated, detected = self.translate_to_english(audio, heard)
        self.commit_context(heard, translated)
        return heard, translated, detected

    def transcribe(self, audio: np.ndarray) -> str:
        """Backward-compatible: return English translation only."""
        translated, _ = self.translate_to_english(audio)
        self.commit_context("", translated)
        return translated

    @property
    def detected_language(self) -> str:
        return self._detected_language

    @detected_language.setter
    def detected_language(self, value: str):
        self._detected_language = value

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._load_error is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error
