"""LLM-based translation refinement using OpenRouter or NVIDIA API.

Improves Whisper's raw translation output by fixing grammar, improving
fluency, and keeping output concise for subtitle display.
"""

import os
import time
import json
import logging
from typing import Optional, Callable
from functools import lru_cache

import requests

from .prompts import build_messages

logger = logging.getLogger(__name__)

# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Rate limit: minimum interval between API calls (seconds)
MIN_INTERVAL = 2.0


class LLMRefiner:
    """Improves Whisper translations using an LLM API.

    Supports OpenRouter and NVIDIA backends. Falls back to the original
    translation text on any API error or timeout.

    Usage:
        refiner = LLMRefiner(provider="openrouter", model="meta-llama/llama-3.1-8b-instruct")
        refiner.load_api_key()

        refined = refiner.refine("Raw translation text.")
        # Returns either the refined text, or the original if API fails.
    """

    def __init__(
        self,
        provider: str = "openrouter",
        model: str = "meta-llama/llama-3.1-8b-instruct",
        api_key_env: str = "LLM_API_KEY",
        temperature: float = 0.1,
        max_tokens: int = 256,
        timeout: float = 10.0,
        enabled: bool = True,
    ):
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.enabled = enabled
        self._api_key: Optional[str] = None
        self._last_call_time: float = 0.0

    def load_api_key(self) -> bool:
        """Read the API key from the environment variable.

        Returns:
            True if key was found, False otherwise.
        """
        key = os.environ.get(self.api_key_env)
        if key:
            self._api_key = key.strip()
            return True
        logger.warning("API key not found in env var %s", self.api_key_env)
        return False

    def refine(self, text: str) -> str:
        """Refine a raw translation using the LLM.

        Args:
            text: Raw Whisper translation output.

        Returns:
            Refined text, or the original text if refinement is disabled
            or the API call fails.
        """
        if not self.enabled or not text:
            return text

        # Rate limiting
        self._rate_limit()

        messages = build_messages(text)

        try:
            if self.provider == "openrouter":
                result = self._call_openrouter(messages)
            elif self.provider == "nvidia":
                result = self._call_nvidia(messages)
            else:
                logger.warning("Unknown LLM provider: %s", self.provider)
                return text

            if result:
                logger.debug("Refined: '%s' -> '%s'", text[:40], result[:40])
                return result

        except Exception as exc:
            logger.error("LLM refinement failed: %s", exc)

        return text

    def refine_with_callback(
        self,
        text: str,
        on_result: Callable[[str], None],
        on_fallback: Optional[Callable[[str], None]] = None,
    ):
        """Refine text and deliver result via callback.

        Falls back to the original text if the API call fails,
        calling on_fallback if provided.

        Args:
            text: Raw translation text.
            on_result: Called with the refined text on success.
            on_fallback: Called with the original text on failure.
        """
        result = self.refine(text)
        if result != text:
            on_result(result)
        elif on_fallback:
            on_fallback(text)

    def _rate_limit(self):
        """Enforce minimum interval between API calls."""
        elapsed = time.monotonic() - self._last_call_time
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        self._last_call_time = time.monotonic()

    def _call_openrouter(self, messages: list) -> Optional[str]:
        """Call the OpenRouter chat completions API."""
        if not self._api_key:
            logger.warning("OpenRouter API key not loaded")
            return None

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip() if content else None

    def _call_nvidia(self, messages: list) -> Optional[str]:
        """Call the NVIDIA NIM chat completions API."""
        if not self._api_key:
            logger.warning("NVIDIA API key not loaded")
            return None

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
        }

        resp = requests.post(
            NVIDIA_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip() if content else None
