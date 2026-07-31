"""Offline NMT translation via Argos Translate (CTranslate2 under the hood).

Whisper's built-in `translate` task only outputs English. For any other
target language the pipeline transcribes the source text with Whisper and
hands it to this module, which translates text→text fully offline.

Language packages (~100 MB per direction) are downloaded from the Argos
index on first use and cached under ~/.local/share/argos-translate.
Pairs without a direct package are pivoted through English automatically.
"""

import logging
import threading
from typing import Optional

# NOTE: argostranslate is imported inside methods on purpose — it is an
# optional dependency with a slow import (spacy/stanza stack), and the app
# must start and run English targets even when it is not installed.

logger = logging.getLogger(__name__)


class NMTTranslator:
    """Lazy Argos Translate wrapper with on-demand package installs.

    Thread-safe: `translate()` may be called from the STT worker thread while
    `ensure_pair()` runs from a background prefetch thread.
    """

    def __init__(self, on_status: Optional[callable] = None):
        self._lock = threading.Lock()
        self._ready_pairs: set = set()
        self._failed_pairs: set = set()
        self._on_status = on_status
        self._available: Optional[bool] = None

    @property
    def available(self) -> bool:
        """True if argostranslate is importable."""
        if self._available is None:
            try:
                import argostranslate.translate  # noqa: F401
                self._available = True
            except ImportError:
                logger.warning(
                    "argostranslate not installed — non-English targets unavailable"
                )
                self._available = False
        return self._available

    def _status(self, message: str) -> None:
        if self._on_status:
            try:
                self._on_status(message)
            except Exception:
                pass

    def _install_package(self, from_code: str, to_code: str) -> bool:
        """Download and install one Argos package (from→to). Blocking."""
        import argostranslate.package as pkg

        installed = {
            (p.from_code, p.to_code) for p in pkg.get_installed_packages()
        }
        if (from_code, to_code) in installed:
            return True

        available = pkg.get_available_packages()
        match = next(
            (p for p in available
             if p.from_code == from_code and p.to_code == to_code),
            None,
        )
        if match is None:
            logger.warning("No Argos package for %s→%s", from_code, to_code)
            return False

        logger.info("Downloading translation model %s→%s...", from_code, to_code)
        self._status(f"Downloading translation model ({from_code}→{to_code})...")
        pkg.install_from_path(match.download())
        logger.info("Installed translation model %s→%s", from_code, to_code)
        return True

    def ensure_pair(self, src: str, tgt: str) -> bool:
        """Make sure src→tgt is translatable (direct or pivoted via English).

        Downloads missing packages, which can take minutes on first use.
        Returns True when the pair is usable.
        """
        if not src or not tgt or src == tgt:
            return True
        if not self.available:
            return False

        key = (src, tgt)
        with self._lock:
            if key in self._ready_pairs:
                return True
            if key in self._failed_pairs:
                return False

            try:
                import argostranslate.package as pkg
                pkg.update_package_index()
            except Exception as exc:
                logger.warning("Could not refresh Argos package index: %s", exc)

            try:
                ok = True
                # Argos pivots through English; install both legs as needed.
                if src != "en":
                    ok = self._install_package(src, "en") and ok
                if tgt != "en":
                    ok = self._install_package("en", tgt) and ok
            except Exception as exc:
                logger.error("Translation model install failed (%s→%s): %s", src, tgt, exc)
                ok = False

            if ok:
                self._ready_pairs.add(key)
                self._status("")
            else:
                self._failed_pairs.add(key)
                self._status(f"Translation {src}→{tgt} unavailable — showing English")
            return ok

    def retry_pair(self, src: str, tgt: str) -> None:
        """Forget a failed pair so the next ensure_pair tries again."""
        with self._lock:
            self._failed_pairs.discard((src, tgt))

    def is_ready(self, src: str, tgt: str) -> bool:
        """True if src→tgt can translate right now, without any download."""
        if src == tgt:
            return True
        with self._lock:
            return (src, tgt) in self._ready_pairs

    def translate(self, text: str, src: str, tgt: str) -> Optional[str]:
        """Translate text from src to tgt. Returns None on failure.

        Blocking; call from a worker thread. Downloads models on first use.
        """
        if not text or not src or not tgt:
            return None
        if src == tgt:
            return text
        if not self.ensure_pair(src, tgt):
            return None
        try:
            import argostranslate.translate as tr
            result = tr.translate(text, src, tgt)
            return result.strip() if result else None
        except Exception as exc:
            logger.error("NMT translation failed (%s→%s): %s", src, tgt, exc)
            return None
