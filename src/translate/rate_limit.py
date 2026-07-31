"""Persistent client-side counters for free-tier API budgets.

No health probes — just local bookkeeping. When a budget is exhausted,
callers fall back to offline NMT / passthrough.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Optional

logger = logging.getLogger(__name__)


def _default_state_path() -> Path:
    return Path.home() / ".subtitle_translator" / "nvidia_usage.json"


class RateBudget:
    """Sliding-window RPM + weekly request counters."""

    def __init__(
        self,
        *,
        rpm_limit: int = 40,
        weekly_limit: int = 1000,
        state_path: Optional[Path] = None,
        name: str = "nvidia",
    ):
        self.rpm_limit = max(1, int(rpm_limit))
        self.weekly_limit = max(1, int(weekly_limit))
        self.state_path = Path(state_path) if state_path else _default_state_path()
        self.name = name
        self._lock = threading.Lock()
        self._minute: Deque[float] = deque()
        self._week_count = 0
        self._week_id = self._current_week_id()
        self._load()

    @staticmethod
    def _current_week_id() -> str:
        # ISO week — resets Monday UTC.
        now = datetime.now(timezone.utc)
        return f"{now.isocalendar().year}-W{now.isocalendar().week:02d}"

    def _load(self) -> None:
        try:
            if not self.state_path.exists():
                return
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            week_id = data.get("week_id") or ""
            if week_id == self._week_id:
                self._week_count = int(data.get("week_count", 0) or 0)
            # Minute window is ephemeral — don't restore timestamps older than 60s.
            now = time.monotonic()
            for ts in data.get("minute_wall", []):
                # Can't restore wall→monotonic; start fresh each process for RPM.
                pass
            logger.info(
                "%s budget loaded: week %s count=%d / %d",
                self.name,
                self._week_id,
                self._week_count,
                self.weekly_limit,
            )
        except Exception as exc:
            logger.debug("Budget load skipped: %s", exc)

    def _save(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "week_id": self._week_id,
                "week_count": self._week_count,
                "rpm_limit": self.rpm_limit,
                "weekly_limit": self.weekly_limit,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.debug("Budget save failed: %s", exc)

    def _prune_minute(self, now: float) -> None:
        while self._minute and now - self._minute[0] >= 60.0:
            self._minute.popleft()

    def _roll_week(self) -> None:
        wid = self._current_week_id()
        if wid != self._week_id:
            self._week_id = wid
            self._week_count = 0

    def can_consume(self) -> bool:
        with self._lock:
            self._roll_week()
            now = time.monotonic()
            self._prune_minute(now)
            if self._week_count >= self.weekly_limit:
                return False
            if len(self._minute) >= self.rpm_limit:
                return False
            return True

    def remaining_rpm(self) -> int:
        with self._lock:
            now = time.monotonic()
            self._prune_minute(now)
            return max(0, self.rpm_limit - len(self._minute))

    def remaining_week(self) -> int:
        with self._lock:
            self._roll_week()
            return max(0, self.weekly_limit - self._week_count)

    def try_consume(self) -> bool:
        """Reserve one request slot. False → caller must use backup path."""
        with self._lock:
            self._roll_week()
            now = time.monotonic()
            self._prune_minute(now)
            if self._week_count >= self.weekly_limit:
                logger.info(
                    "%s weekly budget exhausted (%d/%d) — using local backup",
                    self.name,
                    self._week_count,
                    self.weekly_limit,
                )
                return False
            if len(self._minute) >= self.rpm_limit:
                logger.info(
                    "%s RPM budget exhausted (%d/%d) — using local backup",
                    self.name,
                    len(self._minute),
                    self.rpm_limit,
                )
                return False
            self._minute.append(now)
            self._week_count += 1
            self._save()
            return True

    def status(self) -> str:
        return (
            f"{self.name}: rpm {self.remaining_rpm()}/{self.rpm_limit}, "
            f"week {self.remaining_week()}/{self.weekly_limit}"
        )
