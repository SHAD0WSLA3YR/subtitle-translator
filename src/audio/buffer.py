"""Thread-safe ring buffer for streaming audio chunks."""

import numpy as np
import threading
from typing import Optional


class AudioBuffer:
    """Thread-safe ring buffer for float32 mono audio data.
    
    Stores up to `max_seconds` of audio at `sample_rate` Hz.
    New data overwrites oldest data when full.
    """

    def __init__(self, sample_rate: int = 16000, max_seconds: float = 10.0):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_seconds)
        self._buffer = np.zeros(self.max_samples, dtype=np.float32)
        self._cursor = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """Append audio samples. Overwrites oldest if full."""
        n = len(data)
        if n == 0:
            return
        if n >= self.max_samples:
            # Data larger than buffer — keep most recent portion
            data = data[-self.max_samples:]
            n = self.max_samples

        with self._lock:
            start = self._cursor % self.max_samples
            end = start + n

            if end <= self.max_samples:
                self._buffer[start:end] = data
            else:
                # Wraparound
                first_part = self.max_samples - start
                self._buffer[start:] = data[:first_part]
                self._buffer[:end - self.max_samples] = data[first_part:]

            self._cursor += n

    def read_all(self) -> np.ndarray:
        """Return all currently buffered audio (oldest first)."""
        with self._lock:
            n = min(self._cursor, self.max_samples)
            start = (self._cursor - n) % self.max_samples
            if start + n <= self.max_samples:
                return self._buffer[start:start + n].copy()
            else:
                first_part = self.max_samples - start
                return np.concatenate([
                    self._buffer[start:],
                    self._buffer[:n - first_part]
                ])

    def read_last(self, n_samples: int) -> np.ndarray:
        """Return the most recent `n_samples` (or fewer if buffer has less)."""
        with self._lock:
            available = min(self._cursor, self.max_samples)
            n = min(n_samples, available)
            start = (self._cursor - n) % self.max_samples
            if start + n <= self.max_samples:
                return self._buffer[start:start + n].copy()
            else:
                first_part = self.max_samples - start
                return np.concatenate([
                    self._buffer[start:],
                    self._buffer[:n - first_part]
                ])

    def clear(self) -> None:
        """Reset buffer to empty state."""
        with self._lock:
            self._buffer.fill(0.0)
            self._cursor = 0

    def duration(self) -> float:
        """Return duration of audio currently in buffer (seconds)."""
        with self._lock:
            n = min(self._cursor, self.max_samples)
            return n / self.sample_rate

    @property
    def is_full(self) -> bool:
        """True if the buffer has been completely filled at least once."""
        return self._cursor >= self.max_samples
