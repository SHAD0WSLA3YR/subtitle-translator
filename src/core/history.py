"""Session history manager using SQLite.

Logs translation sessions with start/end times, source/target language,
and each subtitle's raw + refined text. Viewable from the history dialog.
"""

import sqlite3
import datetime
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HistoryManager:
    """Persists translation sessions to a local SQLite database.

    Usage:
        history = HistoryManager()
        history.start_session(source="ja", target="en")
        history.log_subtitle("Raw text", "Refined text")
        history.end_session()
        sessions = history.get_recent_sessions()
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = str(Path.home() / ".subtitle_translator" / "history.db")
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self._session_id: Optional[int] = None
        logger.info("History database: %s", db_path)

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                source_lang TEXT DEFAULT 'ja',
                target_lang TEXT DEFAULT 'en',
                subtitle_count INTEGER DEFAULT 0,
                audio_duration_sec REAL DEFAULT 0
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS subtitles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                raw_text TEXT,
                refined_text TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        self._conn.commit()

    def start_session(self, source: str = "ja", target: str = "en"):
        """Begin a new translation session."""
        if self._session_id is not None:
            self.end_session()
        self._conn.execute(
            "INSERT INTO sessions (start_time, source_lang, target_lang) VALUES (?, ?, ?)",
            (datetime.datetime.now().isoformat(), source, target),
        )
        self._session_id = self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self._conn.commit()
        logger.debug("Session started: id=%d", self._session_id)

    def log_subtitle(self, raw: str, refined: Optional[str] = None):
        """Log a subtitle entry for the current session."""
        if self._session_id is None:
            return
        self._conn.execute(
            "INSERT INTO subtitles (session_id, timestamp, raw_text, refined_text) VALUES (?, ?, ?, ?)",
            (self._session_id, datetime.datetime.now().isoformat(), raw, refined),
        )
        self._conn.execute(
            "UPDATE sessions SET subtitle_count = subtitle_count + 1 WHERE id = ?",
            (self._session_id,),
        )
        self._conn.commit()

    def end_session(self):
        """End the current session."""
        if self._session_id is None:
            return
        self._conn.execute(
            "UPDATE sessions SET end_time = ? WHERE id = ?",
            (datetime.datetime.now().isoformat(), self._session_id),
        )
        self._conn.commit()
        logger.debug("Session ended: id=%d", self._session_id)
        self._session_id = None

    def get_recent_sessions(self, limit: int = 20):
        """Return the most recent sessions."""
        return self._conn.execute(
            "SELECT id, start_time, end_time, source_lang, target_lang, subtitle_count, "
            "COALESCE(audio_duration_sec, 0) FROM sessions ORDER BY start_time DESC LIMIT ?",
            (limit,),
        ).fetchall()

    def get_session_subtitles(self, session_id: int):
        """Return all subtitles for a given session."""
        return self._conn.execute(
            "SELECT id, timestamp, raw_text, COALESCE(refined_text, raw_text) "
            "FROM subtitles WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

    def get_session_count(self) -> int:
        """Total number of sessions."""
        row = self._conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return row[0] if row else 0

    def clear_all(self):
        """Delete all history."""
        self._conn.execute("DELETE FROM subtitles")
        self._conn.execute("DELETE FROM sessions")
        self._conn.commit()

    def close(self):
        """Close the database connection."""
        self.end_session()
        self._conn.close()
