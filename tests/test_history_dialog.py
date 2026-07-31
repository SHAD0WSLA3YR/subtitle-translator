"""Regression tests for the history dialog.

The dialog must never raise from a Qt slot (PyQt5 aborts the whole app on
unhandled slot exceptions), even with malformed or legacy database rows.

Run:  python -m unittest tests.test_history_dialog -v
"""

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.core.history import HistoryManager
from src.ui.history_dialog import HistoryDialog

_app = None


def setUpModule():
    global _app
    _app = QApplication.instance() or QApplication([])


class HistoryDialogTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self._tmp.name) / "history.db")
        self.history = HistoryManager(db_path=self.db_path)

    def tearDown(self):
        self.history.close()
        self._tmp.cleanup()

    def test_loads_sessions_and_subtitles(self):
        self.history.start_session(source="auto", target="en")
        self.history.log_subtitle("こんにちは", "Hello", detected_lang="ja")
        self.history.log_subtitle("さようなら", "Goodbye", detected_lang="ja")
        self.history.end_session()

        dlg = HistoryDialog(self.history)
        self.assertEqual(dlg.session_list.count(), 1)
        self.assertIn("Auto\u2192en", dlg.session_list.item(0).text())

        dlg.session_list.setCurrentRow(0)
        text = dlg.subtitle_text.toPlainText()
        self.assertIn("Hello", text)
        self.assertIn("Goodbye", text)
        dlg.close()

    def test_empty_database_shows_placeholder(self):
        dlg = HistoryDialog(self.history)
        self.assertEqual(dlg.session_list.count(), 1)
        self.assertEqual(dlg.session_list.item(0).text(), "No sessions yet")
        # Selecting the placeholder must be a no-op, not a crash.
        dlg.session_list.setCurrentRow(0)
        dlg.close()

    def test_malformed_rows_do_not_crash(self):
        self.history.start_session(source="auto", target="en")
        self.history.log_subtitle("ok", "ok")
        self.history.end_session()

        original = self.history.get_recent_sessions

        def bad_rows(limit=20):
            rows = list(original(limit))
            rows.append((99,))          # too short
            rows.append(None)           # not a row at all
            return rows

        self.history.get_recent_sessions = bad_rows
        dlg = HistoryDialog(self.history)
        # Only the valid session survives; malformed rows are skipped.
        self.assertEqual(dlg.session_list.count(), 1)
        dlg.session_list.setCurrentRow(0)
        self.assertIn("ok", dlg.subtitle_text.toPlainText())
        dlg.close()

    def test_selection_uses_stored_session_id(self):
        """Rows must resolve by stored id, not by re-querying positions."""
        self.history.start_session(source="ja", target="en")
        self.history.log_subtitle("first-session-line", "first-session-line")
        self.history.end_session()

        dlg = HistoryDialog(self.history)
        # A new session appearing after the dialog was opened must not
        # shift which session a click resolves to.
        self.history.start_session(source="ja", target="en")
        self.history.log_subtitle("newer-line", "newer-line")

        dlg.session_list.setCurrentRow(0)
        self.assertIn("first-session-line", dlg.subtitle_text.toPlainText())
        dlg.close()


if __name__ == "__main__":
    unittest.main()
