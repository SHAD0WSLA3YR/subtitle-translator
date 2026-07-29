"""Session history viewer dialog.

Displays past translation sessions with their subtitles.
Users can browse sessions and export subtitle text.
"""

import logging
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QListWidget,
    QTextEdit, QPushButton, QSplitter, QMessageBox, QApplication,
    QWidget,
)

from src.core.history import HistoryManager

logger = logging.getLogger(__name__)


class HistoryDialog(QDialog):
    """Modal dialog showing past translation sessions and their content."""

    def __init__(self, history: HistoryManager, parent=None):
        super().__init__(parent)
        self._history = history
        self.setWindowTitle("Translation History")
        self.setMinimumSize(700, 450)
        self.resize(750, 500)

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # --- Left: session list ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_label = QLabel("Sessions")
        left_label.setStyleSheet("font-weight: bold;")
        left_layout.addWidget(left_label)
        self.session_list = QListWidget()
        self.session_list.currentRowChanged.connect(self._on_session_selected)
        left_layout.addWidget(self.session_list)
        splitter.addWidget(left_widget)

        # --- Right: subtitle text ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_label = QLabel("Subtitles")
        right_label.setStyleSheet("font-weight: bold;")
        right_layout.addWidget(right_label)
        self.subtitle_text = QTextEdit()
        self.subtitle_text.setReadOnly(True)
        right_layout.addWidget(self.subtitle_text)
        splitter.addWidget(right_widget)

        splitter.setSizes([280, 420])
        layout.addWidget(splitter)

        # --- Bottom buttons ---
        btn_layout = QHBoxLayout()
        self.copy_btn = QPushButton("Copy Subtitles")
        self.copy_btn.clicked.connect(self._copy_subtitles)
        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.clicked.connect(self._clear_history)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.copy_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        self._refresh_sessions()

    def _refresh_sessions(self):
        """Populate the session list from the history manager."""
        self.session_list.blockSignals(True)
        self.session_list.clear()
        sessions = self._history.get_recent_sessions()
        for s in sessions:
            sid, start, end, src, tgt, count, dur = s
            try:
                dt = datetime.fromisoformat(start)
                label = dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                label = start[:16]
            label += f"  [{src}\u2192{tgt}]  {count} subs"
            self.session_list.addItem(label)
        if not sessions:
            self.session_list.addItem("No sessions yet")
        self.session_list.blockSignals(False)
        self.subtitle_text.clear()

    def _on_session_selected(self, row: int):
        """Display subtitles for the selected session."""
        sessions = self._history.get_recent_sessions()
        if row < 0 or row >= len(sessions) or not sessions:
            return
        sid = sessions[row][0]
        subtitles = self._history.get_session_subtitles(sid)
        if not subtitles:
            self.subtitle_text.setPlainText("(no subtitles recorded)")
            return
        lines = [refined for _, _, _, refined in subtitles]
        self.subtitle_text.setPlainText("\n".join(lines))

    def _copy_subtitles(self):
        """Copy current subtitle text to clipboard."""
        text = self.subtitle_text.toPlainText()
        if text:
            QApplication.clipboard().setText(text)
            self.copy_btn.setText("Copied!")
            QTimer.singleShot(1500, lambda: self.copy_btn.setText("Copy Subtitles"))

    def _clear_history(self):
        """Clear all history after confirmation."""
        reply = QMessageBox.question(
            self, "Clear History",
            "Delete all session history? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._history.clear_all()
            self._refresh_sessions()
            self.subtitle_text.clear()
