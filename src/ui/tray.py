"""System tray icon with right-click context menu.

Provides:
- App icon in the system tray
- Pause/Resume toggle
- Playback speed presets (0.75x–1.5x)
- Open Settings dialog
- Open History dialog
- Quick quit
"""

import logging
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QFont
from PyQt5.QtWidgets import (
    QSystemTrayIcon,
    QMenu,
    QAction,
    QActionGroup,
    QApplication,
)

from src.stt.whisper_stt import PLAYBACK_SPEED_PRESETS, clamp_playback_speed

logger = logging.getLogger(__name__)


def _create_default_icon() -> QIcon:
    """Load assets/icon.png if present, else paint a CC badge."""
    icon_path = Path(__file__).resolve().parents[2] / "assets" / "icon.png"
    if icon_path.exists():
        return QIcon(str(icon_path))

    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    painter.setBrush(QColor("#2563eb"))
    painter.setPen(Qt.NoPen)
    painter.drawRoundedRect(4, 4, 56, 56, 10, 10)

    painter.setPen(QColor("white"))
    font = QFont("Segoe UI", 22, QFont.Bold)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignCenter, "CC")
    painter.end()
    return QIcon(pixmap)


class TrayIcon(QSystemTrayIcon):
    """System tray icon controlling app lifecycle and visibility."""

    pause_toggled = pyqtSignal()
    show_settings = pyqtSignal()
    show_history = pyqtSignal()
    quit_app = pyqtSignal()
    playback_speed_changed = pyqtSignal(float)
    toggle_overlay = pyqtSignal()

    PAUSED_TOOLTIP = "Subtitle Translator \u23f8 Paused"
    RUNNING_TOOLTIP = "Subtitle Translator \u25b6 Running"

    def __init__(self, parent=None, playback_speed: float = 1.0):
        icon = _create_default_icon()
        super().__init__(icon, parent)
        self._paused = False
        self._playback_speed = clamp_playback_speed(playback_speed)
        self._speed_actions: dict[float, QAction] = {}
        self.setToolTip(self._tooltip_text())

        self._menu = QMenu()
        self._build_menu()
        self.setContextMenu(self._menu)
        self.activated.connect(self._on_activated)

    def _tooltip_text(self) -> str:
        base = self.PAUSED_TOOLTIP if self._paused else self.RUNNING_TOOLTIP
        return f"{base} · {self._playback_speed:g}x"

    def _build_menu(self):
        self._pause_action = QAction("\u23f8 Pause", self._menu)
        self._pause_action.triggered.connect(self._on_toggle)
        self._menu.addAction(self._pause_action)

        self._overlay_action = QAction("\U0001f5a5 Hide subtitles", self._menu)
        self._overlay_action.triggered.connect(self.toggle_overlay.emit)
        self._menu.addAction(self._overlay_action)

        self._menu.addSeparator()

        speed_menu = self._menu.addMenu("Playback speed")
        speed_group = QActionGroup(speed_menu)
        speed_group.setExclusive(True)
        for speed in PLAYBACK_SPEED_PRESETS:
            action = QAction(f"{speed:g}x", speed_menu)
            action.setCheckable(True)
            action.setData(speed)
            if abs(speed - self._playback_speed) < 0.001:
                action.setChecked(True)
            action.triggered.connect(self._on_speed_action)
            speed_group.addAction(action)
            speed_menu.addAction(action)
            self._speed_actions[speed] = action

        self._menu.addSeparator()

        settings_action = QAction("\u2699 Settings", self._menu)
        settings_action.triggered.connect(self.show_settings.emit)
        self._menu.addAction(settings_action)

        history_action = QAction("\U0001f4cb History", self._menu)
        history_action.triggered.connect(self.show_history.emit)
        self._menu.addAction(history_action)

        self._menu.addSeparator()

        quit_action = QAction("\u2716 Quit", self._menu)
        quit_action.triggered.connect(self.quit_app.emit)
        self._menu.addAction(quit_action)

    def _on_speed_action(self):
        action = self.sender()
        if not isinstance(action, QAction):
            return
        speed = clamp_playback_speed(float(action.data()))
        self.set_playback_speed(speed, emit=True)

    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.DoubleClick:
            self._on_toggle()

    def _on_toggle(self):
        self._paused = not self._paused
        if self._paused:
            self._pause_action.setText("\u25b6 Resume")
        else:
            self._pause_action.setText("\u23f8 Pause")
        self.setToolTip(self._tooltip_text())
        self.pause_toggled.emit()

    def set_paused(self, paused: bool):
        if paused != self._paused:
            self._paused = paused
            if self._paused:
                self._pause_action.setText("\u25b6 Resume")
            else:
                self._pause_action.setText("\u23f8 Pause")
            self.setToolTip(self._tooltip_text())

    def set_overlay_visible(self, visible: bool):
        """Keep the tray menu label in sync with overlay visibility."""
        self._overlay_action.setText(
            "\U0001f5a5 Hide subtitles" if visible else "\U0001f5a5 Show subtitles"
        )

    def set_playback_speed(self, speed: float, emit: bool = False):
        speed = clamp_playback_speed(speed)
        self._playback_speed = speed
        nearest = min(PLAYBACK_SPEED_PRESETS, key=lambda s: abs(s - speed))
        for preset, action in self._speed_actions.items():
            action.setChecked(abs(preset - nearest) < 0.001)
        self.setToolTip(self._tooltip_text())
        if emit:
            self.playback_speed_changed.emit(speed)

    def show_balloon(self, title: str, message: str):
        if QSystemTrayIcon.supportsMessages():
            self.showMessage(title, message, QSystemTrayIcon.Information, 3000)
