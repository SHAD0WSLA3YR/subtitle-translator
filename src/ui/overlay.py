"""Transparent always-on-top subtitle overlay using PyQt5.

A draggable, resizable subtitle box with minimize/close controls and
configurable text/background colors. Starts bottom-center.
"""

import logging
from typing import Optional

from PyQt5.QtCore import (
    Qt, QTimer, QPropertyAnimation, QRect, QPoint, QEasingCurve, pyqtSignal
)
from PyQt5.QtGui import QFont, QColor, QPainter, QPen, QBrush, QMouseEvent
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QApplication, QSizePolicy,
)

from src.ui.win_blur import disable_blur, enable_blur

logger = logging.getLogger(__name__)

# Selectable subtitle text colors
FONT_COLORS = {
    "White": (255, 255, 255),
    "Yellow": (255, 212, 0),
    "Black": (0, 0, 0),
    "Red": (255, 68, 68),
    "Grey": (190, 190, 190),
    "Green": (91, 227, 125),
    "Cyan": (91, 214, 227),
}

# Selectable background styles. "Blur" uses the Windows acrylic backdrop.
BG_COLORS = {
    "Black": (0, 0, 0),
    "Grey": (64, 64, 64),
    "White": (255, 255, 255),
    "Blur": (20, 20, 20),
    "None": (0, 0, 0),
}

DEFAULTS = {
    "font_color": "White",
    "bg_color": "Black",
    "font_size": 26,
    "font_opacity": 255,
    "opacity": 200,
    "auto_hide_delay": 6.0,
    "max_lines": 3,
    "height": 180,
}

MIN_WIDTH = 320
MIN_HEIGHT = 90
GRIP_SIZE = 18
TOPBAR_HEIGHT = 20


class SubtitleOverlay(QWidget):
    """Frameless, always-on-top subtitle box.

    Painted background (stylesheet backgrounds do not render on translucent
    windows), draggable body, bottom-right resize grip, and minimize/close
    buttons in the top bar.
    """

    subtitle_shown = pyqtSignal(str)
    position_changed = pyqtSignal(int, int)     # x, y
    size_changed = pyqtSignal(int, int)         # width, height
    hidden_by_user = pyqtSignal()

    def __init__(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        width: Optional[int] = None,
        height: int = 180,
        font_size: int = 26,
        opacity: int = 200,
        auto_hide_delay: float = 6.0,
        max_lines: int = 3,
        font_color: str = "White",
        bg_color: str = "Black",
        font_opacity: int = 255,
    ):
        super().__init__()

        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setMouseTracking(True)

        self._auto_hide_delay = auto_hide_delay
        self._max_lines = max_lines
        self._bg_opacity = _clamp_byte(opacity)
        self._font_opacity = _clamp_byte(font_opacity)
        self._font_size = font_size
        self._font_color = font_color if font_color in FONT_COLORS else "White"
        self._bg_color = bg_color if bg_color in BG_COLORS else "Black"

        self._drag_offset: Optional[QPoint] = None
        self._resize_origin: Optional[QPoint] = None
        self._resize_start_size: Optional[tuple] = None
        self._current_texts: list[str] = []
        self._minimized = False
        self._expanded_height = max(MIN_HEIGHT, height)
        self._user_hidden = False
        self._blur_active = False

        screen = (
            QApplication.primaryScreen().availableGeometry()
            if QApplication.instance()
            else QRect(0, 0, 1920, 1080)
        )
        if width is None or width <= 0:
            width = max(640, int(screen.width() * 0.9))
        width = max(MIN_WIDTH, width)

        self._default_x = x if x is not None else (screen.width() - width) // 2
        self._default_y = y if y is not None else int(screen.height() * 0.78)
        self._box_height = max(MIN_HEIGHT, height)
        self.setGeometry(self._default_x, self._default_y, width, self._box_height)
        # Fixed size stops Windows from growing the translucent window when the
        # wrapped label wants more room (the setGeometry warning).
        self.setFixedSize(width, self._box_height)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 8, 16, 14)
        outer.setSpacing(2)

        # --- Top bar: drag hint + window controls ---
        top_bar = QHBoxLayout()
        top_bar.setContentsMargins(0, 0, 0, 0)
        top_bar.setSpacing(4)

        self._hint = QLabel("⠿ drag", self)
        self._hint.setFixedHeight(TOPBAR_HEIGHT)
        self._hint.setStyleSheet(
            "color: rgba(255,255,255,120); font-size: 11px; background: transparent;"
        )
        top_bar.addWidget(self._hint)

        self._lang_label = QLabel("", self)
        self._lang_label.setFixedHeight(TOPBAR_HEIGHT)
        self._lang_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 180);
                font-size: 11px;
                padding: 0 6px;
                background: rgba(0, 0, 0, 100);
                border-radius: 3px;
            }
        """)
        self._lang_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._lang_label.hide()
        top_bar.addWidget(self._lang_label)

        top_bar.addStretch(1)

        self._min_button = self._make_button("–", "Minimize to one line")
        self._min_button.clicked.connect(self.toggle_minimized)
        top_bar.addWidget(self._min_button)

        self._close_button = self._make_button("✕", "Hide overlay (restore from tray)")
        self._close_button.clicked.connect(self.hide_by_user)
        top_bar.addWidget(self._close_button)

        outer.addLayout(top_bar)

        # --- Subtitle text ---
        self._text = QLabel(self)
        self._text.setWordWrap(True)
        self._text.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self._text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self._text.setTextInteractionFlags(Qt.NoTextInteraction)
        self._text.setAttribute(Qt.WA_TransparentForMouseEvents)
        outer.addWidget(self._text, stretch=1)

        self._apply_text_style()

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._on_auto_hide)

        self._lang_badge_timer = QTimer(self)
        self._lang_badge_timer.setSingleShot(True)
        self._lang_badge_timer.timeout.connect(self._lang_label.hide)

        self._fade = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade.setDuration(220)
        self._fade.setEasingCurve(QEasingCurve.OutCubic)

    # --- Construction helpers ---

    def _make_button(self, glyph: str, tooltip: str) -> QPushButton:
        button = QPushButton(glyph, self)
        button.setToolTip(tooltip)
        button.setFixedSize(TOPBAR_HEIGHT, TOPBAR_HEIGHT)
        button.setCursor(Qt.ArrowCursor)
        button.setFocusPolicy(Qt.NoFocus)
        button.setStyleSheet("""
            QPushButton {
                color: rgba(255,255,255,170);
                background: rgba(255,255,255,25);
                border: none;
                border-radius: 9px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: rgba(255,255,255,70);
                color: #FFFFFF;
            }
        """)
        return button

    def set_detected_language(self, lang_code: str) -> None:
        """Show detected language badge in the top bar."""
        if not lang_code:
            self._lang_label.hide()
            return
        display_name = self._lang_code_to_name(lang_code)
        self._lang_label.setText(display_name)
        self._lang_label.show()
        self._lang_badge_timer.start(10000)

    @staticmethod
    def _lang_code_to_name(code: str) -> str:
        names = {
            "ja": "Japanese", "zh": "Chinese", "ko": "Korean",
            "es": "Spanish", "fr": "French", "de": "German",
            "pt": "Portuguese", "ru": "Russian", "it": "Italian",
            "en": "English",
        }
        return names.get(code, code.upper())

    def _apply_text_style(self) -> None:
        r, g, b = FONT_COLORS[self._font_color]
        self._text.setStyleSheet(f"""
            QLabel {{
                color: rgba({r}, {g}, {b}, {self._font_opacity});
                font-size: {self._font_size}px;
                font-weight: bold;
                background: transparent;
                padding: 2px;
            }}
        """)
        font = QFont("Segoe UI", self._font_size, QFont.Bold)
        font.setStyleHint(QFont.SansSerif)
        self._text.setFont(font)

    # --- Public API ---

    def set_theme(
        self,
        font_color: Optional[str] = None,
        bg_color: Optional[str] = None,
        font_size: Optional[int] = None,
        font_opacity: Optional[int] = None,
        bg_opacity: Optional[int] = None,
    ) -> None:
        """Apply appearance settings live (no restart)."""
        if font_color in FONT_COLORS:
            self._font_color = font_color
        if bg_color in BG_COLORS:
            self._bg_color = bg_color
        if font_size:
            self._font_size = max(10, min(96, int(font_size)))
        if font_opacity is not None:
            self._font_opacity = _clamp_byte(font_opacity)
        if bg_opacity is not None:
            self._bg_opacity = _clamp_byte(bg_opacity)

        self._apply_text_style()
        self._refresh_backdrop()
        self.update()

    def _refresh_backdrop(self) -> None:
        """Enable/disable the Windows acrylic backdrop for the Blur style."""
        want_blur = self._bg_color == "Blur"
        if want_blur == self._blur_active:
            return
        try:
            hwnd = int(self.winId())
        except (TypeError, RuntimeError):
            return
        if want_blur:
            self._blur_active = enable_blur(
                hwnd, BG_COLORS["Blur"], min(self._bg_opacity, 180)
            )
        else:
            disable_blur(hwnd)
            self._blur_active = False

    def show_subtitle(self, text: str) -> None:
        """Display a subtitle line (most recent at the bottom)."""
        if not text:
            self.clear()
            return
        if self._user_hidden:
            # Still track text so restoring shows recent context
            self._current_texts.append(text)
            self._current_texts = self._current_texts[-self._max_lines:]
            return

        self._current_texts.append(text)
        if len(self._current_texts) > self._max_lines:
            self._current_texts = self._current_texts[-self._max_lines:]

        self._render_texts()
        self._fade_in()
        self._reset_hide_timer()
        self.subtitle_shown.emit(text)

    def replace_last_subtitle(self, text: str) -> None:
        """Replace the most recent subtitle line in-place."""
        if not text or not self._current_texts:
            return
        self._current_texts[-1] = text
        self._render_texts()
        if not self._user_hidden:
            self.show()
            self._reset_hide_timer()

    def clear(self) -> None:
        self._disconnect_fade_finished()
        self._fade.stop()
        self._current_texts = []
        self._text.setText("")
        self._hide_timer.stop()
        self.update()

    def set_position(self, x: int, y: int, width: int, height: int) -> None:
        width = max(MIN_WIDTH, width)
        height = max(MIN_HEIGHT, height)
        self._box_height = height
        if not self._minimized:
            self._expanded_height = height
        self.setFixedSize(width, height)
        self.move(x, y)

    def toggle_minimized(self) -> None:
        """Collapse to a single-line bar, or restore the previous height."""
        if self._minimized:
            self._minimized = False
            self._min_button.setText("–")
            self._min_button.setToolTip("Minimize to one line")
            target = self._expanded_height
        else:
            self._minimized = True
            self._expanded_height = self.height()
            self._min_button.setText("▣")
            self._min_button.setToolTip("Restore size")
            target = MIN_HEIGHT
        self._box_height = target
        self.setFixedSize(self.width(), target)
        self._render_texts()
        self.size_changed.emit(self.width(), target)
        self._reset_hide_timer()

    def hide_by_user(self) -> None:
        """Close button: hide until restored from the tray."""
        self._user_hidden = True
        self._hide_timer.stop()
        self._fade.stop()
        self.hide()
        self.hidden_by_user.emit()

    def restore_from_user_hide(self) -> None:
        """Tray action: bring the overlay back."""
        self._user_hidden = False
        self.setWindowOpacity(1.0)
        self._render_texts()
        self.show()
        self.raise_()
        self._reset_hide_timer()

    @property
    def is_user_hidden(self) -> bool:
        return self._user_hidden

    # --- Internals ---

    def _reset_hide_timer(self) -> None:
        self._hide_timer.stop()
        self._hide_timer.start(int(self._auto_hide_delay * 1000))
        self._lang_badge_timer.stop()
        self._lang_badge_timer.start(10000)

    def _disconnect_fade_finished(self) -> None:
        try:
            self._fade.finished.disconnect(self._on_fade_out_complete)
        except TypeError:
            pass

    def _render_texts(self) -> None:
        lines = self._current_texts
        if self._minimized and lines:
            lines = lines[-1:]
        self._text.setText("\n".join(lines))
        self.update()

    def _fade_in(self) -> None:
        self._fade.stop()
        # A previous fade-out leaves finished→clear connected; if it survives
        # into fade-in, completion instantly wipes the new subtitle.
        self._disconnect_fade_finished()
        self.show()
        self.raise_()
        self.setWindowOpacity(0.0)
        self._fade.setStartValue(0.0)
        self._fade.setEndValue(1.0)
        self._fade.start()

    def _on_auto_hide(self) -> None:
        self._fade.stop()
        self._disconnect_fade_finished()
        self._fade.setStartValue(self.windowOpacity())
        self._fade.setEndValue(0.0)
        self._fade.finished.connect(self._on_fade_out_complete)
        self._fade.start()

    def _on_fade_out_complete(self) -> None:
        self._disconnect_fade_finished()
        self.clear()
        self.hide()
        self.setWindowOpacity(1.0)

    def _grip_rect(self) -> QRect:
        return QRect(
            self.width() - GRIP_SIZE,
            self.height() - GRIP_SIZE,
            GRIP_SIZE,
            GRIP_SIZE,
        )

    # --- Painting ---

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self._bg_color == "None":
            bg_alpha = 0
        elif self._bg_color == "Blur" and self._blur_active:
            # Acrylic already tints behind the window; keep the overlay light.
            bg_alpha = min(self._bg_opacity, 90)
        else:
            bg_alpha = self._bg_opacity

        r, g, b = BG_COLORS[self._bg_color]
        border_rgb = 0 if self._bg_color == "White" else 255
        painter.setPen(QPen(QColor(border_rgb, border_rgb, border_rgb, 45), 1))
        painter.setBrush(QBrush(QColor(r, g, b, bg_alpha)))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 14, 14)

        # Resize grip: three diagonal ticks in the bottom-right corner
        grip = self._grip_rect()
        painter.setPen(QPen(QColor(border_rgb, border_rgb, border_rgb, 110), 2))
        for offset in (4, 9, 14):
            painter.drawLine(
                grip.right() - offset, grip.bottom() - 2,
                grip.right() - 2, grip.bottom() - offset,
            )

    # --- Mouse: drag body, resize from corner ---

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        self._hide_timer.stop()
        if self._grip_rect().contains(event.pos()):
            self._resize_origin = event.globalPos()
            self._resize_start_size = (self.width(), self.height())
        else:
            self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._resize_origin is not None and self._resize_start_size is not None:
            delta = event.globalPos() - self._resize_origin
            new_w = max(MIN_WIDTH, self._resize_start_size[0] + delta.x())
            new_h = max(MIN_HEIGHT, self._resize_start_size[1] + delta.y())
            self._box_height = new_h
            if not self._minimized:
                self._expanded_height = new_h
            self.setFixedSize(new_w, new_h)
            event.accept()
            return

        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_offset)
            event.accept()
            return

        self.setCursor(
            Qt.SizeFDiagCursor
            if self._grip_rect().contains(event.pos())
            else Qt.SizeAllCursor
        )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return

        if self._resize_origin is not None:
            self._resize_origin = None
            self._resize_start_size = None
            self.size_changed.emit(self.width(), self.height())
        elif self._drag_offset is not None:
            self._drag_offset = None
            self.position_changed.emit(self.x(), self.y())

        self._reset_hide_timer()
        event.accept()

    def closeEvent(self, event) -> None:
        self._hide_timer.stop()
        self._fade.stop()
        super().closeEvent(event)


def _clamp_byte(value) -> int:
    try:
        return max(0, min(255, int(value)))
    except (TypeError, ValueError):
        return 255
