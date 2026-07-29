"""Checks for the overlay window controls: minimize, close, resize, theming.

Run:  python -m unittest tests.test_overlay_controls -v
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication

from src.ui.overlay import MIN_HEIGHT, MIN_WIDTH, SubtitleOverlay

_app = None


def setUpModule():
    global _app
    _app = QApplication.instance() or QApplication([])


class MinimizeTests(unittest.TestCase):
    def test_toggle_collapses_and_restores_height(self):
        overlay = SubtitleOverlay(width=800, height=200)
        overlay.toggle_minimized()
        self.assertEqual(overlay.height(), MIN_HEIGHT)

        overlay.toggle_minimized()
        self.assertEqual(overlay.height(), 200)
        overlay.close()

    def test_minimized_shows_only_the_latest_line(self):
        overlay = SubtitleOverlay(width=800, height=200, max_lines=3)
        overlay.show_subtitle("first")
        overlay.show_subtitle("second")
        self.assertIn("first", overlay._text.text())

        overlay.toggle_minimized()
        self.assertEqual(overlay._text.text(), "second")
        overlay.close()


class CloseButtonTests(unittest.TestCase):
    def test_hide_by_user_suppresses_new_subtitles(self):
        overlay = SubtitleOverlay(width=800, height=200)
        emitted = []
        overlay.hidden_by_user.connect(lambda: emitted.append(True))

        overlay.hide_by_user()
        self.assertTrue(overlay.is_user_hidden)
        self.assertEqual(emitted, [True])

        overlay.show_subtitle("while hidden")
        self.assertTrue(overlay.isHidden())

        overlay.restore_from_user_hide()
        self.assertFalse(overlay.is_user_hidden)
        self.assertIn("while hidden", overlay._text.text())
        overlay.close()


def _mouse_event(kind, local, global_pos, buttons):
    """QMouseEvent needs an explicit screenPos; it does not derive it."""
    return QMouseEvent(
        kind, local, global_pos, Qt.LeftButton, buttons, Qt.NoModifier
    )


class ResizeTests(unittest.TestCase):
    def _drag_grip(self, overlay, dx, dy):
        local = overlay.rect().bottomRight() - QPoint(4, 4)
        origin = overlay.mapToGlobal(local)
        moved = origin + QPoint(dx, dy)
        overlay.mousePressEvent(
            _mouse_event(QMouseEvent.MouseButtonPress, local, origin, Qt.LeftButton)
        )
        overlay.mouseMoveEvent(
            _mouse_event(
                QMouseEvent.MouseMove, local + QPoint(dx, dy), moved, Qt.LeftButton
            )
        )
        overlay.mouseReleaseEvent(
            _mouse_event(
                QMouseEvent.MouseButtonRelease,
                local + QPoint(dx, dy),
                moved,
                Qt.NoButton,
            )
        )

    def test_corner_drag_resizes_and_reports(self):
        overlay = SubtitleOverlay(x=100, y=100, width=800, height=200)
        overlay.show()
        sizes = []
        overlay.size_changed.connect(lambda w, h: sizes.append((w, h)))

        self._drag_grip(overlay, 120, 60)
        self.assertEqual(overlay.width(), 920)
        self.assertEqual(overlay.height(), 260)
        self.assertEqual(sizes, [(920, 260)])
        overlay.close()

    def test_resize_respects_minimums(self):
        overlay = SubtitleOverlay(x=100, y=100, width=MIN_WIDTH, height=MIN_HEIGHT)
        overlay.show()
        self._drag_grip(overlay, -500, -500)
        self.assertEqual(overlay.width(), MIN_WIDTH)
        self.assertEqual(overlay.height(), MIN_HEIGHT)
        overlay.close()

    def test_body_drag_moves_without_resizing(self):
        overlay = SubtitleOverlay(x=100, y=100, width=800, height=200)
        overlay.show()
        positions = []
        overlay.position_changed.connect(lambda x, y: positions.append((x, y)))

        center = overlay.rect().center()
        global_center = overlay.mapToGlobal(center)
        overlay.mousePressEvent(_mouse_event(
            QMouseEvent.MouseButtonPress, center, global_center, Qt.LeftButton
        ))
        overlay.mouseReleaseEvent(_mouse_event(
            QMouseEvent.MouseButtonRelease, center, global_center, Qt.NoButton
        ))
        self.assertEqual(overlay.size().width(), 800)
        self.assertEqual(len(positions), 1)
        overlay.close()


class ThemeTests(unittest.TestCase):
    def test_set_theme_updates_colors_live(self):
        overlay = SubtitleOverlay(width=800, height=200)
        overlay.set_theme(
            font_color="Yellow", bg_color="Grey", font_size=40,
            font_opacity=200, bg_opacity=120,
        )
        self.assertEqual(overlay._font_color, "Yellow")
        self.assertEqual(overlay._bg_color, "Grey")
        self.assertEqual(overlay._bg_opacity, 120)
        self.assertIn("255, 212, 0, 200", overlay._text.styleSheet())
        self.assertIn("font-size: 40px", overlay._text.styleSheet())
        overlay.close()

    def test_unknown_values_are_ignored(self):
        overlay = SubtitleOverlay(width=800, height=200, font_color="Nope")
        self.assertEqual(overlay._font_color, "White")
        overlay.set_theme(font_color="Chartreuse", bg_opacity=999)
        self.assertEqual(overlay._font_color, "White")
        self.assertEqual(overlay._bg_opacity, 255)
        overlay.close()


if __name__ == "__main__":
    unittest.main()
