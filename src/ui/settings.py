"""Settings dialog for the subtitle overlay.

Controls language, playback speed, overlay geometry, colors/opacity,
performance, and LLM refinement. Changes persist to config.yaml.
"""

import logging
from pathlib import Path

import yaml
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QSlider, QSpinBox, QVBoxLayout,
)

from src.stt.whisper_stt import PLAYBACK_SPEED_PRESETS, clamp_playback_speed
from src.ui.overlay import BG_COLORS, DEFAULTS, FONT_COLORS

logger = logging.getLogger(__name__)

# Language code mapping (first entry is the default)
LANGUAGE_NAMES = {
    "Auto (Detect)": "auto",
    "Japanese": "ja",
    "English": "en",
    "Chinese": "zh",
    "Korean": "ko",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
}
NAME_TO_CODE = LANGUAGE_NAMES
CODE_TO_NAME = {v: k for k, v in LANGUAGE_NAMES.items()}

# Applied by the "Reset to defaults" button
DEFAULT_OVERLAY = {
    "x": None,
    "y": None,
    "width": None,
    "height": DEFAULTS["height"],
    "font_size": DEFAULTS["font_size"],
    "font_color": DEFAULTS["font_color"],
    "font_opacity": DEFAULTS["font_opacity"],
    "bg_color": DEFAULTS["bg_color"],
    "opacity": DEFAULTS["opacity"],
    "auto_hide_delay": DEFAULTS["auto_hide_delay"],
    "max_lines": DEFAULTS["max_lines"],
}


def _default_geometry(height: int) -> tuple:
    """Stock overlay geometry: 90% wide, bottom-center of the primary screen."""
    screen = (
        QApplication.primaryScreen().availableGeometry()
        if QApplication.instance()
        else QRect(0, 0, 1920, 1080)
    )
    width = max(640, int(screen.width() * 0.9))
    return width, (screen.width() - width) // 2, int(screen.height() * 0.78)


class SettingsDialog(QDialog):
    """Settings panel for the subtitle overlay and pipeline."""

    # Emitted whenever config is written, so the overlay can restyle live.
    settings_applied = pyqtSignal()

    def __init__(self, config: dict, config_path: str = "config.yaml", parent=None):
        super().__init__(parent)
        self.config = config
        self.config_path = config_path
        self.setWindowTitle("Subtitle Translator Settings")
        self.setMinimumWidth(460)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        model_cfg = config.get("model", {})
        audio_cfg = config.get("audio", {})
        overlay_cfg = config.get("overlay", {})
        llm_cfg = config.get("llm", {})
        compare_cfg = config.get("compare", {})

        # --- Language ---
        lang_group = QGroupBox("Language")
        lang_layout = QFormLayout(lang_group)

        names = sorted(k for k in LANGUAGE_NAMES if k != "Auto (Detect)")
        names.insert(0, "Auto (Detect)")
        self.source_combo = QComboBox()
        self.source_combo.addItems(names)
        self.source_combo.setCurrentText(
            CODE_TO_NAME.get(model_cfg.get("language", "auto"), "Auto (Detect)")
        )
        lang_layout.addRow("Source Language:", self.source_combo)

        target_names = sorted(k for k in LANGUAGE_NAMES if k != "Auto (Detect)")
        self.target_combo = QComboBox()
        self.target_combo.addItems(target_names)
        self.target_combo.setCurrentText(
            CODE_TO_NAME.get(model_cfg.get("target_language", "en"), "English")
        )
        lang_layout.addRow("Target Language:", self.target_combo)

        self.lang_warning = QLabel("")
        self.lang_warning.setStyleSheet("color: orange; font-size: 11px;")
        self.lang_warning.hide()
        lang_layout.addRow("", self.lang_warning)
        self.source_combo.currentTextChanged.connect(self._validate_language)
        self.target_combo.currentTextChanged.connect(self._validate_language)
        layout.addWidget(lang_group)

        # --- Playback speed ---
        speed_group = QGroupBox("Video Playback Speed")
        speed_layout = QFormLayout(speed_group)
        self.speed_combo = QComboBox()
        for speed in PLAYBACK_SPEED_PRESETS:
            self.speed_combo.addItem(f"{speed:g}x", speed)
        current_speed = clamp_playback_speed(audio_cfg.get("playback_speed", 1.0))
        nearest = min(PLAYBACK_SPEED_PRESETS, key=lambda s: abs(s - current_speed))
        self.speed_combo.setCurrentIndex(PLAYBACK_SPEED_PRESETS.index(nearest))
        self.speed_combo.setToolTip(
            "Match the speed you play the video at (0.75x–1.5x)."
        )
        speed_layout.addRow("Player speed:", self.speed_combo)
        layout.addWidget(speed_group)

        # --- Overlay geometry ---
        pos_group = QGroupBox("Overlay Size && Position")
        pos_layout = QFormLayout(pos_group)

        # Unset geometry means "stock bottom-center"; show that rather than an
        # arbitrary number, so pressing Apply does not move or shrink the box.
        stock_h = overlay_cfg.get("height") or DEFAULTS["height"]
        stock_w, stock_x, stock_y = _default_geometry(stock_h)

        self.x_spin = self._spin(0, 9999, overlay_cfg.get("x") or stock_x)
        pos_layout.addRow("X:", self.x_spin)
        self.y_spin = self._spin(0, 9999, overlay_cfg.get("y") or stock_y)
        pos_layout.addRow("Y:", self.y_spin)
        self.width_spin = self._spin(320, 9999, overlay_cfg.get("width") or stock_w)
        pos_layout.addRow("Width:", self.width_spin)
        self.height_spin = self._spin(90, 999, stock_h)
        pos_layout.addRow("Height:", self.height_spin)

        geo_hint = QLabel("Tip: drag the box to move, drag its corner to resize.")
        geo_hint.setStyleSheet("color: gray; font-size: 11px;")
        pos_layout.addRow("", geo_hint)
        layout.addWidget(pos_group)

        # --- Colors and appearance ---
        appear_group = QGroupBox("Appearance")
        appear_layout = QFormLayout(appear_group)

        self.font_color_combo = QComboBox()
        self.font_color_combo.addItems(list(FONT_COLORS.keys()))
        self.font_color_combo.setCurrentText(
            overlay_cfg.get("font_color", DEFAULTS["font_color"])
        )
        appear_layout.addRow("Text Color:", self.font_color_combo)

        self.bg_color_combo = QComboBox()
        self.bg_color_combo.addItems(list(BG_COLORS.keys()))
        self.bg_color_combo.setCurrentText(
            overlay_cfg.get("bg_color", DEFAULTS["bg_color"])
        )
        self.bg_color_combo.setToolTip(
            "Blur uses the Windows acrylic backdrop; None makes the box invisible."
        )
        appear_layout.addRow("Background:", self.bg_color_combo)

        self.font_slider, font_row, self.font_label = self._slider_row(
            10, 96, overlay_cfg.get("font_size", DEFAULTS["font_size"]), "px"
        )
        appear_layout.addRow("Font Size:", font_row)

        self.font_opacity_slider, fo_row, self.font_opacity_label = self._slider_row(
            40, 255, overlay_cfg.get("font_opacity", DEFAULTS["font_opacity"])
        )
        appear_layout.addRow("Text Opacity:", fo_row)

        self.opacity_slider, bo_row, self.opacity_label = self._slider_row(
            0, 255, overlay_cfg.get("opacity", DEFAULTS["opacity"])
        )
        appear_layout.addRow("Bg Opacity:", bo_row)

        self.hide_spin = self._spin(
            1, 30, int(overlay_cfg.get("auto_hide_delay", DEFAULTS["auto_hide_delay"]))
        )
        self.hide_spin.setSuffix("s")
        appear_layout.addRow("Auto-hide:", self.hide_spin)

        self.lines_spin = self._spin(
            1, 5, int(overlay_cfg.get("max_lines", DEFAULTS["max_lines"]))
        )
        appear_layout.addRow("Max Lines:", self.lines_spin)
        layout.addWidget(appear_group)

        # --- Performance ---
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.model_combo.setCurrentText(model_cfg.get("size", "small"))
        self.model_combo.setToolTip("Smaller = faster and lighter on your GPU/CPU.")
        perf_layout.addRow("Whisper model:", self.model_combo)

        self.beam_spin = self._spin(1, 5, int(model_cfg.get("beam_size", 3)))
        self.beam_spin.setToolTip("1–2 is fastest. 5 is most accurate.")
        perf_layout.addRow("Beam size:", self.beam_spin)

        self.threads_spin = self._spin(1, 16, int(model_cfg.get("cpu_threads", 4)))
        self.threads_spin.setToolTip("CPU threads Whisper may use.")
        perf_layout.addRow("CPU threads:", self.threads_spin)

        self.compare_check = QCheckBox("Log Japanese + English for comparison")
        self.compare_check.setChecked(bool(compare_cfg.get("enabled", False)))
        self.compare_check.setToolTip(
            "Doubles Whisper work (two passes). Turn on only for accuracy testing."
        )
        perf_layout.addRow("", self.compare_check)
        perf_hint = QLabel("Model size and comparison logging take effect on restart.")
        perf_hint.setStyleSheet("color: gray; font-size: 11px;")
        perf_layout.addRow("", perf_hint)
        layout.addWidget(perf_group)

        # --- LLM ---
        llm_group = QGroupBox("Translation Refinement")
        llm_layout = QVBoxLayout(llm_group)
        self.llm_check = QCheckBox("Enable LLM refinement (needs LLM_API_KEY)")
        self.llm_check.setChecked(llm_cfg.get("enabled", False))
        self.llm_check.setToolTip(
            "Optional. Whisper already translates to English offline."
        )
        llm_layout.addWidget(self.llm_check)
        layout.addWidget(llm_group)

        # --- Buttons ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
            | QDialogButtonBox.Apply
            | QDialogButtonBox.RestoreDefaults
        )
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_settings)
        buttons.button(QDialogButtonBox.RestoreDefaults).setText("Reset to defaults")
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            self.reset_to_defaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # --- Widget helpers ---

    def _spin(self, low: int, high: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setValue(max(low, min(high, int(value or low))))
        return spin

    def _slider_row(self, low: int, high: int, value: int, suffix: str = ""):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(max(low, min(high, int(value))))
        label = QLabel(f"{slider.value()}{suffix}")
        slider.valueChanged.connect(lambda v: label.setText(f"{v}{suffix}"))
        row = QHBoxLayout()
        row.addWidget(slider)
        row.addWidget(label)
        return slider, row, label

    def _validate_language(self):
        src = self.source_combo.currentText()
        tgt = self.target_combo.currentText()
        if src == "Auto (Detect)":
            self.lang_warning.setText("Language will be detected automatically from speech")
            self.lang_warning.setStyleSheet("color: gray; font-size: 11px;")
            self.lang_warning.show()
        elif src == tgt:
            self.lang_warning.setText("Source and target should differ!")
            self.lang_warning.setStyleSheet("color: orange; font-size: 11px;")
            self.lang_warning.show()
        else:
            self.lang_warning.hide()

    # --- Actions ---

    def reset_to_defaults(self) -> None:
        """Restore stock overlay appearance and safe performance settings."""
        self.font_color_combo.setCurrentText(DEFAULTS["font_color"])
        self.bg_color_combo.setCurrentText(DEFAULTS["bg_color"])
        self.font_slider.setValue(DEFAULTS["font_size"])
        self.font_opacity_slider.setValue(DEFAULTS["font_opacity"])
        self.opacity_slider.setValue(DEFAULTS["opacity"])
        self.hide_spin.setValue(int(DEFAULTS["auto_hide_delay"]))
        self.lines_spin.setValue(DEFAULTS["max_lines"])

        width, x, y = _default_geometry(DEFAULTS["height"])
        self.height_spin.setValue(DEFAULTS["height"])
        self.width_spin.setValue(width)
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)

        self.model_combo.setCurrentText("small")
        self.beam_spin.setValue(3)
        self.threads_spin.setValue(4)
        self.compare_check.setChecked(False)
        self.apply_settings()

    def apply_settings(self) -> dict:
        """Write UI values into the config dict and save config.yaml."""
        overlay = self.config.setdefault("overlay", {})
        overlay["x"] = self.x_spin.value()
        overlay["y"] = self.y_spin.value()
        overlay["width"] = self.width_spin.value()
        overlay["height"] = self.height_spin.value()
        overlay["font_size"] = self.font_slider.value()
        overlay["font_color"] = self.font_color_combo.currentText()
        overlay["font_opacity"] = self.font_opacity_slider.value()
        overlay["bg_color"] = self.bg_color_combo.currentText()
        overlay["opacity"] = self.opacity_slider.value()
        overlay["auto_hide_delay"] = self.hide_spin.value()
        overlay["max_lines"] = self.lines_spin.value()

        model = self.config.setdefault("model", {})
        model["language"] = NAME_TO_CODE.get(self.source_combo.currentText(), "auto")
        model["target_language"] = NAME_TO_CODE.get(
            self.target_combo.currentText(), "en"
        )
        model["size"] = self.model_combo.currentText()
        model["beam_size"] = self.beam_spin.value()
        model["cpu_threads"] = self.threads_spin.value()

        self.config.setdefault("llm", {})["enabled"] = self.llm_check.isChecked()
        self.config.setdefault("audio", {})["playback_speed"] = clamp_playback_speed(
            self.speed_combo.currentData()
        )
        self.config.setdefault("compare", {})["enabled"] = self.compare_check.isChecked()

        try:
            with open(Path(self.config_path), "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info("Settings saved to %s", self.config_path)
        except Exception as e:
            logger.error("Failed to save settings: %s", e)

        self.settings_applied.emit()
        return self.config
