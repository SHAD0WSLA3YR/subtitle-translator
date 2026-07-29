"""Real-Time Japanese to English Subtitle Translator - Application Controller.

Wires together all modules: audio capture, VAD, STT, LLM refinement,
the PyQt5 overlay, system tray icon, settings dialog, and session
history into a single desktop application.
"""

import sys
import yaml
import logging
import argparse
import datetime
from pathlib import Path
from typing import Optional

# IMPORTANT: Import faster_whisper BEFORE PyQt5 to avoid CUDA context conflict.
# When PyQt5 is imported first, its Qt platform plugin initializes CUDA in a way
# that causes faster_whisper's model loading to hang indefinitely.
import faster_whisper  # noqa: F401  (must come before PyQt5 imports)

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QTimer

from src.audio.capture import AudioCapture
from src.audio.vad import VADProcessor
from src.stt.processor import TranslationProcessor
from src.stt.whisper_stt import clamp_playback_speed
from src.llm.refiner import LLMRefiner
from src.ui.overlay import SubtitleOverlay
from src.ui.tray import TrayIcon
from src.ui.settings import SettingsDialog
from src.ui.history_dialog import HistoryDialog
from src.core.history import HistoryManager
from src.core.pipeline import PipelineController, PipelineState
from src.core.comparison import ComparisonLogger, default_comparison_path
from src.version import __version__

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config file not found: {config_path}, using defaults")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class SubtitleApp:
    """Application controller wiring all modules together."""

    def __init__(
        self,
        config: dict,
        srt_path: Optional[str] = None,
        compare_path: Optional[str] = None,
    ):
        self.config = config
        self.srt_path = srt_path
        self._srt_file = None
        self._srt_index = 0
        self._clause_start_time: Optional[datetime.datetime] = None
        self._capture_retries = 0
        self._max_capture_retries = 5
        self._current_detected_lang: str = ""
        self._settings_dialog: Optional[SettingsDialog] = None
        self._compare: Optional[ComparisonLogger] = None

        # Extract config sections
        model_cfg = config.get("model", {})
        audio_cfg = config.get("audio", {})
        vad_cfg = config.get("vad", {})
        llm_cfg = config.get("llm", {})
        overlay_cfg = config.get("overlay", {})
        compare_cfg = config.get("compare", {})

        # --- PyQt5 Application ---
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        self.app.setApplicationName("Subtitle Translator")
        self.app.setApplicationVersion(__version__)
        self.app.setOrganizationName("SubtitleTranslator")

        # --- Overlay ---
        self.overlay = SubtitleOverlay(
            x=overlay_cfg.get("x"),
            y=overlay_cfg.get("y"),
            width=overlay_cfg.get("width"),
            height=overlay_cfg.get("height", 180),
            font_size=overlay_cfg.get("font_size", 26),
            opacity=overlay_cfg.get("opacity", 200),
            auto_hide_delay=overlay_cfg.get("auto_hide_delay", 6.0),
            max_lines=overlay_cfg.get("max_lines", 3),
            font_color=overlay_cfg.get("font_color", "White"),
            bg_color=overlay_cfg.get("bg_color", "Black"),
            font_opacity=overlay_cfg.get("font_opacity", 255),
        )
        self.overlay.position_changed.connect(self._on_overlay_moved)
        self.overlay.size_changed.connect(self._on_overlay_resized)
        self.overlay.hidden_by_user.connect(self._on_overlay_hidden)

        # --- LLM Refiner (optional; Whisper translate works fully offline) ---
        self.refiner: Optional[LLMRefiner] = None
        if llm_cfg.get("enabled", False):
            self.refiner = LLMRefiner(
                provider=llm_cfg.get("provider", "openrouter"),
                model=llm_cfg.get("model", "openrouter/free"),
                api_key_env=llm_cfg.get("api_key_env", "LLM_API_KEY"),
                temperature=llm_cfg.get("temperature", 0.1),
                max_tokens=llm_cfg.get("max_tokens", 256),
                enabled=True,
            )
            key_loaded = self.refiner.load_api_key()
            if not key_loaded:
                logger.warning(
                    "LLM refinement enabled but %s env var not set — using Whisper only",
                    llm_cfg.get("api_key_env", "LLM_API_KEY"),
                )
                self.refiner.enabled = False

        # Comparison logging needs the source-language pass, which costs a
        # second Whisper run per clause. Resolve it before building the STT.
        self._compare_path = self._resolve_compare_path(compare_cfg, compare_path)

        # --- STT Processor (queue + Whisper) ---
        audio_speed = clamp_playback_speed(audio_cfg.get("playback_speed", 1.0))
        self.processor = TranslationProcessor(
            model_size=model_cfg.get("size", "small"),
            device=model_cfg.get("device", "cuda"),
            compute_type=model_cfg.get("compute_type", "int8_float16"),
            language=model_cfg.get("language", "ja"),
            beam_size=model_cfg.get("beam_size", 3),
            playback_speed=audio_speed,
            cpu_threads=model_cfg.get("cpu_threads", 4),
            dual_pass=self._compare_path is not None,
        )

        # --- VAD ---
        self.vad = VADProcessor(
            sample_rate=16000,
            threshold=vad_cfg.get("threshold", 0.3),
            min_speech_duration_ms=vad_cfg.get("min_speech_duration_ms", 500),
            min_silence_duration_ms=vad_cfg.get("min_silence_duration_ms", 800),
            speech_pad_ms=vad_cfg.get("speech_pad_ms", 300),
            max_speech_duration_ms=vad_cfg.get("max_speech_duration_ms", 8000),
            silence_floor=vad_cfg.get("silence_floor", 0.004),
            on_clause=self._on_clause,
        )

        # --- Audio Capture ---
        self.capture = AudioCapture(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            blocksize=audio_cfg.get("blocksize", 1024),
        )

        # --- Session History ---
        self.history = HistoryManager()

        # --- Pipeline Controller ---
        self.pipeline = PipelineController(
            capture=self.capture,
            vad=self.vad,
            processor=self.processor,
            overlay=self.overlay,
            refiner=self.refiner,
            parent=self.overlay,
        )
        self.pipeline.state_changed.connect(self._on_pipeline_state)
        self.pipeline.translation_output.connect(self._on_translation_ui)
        self.pipeline.translation_refined.connect(self._on_translation_refined)

        # --- System Tray ---
        self.tray = TrayIcon(parent=self.overlay, playback_speed=audio_speed)
        self.tray.pause_toggled.connect(self._on_tray_toggle)
        self.tray.show_settings.connect(self._on_show_settings)
        self.tray.show_history.connect(self._on_show_history)
        self.tray.quit_app.connect(self._on_tray_quit)
        self.tray.playback_speed_changed.connect(self._on_playback_speed_changed)
        self.tray.toggle_overlay.connect(self._on_toggle_overlay)

        # --- Health Monitor Timer (checks capture thread every 2s) ---
        self._health_timer = QTimer(self.overlay)
        self._health_timer.setInterval(2000)
        self._health_timer.timeout.connect(self._check_capture_health)

        # Open SRT file if requested
        if self.srt_path:
            try:
                self._srt_file = open(self.srt_path, "w", encoding="utf-8")
                logger.info("Writing subtitles to %s", self.srt_path)
            except Exception as e:
                logger.error("Failed to open SRT file: %s", e)
                self._srt_file = None

        # Comparison log: heard JA + translated EN (for YouTube side-by-side)
        if self._compare_path is not None:
            self._compare = ComparisonLogger(self._compare_path)
            logger.info(
                "Comparison log: %s (+ %s) — source pass on, ~2x Whisper cost",
                self._compare_path,
                self._compare_path.with_suffix(".md"),
            )

    @staticmethod
    def _resolve_compare_path(
        compare_cfg: dict, cli_path: Optional[str]
    ) -> Optional[Path]:
        """Where to write the JA/EN comparison log, or None to skip it."""
        if cli_path:
            return Path(cli_path)
        if not compare_cfg.get("enabled", False):
            return None
        cfg_path = compare_cfg.get("path")
        return Path(cfg_path) if cfg_path else default_comparison_path()

    # --- Event handlers ---

    def _on_clause(self, audio):
        """Called by VAD when a complete clause is detected."""
        self._clause_start_time = datetime.datetime.now()
        self.processor.add_clause(audio)

    def _on_overlay_moved(self, x: int, y: int):
        """Persist drag position into in-memory config (saved via Settings)."""
        self.config.setdefault("overlay", {})
        self.config["overlay"]["x"] = x
        self.config["overlay"]["y"] = y

    def _on_overlay_resized(self, width: int, height: int):
        """Persist the size chosen with the corner grip / minimize button."""
        self.config.setdefault("overlay", {})
        self.config["overlay"]["width"] = width
        self.config["overlay"]["height"] = height

    def _on_overlay_hidden(self):
        """Overlay close button was pressed."""
        self.tray.set_overlay_visible(False)
        self.tray.show_balloon(
            "Subtitles hidden",
            "Right-click the tray icon → Show subtitles to bring them back.",
        )

    def _on_toggle_overlay(self):
        """Tray menu toggled overlay visibility."""
        if self.overlay.is_user_hidden:
            self.overlay.restore_from_user_hide()
            self.tray.set_overlay_visible(True)
        else:
            self.overlay.hide_by_user()

    def _on_translation_ui(self, heard_text: str, translated_text: str, detected_lang: str = ""):
        """Called on Qt main thread via pipeline signal (thread-safe for overlay)."""
        self._current_detected_lang = detected_lang
        display = translated_text or heard_text
        self.overlay.show_subtitle(display)
        if detected_lang:
            if hasattr(self.overlay, "set_detected_language"):
                self.overlay.set_detected_language(detected_lang)
            self.tray.set_detected_language(detected_lang)

        try:
            self.history.log_subtitle(heard_text, translated_text, detected_lang=detected_lang)
        except Exception as e:
            logger.debug("History log failed: %s", e)

        if self._compare is not None:
            try:
                self._compare.log(heard_text, translated_text)
            except Exception as e:
                logger.debug("Comparison log failed: %s", e)

        if self._srt_file is not None:
            self._write_srt(display)

    def _on_translation_refined(self, refined_text: str):
        """Async LLM polish arrived — swap the latest overlay line in place."""
        self.overlay.replace_last_subtitle(refined_text)
        try:
            self.history.log_subtitle(refined_text, refined_text)
        except Exception as e:
            logger.debug("History refined log failed: %s", e)

    def _write_srt(self, text: str):
        """Append a subtitle entry to the SRT file."""
        if self._srt_file is None:
            return
        self._srt_index += 1
        now = datetime.datetime.now()
        start_ts = self._clause_start_time.strftime("%H:%M:%S,%f")[:-3] if self._clause_start_time else now.strftime("%H:%M:%S,%f")[:-3]
        end_ts = now.strftime("%H:%M:%S,%f")[:-3]
        self._srt_file.write(f"{self._srt_index}\n{start_ts} --> {end_ts}\n{text}\n\n")
        self._srt_file.flush()

    def _check_capture_health(self):
        """Periodic health check: restart capture if the thread died."""
        if not self.capture._running:
            return  # Already stopped via shutdown()

        thread = getattr(self.capture, "_thread", None)
        if thread is not None and not thread.is_alive():
            self._capture_retries += 1
            if self._capture_retries > self._max_capture_retries:
                logger.error("Capture thread keeps dying — giving up after %d retries", self._max_capture_retries)
                self._health_timer.stop()
                self.overlay.show_subtitle("Audio error — restart the app")
                return

            logger.warning("Capture thread died (retry %d/%d)", self._capture_retries, self._max_capture_retries)
            self.overlay.show_subtitle("Audio device lost — reconnecting...")
            self._restart_capture()

    def _restart_capture(self):
        """Create a fresh AudioCapture and start it."""
        audio_cfg = self.config.get("audio", {})
        self.capture = AudioCapture(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            blocksize=audio_cfg.get("blocksize", 1024),
        )
        self.capture.start(on_audio=self.vad.process_chunk)
        self.overlay.replace_last_subtitle("Reconnected — listening...")
        QTimer.singleShot(2000, lambda: self.overlay._reset_hide_timer())

    # --- Tray actions ---

    def _on_pipeline_state(self, state: PipelineState):
        """React to pipeline state changes."""
        self.tray.set_paused(state == PipelineState.PAUSED)

    def _on_tray_toggle(self):
        """Toggle pipeline pause/resume from tray menu."""
        self.pipeline.toggle()

    def _on_show_settings(self):
        """Open the settings dialog."""
        dlg = SettingsDialog(self.config, parent=self.overlay)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        # Apply / Reset restyle the overlay immediately, not just on OK.
        dlg.settings_applied.connect(self._apply_settings)
        if dlg.exec_() == dlg.Accepted:
            dlg.apply_settings()
        self._settings_dialog = None

    def _on_show_history(self):
        """Open the history dialog."""
        dlg = HistoryDialog(self.history, parent=self.overlay)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.exec_()

    def _on_playback_speed_changed(self, speed: float):
        """Tray menu changed playback speed — apply live and persist."""
        applied = self.processor.stt.set_playback_speed(speed)
        self.config.setdefault("audio", {})
        self.config["audio"]["playback_speed"] = applied
        self._persist_config()
        self.tray.show_balloon(
            "Playback speed",
            f"Compensating for {applied:g}x video playback",
        )
        self.overlay.show_subtitle(f"Speed set to {applied:g}x")

    def _persist_config(self):
        """Write current config to config.yaml."""
        try:
            path = Path("config.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            logger.error("Failed to save config: %s", e)

    def _on_tray_quit(self):
        """Quit the application from tray menu."""
        reply = QMessageBox.question(
            self.overlay,
            "Quit Subtitle Translator",
            "Are you sure you want to quit?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.shutdown()
            self.app.quit()

    def _apply_settings(self):
        """Apply settings changes that need runtime updates."""
        ov = self.config.get("overlay", {})
        self.overlay.set_position(
            ov.get("x", 0) or 0,
            ov.get("y", 0) or 0,
            ov.get("width") or self.overlay.width(),
            ov.get("height", 180),
        )
        self.overlay.set_theme(
            font_color=ov.get("font_color"),
            bg_color=ov.get("bg_color"),
            font_size=ov.get("font_size"),
            font_opacity=ov.get("font_opacity"),
            bg_opacity=ov.get("opacity"),
        )
        self.overlay._auto_hide_delay = float(ov.get("auto_hide_delay", 6.0))
        self.overlay._max_lines = int(ov.get("max_lines", 3))

        speed = self.config.get("audio", {}).get("playback_speed", 1.0)
        applied = self.processor.stt.set_playback_speed(speed)
        self.tray.set_playback_speed(applied, emit=False)

        self.processor.stt.beam_size = int(
            self.config.get("model", {}).get("beam_size", 3)
        )
        logger.info("Settings applied (restart required for Whisper model / LLM changes)")

    # --- Lifecycle ---

    def run(self) -> int:
        """Start the application.

        Returns:
            Exit code (0 for success).
        """
        model_cfg = self.config.get("model", {})
        audio_cfg = self.config.get("audio", {})
        llm_enabled = bool(self.refiner and self.refiner.enabled)
        srt_enabled = self.srt_path is not None

        print("=" * 55)
        print("  Real-Time Japanese to English Subtitle Translator")
        print("=" * 55)
        print(f"  Model:     {model_cfg.get('size', 'small')} (beam {model_cfg.get('beam_size', 3)})")
        print(f"  Device:    {model_cfg.get('device', 'cuda')}")
        lang_display = model_cfg.get("language", "auto")
        if lang_display == "auto":
            lang_display = "Auto-detect"
        print(f"  Language:  {lang_display}")
        print(f"  Speed:     {audio_cfg.get('playback_speed', 1.0)}x (set to match player)")
        print(f"  Passes:    {'2 (translate + source)' if self._compare_path else '1 (translate only)'}")
        print(f"  LLM:       {'enabled' if llm_enabled else 'disabled'}")
        print(f"  Version:   v{__version__}")
        if self.srt_path:
            print(f"  SRT:       {self.srt_path}")
        if self._compare is not None:
            print(f"  Compare:   {self._compare.path}")
            print(f"             {self._compare.md_path}")
        print("=" * 55)
        print("  Starting pipeline...")
        print("  Drag the box to move, drag its corner to resize.")
        print("  – minimizes to one line, ✕ hides it (restore from tray).")
        print("=" * 55)

        # Start session history
        self.history.start_session(
            source=model_cfg.get("language", "ja"),
            target=model_cfg.get("target_language", "en"),
        )

        # Start the pipeline (processor preloads Whisper model, then starts capture)
        # Translation results arrive via pipeline.translation_output signal → _on_translation_ui
        self.pipeline.start()

        # Show the overlay
        self.overlay.show()
        self.overlay.show_subtitle("Waiting for audio...")
        self.overlay._reset_hide_timer()
        QTimer.singleShot(3000, lambda: self.overlay._reset_hide_timer())

        # Show tray
        self.tray.show()

        # Start health monitor
        self._health_timer.start()

        print("  Pipeline active. Right-click tray icon to pause/settings.")
        print("=" * 55)

        # Enter PyQt5 event loop
        try:
            exit_code = self.app.exec_()
        except KeyboardInterrupt:
            exit_code = 0

        self.shutdown()
        return exit_code

    def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down...")
        self._health_timer.stop()
        self.overlay._hide_timer.stop()
        self.pipeline.shutdown()
        self.overlay.hide()
        self.tray.hide()
        self.history.close()
        if self._compare is not None:
            self._compare.close()
            self._compare = None
        if self._srt_file:
            self._srt_file.close()
        logger.info("Shutdown complete")


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-Time Japanese to English Subtitle Translator",
    )
    parser.add_argument(
        "-c", "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--srt",
        help="Export subtitles to SRT file (e.g., subtitles.srt)",
    )
    parser.add_argument(
        "--compare",
        nargs="?",
        const="auto",
        default=None,
        help="Write heard(JA)+translated(EN) comparison log "
             "(optional path; default timestamped file under comparison/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Application entry point."""
    args = parse_args(argv)
    setup_logging(args.verbose)
    config = load_config(args.config)

    compare_path = None
    if args.compare == "auto":
        compare_path = str(default_comparison_path())
    elif args.compare:
        compare_path = args.compare

    app = SubtitleApp(config, srt_path=args.srt, compare_path=compare_path)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
