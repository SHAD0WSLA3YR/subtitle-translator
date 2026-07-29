"""Session comparison log: heard (JA) vs translated (EN) side-by-side.

Writes TSV + Markdown so you can paste YouTube transcript into a third column
and compare mid-sentence cuts / translation quality.
"""

import csv
import datetime
from pathlib import Path
from typing import Optional, TextIO


class ComparisonLogger:
    """Append-only comparison log for one capture session."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tsv: Optional[TextIO] = None
        self._md: Optional[TextIO] = None
        self._index = 0
        self.md_path = self.path.with_suffix(".md")

        self._tsv = open(self.path, "w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._tsv, delimiter="\t")
        self._writer.writerow(
            ["index", "timestamp", "heard_ja", "translated_en", "youtube_ref"]
        )
        self._tsv.flush()

        self._md = open(self.md_path, "w", encoding="utf-8")
        self._md.write("# Subtitle comparison log\n\n")
        self._md.write(
            "Fill in **YouTube** lines from the official transcript, then compare.\n\n"
        )
        self._md.write("| # | Time | Heard (JA) | Translated (EN) | YouTube |\n")
        self._md.write("|---|------|------------|-----------------|--------|\n")
        self._md.flush()

    def log(self, heard: str, translated: str) -> None:
        if self._tsv is None:
            return
        self._index += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        heard_clean = (heard or "").replace("\t", " ").replace("\n", " ").strip()
        en_clean = (translated or "").replace("\t", " ").replace("\n", " ").strip()
        self._writer.writerow([self._index, ts, heard_clean, en_clean, ""])
        self._tsv.flush()

        if self._md is not None:
            # Escape pipes for markdown tables
            h = heard_clean.replace("|", "\\|")
            e = en_clean.replace("|", "\\|")
            self._md.write(f"| {self._index} | {ts} | {h} | {e} |  |\n")
            self._md.flush()

    def close(self) -> None:
        if self._tsv is not None:
            self._tsv.close()
            self._tsv = None
        if self._md is not None:
            self._md.close()
            self._md = None


def default_comparison_path() -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("comparison") / f"session_{stamp}.tsv"
