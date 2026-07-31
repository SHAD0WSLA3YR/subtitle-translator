"""Download + convert facebook/nllb-200-distilled-600M to CT2 int8.

Usage:
  .\\.venv\\Scripts\\python.exe scripts\\install_nllb_ct2.py

Output (default):
  %USERPROFILE%\\.subtitle_translator\\nllb-600m-ct2
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path.home() / ".subtitle_translator" / "nllb-600m-ct2"
HF_MODEL = "facebook/nllb-200-distilled-600M"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out: Path = args.out_dir
    if out.is_dir() and any(out.iterdir()) and not args.force:
        print(f"Already installed: {out}")
        print("Pass --force to rebuild.")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.force and out.exists():
        shutil.rmtree(out)

    cmd = [
        sys.executable,
        "-m",
        "ctranslate2.converters.transformers",
        "--model",
        HF_MODEL,
        "--output_dir",
        str(out),
        "--quantization",
        args.quantization,
        "--force",
    ]
    # Prefer CLI if present
    ct2_cli = shutil.which("ct2-transformers-converter")
    if ct2_cli:
        cmd = [
            ct2_cli,
            "--model",
            HF_MODEL,
            "--output_dir",
            str(out),
            "--quantization",
            args.quantization,
            "--force",
        ]

    print("Converting", HF_MODEL, "->", out)
    print("Command:", " ".join(cmd))
    print("This downloads ~1GB and may take several minutes...")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        # Fallback: older entry point module path
        cmd2 = [
            sys.executable,
            "-c",
            (
                "from ctranslate2.converters.transformers import TransformersConverter as C; "
                f"C('{HF_MODEL}').convert(str(r'{out}'), quantization='{args.quantization}', force=True)"
            ),
        ]
        print("Retrying via TransformersConverter API...")
        proc = subprocess.run(cmd2, cwd=str(ROOT))
    if proc.returncode != 0:
        print("FAILED — install transformers sentencepiece and retry", file=sys.stderr)
        return proc.returncode or 1
    print("OK:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
