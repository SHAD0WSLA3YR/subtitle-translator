"""Offline ASR bench: WAV × config matrix → RTF / VRAM / clause stats / JA dump.

Never uses Whisper task=translate — only VAD + transcribe_source.

Usage:
  python scripts/bench_asr.py --wav path\\to\\clip.wav --matrix mx450
  python scripts/bench_asr.py --wav clip.wav --matrix mx450 --out-dir .bench_out

Run on a ~60s clip of your real content before changing the default model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class MatrixRow:
    id: str
    model_size: str
    beam_size: int
    compute_type: str
    use_vad_batching: bool
    note: str


MX450_MATRIX: List[MatrixRow] = [
    MatrixRow("A_base_legacy_vad", "base", 1, "int8_float16", False, "legacy short VAD"),
    MatrixRow("B_base_batched_vad", "base", 1, "int8_float16", True, "3–6s VAD batching"),
    MatrixRow("C_small_batched", "small", 3, "int8_float16", True, "quality candidate"),
]


def _load_wav_mono_16k(path: Path) -> np.ndarray:
    try:
        import soundfile as sf

        audio, sr = sf.read(str(path), always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)
    except ImportError:
        import wave

        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
        if sw == 2:
            pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            raise RuntimeError(f"Unsupported sample width {sw}; install soundfile")
        if nch > 1:
            pcm = pcm.reshape(-1, nch).mean(axis=1)
        audio = pcm
    if sr != 16000:
        # Lightweight linear resample (avoid hard dep on librosa).
        n_out = int(len(audio) * 16000 / sr)
        x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        audio = np.interp(x_new, x_old, audio).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 1.0:
        audio = audio / peak
    return audio


def _peak_vram_mb() -> Optional[float]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception:
        return None


def _reset_vram_peak() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _collect_clauses(
    audio: np.ndarray,
    *,
    batched: bool,
    synthetic_vad: bool = False,
) -> List[np.ndarray]:
    from src.audio.vad import VADProcessor

    class _AlwaysSpeech:
        def __call__(self, tensor, sample_rate):
            class _P:
                def item(self_inner):
                    return 0.95

            return _P()

        def reset_states(self):
            pass

    clauses: List[np.ndarray] = []
    kwargs = {"on_clause": clauses.append}
    if synthetic_vad:
        kwargs["_vad_model"] = _AlwaysSpeech()
    if batched:
        vad = VADProcessor(
            min_silence_duration_ms=400,
            merge_silence_ms=400,
            target_min_speech_ms=3000,
            speech_pad_ms=250,
            max_speech_duration_ms=6000,
            **kwargs,
        )
    else:
        # Pathological pre-fix scrap regime (matches ~0.3–1s finals from logs),
        # not the later 950ms quality bump — so A/B contrast is meaningful.
        vad = VADProcessor(
            min_speech_duration_ms=150,
            min_silence_duration_ms=250,
            merge_silence_ms=0,
            target_min_speech_ms=0,
            speech_pad_ms=0,
            max_speech_duration_ms=5500,
            **kwargs,
        )
    chunk = 1024
    for i in range(0, len(audio), chunk):
        vad.process_chunk(audio[i : i + chunk])
    # Flush remaining speech by feeding silence.
    vad.process_chunk(np.zeros(16000, dtype=np.float32))
    return clauses


def _run_row(
    row: MatrixRow,
    audio: np.ndarray,
    out_dir: Path,
    device: str,
) -> dict:
    from src.stt.whisper_stt import WhisperSTT

    _reset_vram_peak()
    stt = WhisperSTT(
        model_size=row.model_size,
        device=device,
        compute_type=row.compute_type,
        language="ja",
        beam_size=row.beam_size,
    )
    stt.load()
    stt.lock_language("ja")

    clauses = _collect_clauses(
        audio, batched=row.use_vad_batching, synthetic_vad=False
    )
    min_final = 1.2
    kept = [c for c in clauses if len(c) / 16000.0 >= min_final]
    durations = [len(c) / 16000.0 for c in kept]

    infer_s = 0.0
    texts: List[str] = []
    for clause in kept:
        t0 = time.perf_counter()
        text = stt.transcribe_source(clause, lang_hint="ja") or ""
        infer_s += time.perf_counter() - t0
        texts.append(text.strip())

    audio_dur = len(audio) / 16000.0
    rtf = infer_s / audio_dur if audio_dur > 0 else 0.0
    peak = _peak_vram_mb()

    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / f"{row.id}.ja.txt"
    stats_path = out_dir / f"{row.id}.json"
    transcript_path.write_text("\n".join(texts) + ("\n" if texts else ""), encoding="utf-8")

    stats = {
        "config": row.id,
        "note": row.note,
        "model": row.model_size,
        "beam": row.beam_size,
        "batched_vad": row.use_vad_batching,
        "clause_count": len(kept),
        "clause_count_raw": len(clauses),
        "median_clause_s": statistics.median(durations) if durations else 0.0,
        "mean_clause_s": statistics.mean(durations) if durations else 0.0,
        "min_clause_s": min(durations) if durations else 0.0,
        "max_clause_s": max(durations) if durations else 0.0,
        "rtf": rtf,
        "infer_seconds": infer_s,
        "audio_seconds": audio_dur,
        "peak_vram_mb": peak,
        "transcript_path": str(transcript_path),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    del stt
    _reset_vram_peak()
    return stats


def _print_table(rows: List[dict]) -> None:
    headers = (
        "config",
        "clause_count",
        "median_clause_s",
        "rtf",
        "peak_vram_mb",
        "transcript_path",
    )
    print(" | ".join(headers))
    print(" | ".join("---" for _ in headers))
    for r in rows:
        vram = r.get("peak_vram_mb")
        vram_s = f"{vram:.0f}" if isinstance(vram, (int, float)) else "n/a"
        print(
            " | ".join(
                [
                    str(r["config"]),
                    str(r["clause_count"]),
                    f"{r['median_clause_s']:.2f}",
                    f"{r['rtf']:.3f}",
                    vram_s,
                    str(r["transcript_path"]),
                ]
            )
        )


def _run_vad_only_row(
    row: MatrixRow, audio: np.ndarray, out_dir: Path, *, synthetic_vad: bool
) -> dict:
    """Clause-length proof without Whisper — proves G001 batching on audio shape."""
    clauses = _collect_clauses(
        audio, batched=row.use_vad_batching, synthetic_vad=synthetic_vad
    )
    min_final = 1.2
    all_durs = [len(c) / 16000.0 for c in clauses]
    kept = [c for c in clauses if len(c) / 16000.0 >= min_final]
    durations = [len(c) / 16000.0 for c in kept]
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {
        "config": row.id + "_vad_only",
        "note": row.note + " (VAD-only)",
        "model": "none",
        "beam": 0,
        "batched_vad": row.use_vad_batching,
        "clause_count": len(kept),
        "clause_count_raw": len(clauses),
        "median_clause_s": statistics.median(durations) if durations else (
            statistics.median(all_durs) if all_durs else 0.0
        ),
        "median_clause_s_raw": statistics.median(all_durs) if all_durs else 0.0,
        "mean_clause_s": statistics.mean(durations) if durations else 0.0,
        "min_clause_s": min(all_durs) if all_durs else 0.0,
        "max_clause_s": max(all_durs) if all_durs else 0.0,
        "rtf": 0.0,
        "infer_seconds": 0.0,
        "audio_seconds": len(audio) / 16000.0,
        "peak_vram_mb": None,
        "transcript_path": "(vad-only — no ASR)",
    }
    (out_dir / f"{row.id}_vad_only.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, type=Path, help="16k mono preferred; other rates resampled")
    parser.add_argument("--matrix", default="mx450", choices=["mx450"])
    parser.add_argument("--out-dir", type=Path, default=Path(".bench_out"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--vad-only",
        action="store_true",
        help="Only measure VAD clause lengths (no Whisper) — use to prove batching before ASR quality",
    )
    parser.add_argument(
        "--synthetic-vad",
        action="store_true",
        help="Amplitude-gated speech detector (for synthetic fixtures Silero rejects). Real clips: omit this.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of matrix ids (e.g. B_base_batched_vad)",
    )
    args = parser.parse_args()

    if not args.wav.is_file():
        print(f"WAV not found: {args.wav}", file=sys.stderr)
        return 2

    audio = _load_wav_mono_16k(args.wav)
    print(f"Loaded {args.wav} — {len(audio) / 16000.0:.1f}s @ 16kHz")

    matrix = MX450_MATRIX
    if args.only:
        wanted = set(args.only)
        matrix = [r for r in matrix if r.id in wanted]
        if not matrix:
            print("No matrix rows matched --only", file=sys.stderr)
            return 2

    results: List[dict] = []
    for row in matrix:
        print(f"\n=== {row.id}: {row.note} ===")
        try:
            if args.vad_only:
                stats = _run_vad_only_row(
                    row, audio, args.out_dir, synthetic_vad=args.synthetic_vad
                )
            else:
                stats = _run_row(row, audio, args.out_dir, args.device)
            results.append(stats)
            print(
                f"clauses={stats['clause_count']} median={stats['median_clause_s']:.2f}s "
                f"rtf={stats['rtf']:.3f}"
            )
        except Exception as exc:
            print(f"FAILED {row.id}: {exc}", file=sys.stderr)
            results.append(
                {
                    "config": row.id,
                    "clause_count": 0,
                    "median_clause_s": 0.0,
                    "rtf": 0.0,
                    "peak_vram_mb": None,
                    "transcript_path": f"FAILED: {exc}",
                }
            )

    print("\n## Summary")
    _print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
