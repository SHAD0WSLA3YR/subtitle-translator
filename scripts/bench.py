"""Measure pipeline cost: VAD load on the audio thread and Whisper per clause.

Usage:  python scripts/bench.py [--clause-seconds 5] [--skip-whisper]

Reports real-time factor (RTF). RTF < 1.0 means the stage keeps up with
live playback; the VAD number is the always-on background cost.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def bench_vad(seconds: float, silence_floor: float) -> dict:
    from src.audio.vad import VADProcessor

    vad = VADProcessor(silence_floor=silence_floor, on_clause=lambda a: None)
    sample_rate = 16000
    total = int(sample_rate * seconds)
    chunk = 512

    rng = np.random.default_rng(0)
    silence = np.zeros(total, dtype=np.float32)
    t = np.arange(total) / sample_rate
    speech = (
        0.25 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(total)
    ).astype(np.float32)

    results = {}
    for name, signal in (("silence", silence), ("speech-like", speech)):
        vad.reset()
        start = time.perf_counter()
        for i in range(0, total, chunk):
            vad.process_chunk(signal[i : i + chunk])
        elapsed = time.perf_counter() - start
        results[name] = elapsed / seconds
    return results


def bench_whisper(clause_seconds: float, beams, model_size: str, device: str) -> dict:
    from src.stt.whisper_stt import WhisperSTT

    sample_rate = 16000
    t = np.arange(int(sample_rate * clause_seconds)) / sample_rate
    audio = (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    results = {}
    for beam in beams:
        stt = WhisperSTT(model_size=model_size, device=device, beam_size=beam)
        stt.load()
        stt.translate_to_english(audio)  # warm up kernels

        start = time.perf_counter()
        stt.translate_to_english(audio)
        one_pass = time.perf_counter() - start

        start = time.perf_counter()
        stt.process(audio)
        two_pass = time.perf_counter() - start

        results[beam] = {
            "1 pass (translate only)": one_pass,
            "2 passes (with --compare)": two_pass,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clause-seconds", type=float, default=5.0)
    parser.add_argument("--vad-seconds", type=float, default=30.0)
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-whisper", action="store_true")
    args = parser.parse_args()

    print(f"VAD over {args.vad_seconds:g}s of audio (always-on cost)")
    for floor, label in ((0.0, "no silence gate"), (0.004, "silence gate on")):
        for name, rtf in bench_vad(args.vad_seconds, floor).items():
            print(f"  {label:>16} | {name:<12} RTF {rtf:.4f} ({rtf * 100:.2f}% of one core)")

    if args.skip_whisper:
        return 0

    print(f"\nWhisper on a {args.clause_seconds:g}s clause ({args.model}, {args.device})")
    for beam, timings in bench_whisper(
        args.clause_seconds, (2, 3, 5), args.model, args.device
    ).items():
        for label, secs in timings.items():
            rtf = secs / args.clause_seconds
            print(f"  beam {beam} | {label:<26} {secs:5.2f}s  RTF {rtf:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
