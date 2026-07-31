"""Hardware capability probe → latency-friendly Whisper defaults.

The MX450-class (≤4 GB VRAM) laptop GPUs choke on beam>1 and struggle with
`small`. Measuring once at startup and picking a safe profile is the
difference between "3 sentences behind" and "near real-time".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CapabilityProfile:
    """Recommended runtime knobs for this machine."""

    device: str                 # "cuda" or "cpu"
    gpu_name: str
    vram_gb: float
    model_size: str             # tiny / base / small / ...
    beam_size: int
    compute_type: str
    cpu_threads: int
    # Latency-tuned VAD / processor knobs
    min_silence_ms: int
    max_speech_ms: int
    partial_interval_ms: int
    max_queued: int
    reason: str

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "model_size": self.model_size,
            "beam_size": self.beam_size,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "min_silence_ms": self.min_silence_ms,
            "max_speech_ms": self.max_speech_ms,
            "partial_interval_ms": self.partial_interval_ms,
            "max_queued": self.max_queued,
            "reason": self.reason,
        }


def _probe_cuda() -> tuple[bool, str, float]:
    """Return (available, name, vram_gb). Never raises."""
    try:
        import ctranslate2 as ct2
        if ct2.get_cuda_device_count() <= 0:
            return False, "", 0.0
    except Exception:
        pass
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "", 0.0
        props = torch.cuda.get_device_properties(0)
        name = props.name or "CUDA GPU"
        vram = float(props.total_memory) / (1024 ** 3)
        return True, name, vram
    except Exception as exc:
        logger.debug("CUDA probe failed: %s", exc)
        return False, "", 0.0


def _cpu_threads() -> int:
    try:
        import os
        n = os.cpu_count() or 4
    except Exception:
        n = 4
    # Leave headroom for the UI + VAD + audio capture.
    return max(2, min(6, n // 2))


def detect_capability(prefer_device: Optional[str] = None) -> CapabilityProfile:
    """Pick a near-real-time profile for this machine.

    Profiles bias hard toward low latency. Quality can be raised manually
    in Settings; the goal here is "subtitles that keep up with speech".
    """
    cuda_ok, gpu_name, vram = _probe_cuda()
    threads = _cpu_threads()
    want_cuda = (prefer_device or "cuda").lower() != "cpu"

    if want_cuda and cuda_ok and vram >= 0.5:
        # Entry laptop GPUs (MX450, GTX 1650 4GB, etc.): beam=1 is mandatory.
        # Measured: base/beam1 ≈ 0.25 RTF; base/beam2 ≈ 3.0 RTF on MX450.
        if vram < 3.5:
            # Step 3 bench on MX450: small/beam1 finals ~0.30 RTF vs base;
            # keep config size=small — do not auto-downgrade to base.
            return CapabilityProfile(
                device="cuda",
                gpu_name=gpu_name,
                vram_gb=round(vram, 2),
                model_size="small",
                beam_size=1,
                compute_type="int8_float16",
                cpu_threads=threads,
                min_silence_ms=400,
                max_speech_ms=6000,
                partial_interval_ms=700,
                max_queued=1,
                reason=(
                    f"{gpu_name} ({vram:.1f} GB) — low-VRAM profile: "
                    "small + beam 1 draft + 3–6s VAD batching"
                ),
            )
        if vram < 8.0:
            return CapabilityProfile(
                device="cuda",
                gpu_name=gpu_name,
                vram_gb=round(vram, 2),
                model_size="small",
                beam_size=1,
                compute_type="int8_float16",
                cpu_threads=threads,
                min_silence_ms=400,
                max_speech_ms=6000,
                partial_interval_ms=700,
                max_queued=2,
                reason=(
                    f"{gpu_name} ({vram:.1f} GB) — mid-VRAM profile: "
                    "small + beam 1 + 3–6s VAD batching"
                ),
            )
        return CapabilityProfile(
            device="cuda",
            gpu_name=gpu_name,
            vram_gb=round(vram, 2),
            model_size="small",
            beam_size=2,
            compute_type="float16",
            cpu_threads=threads,
            min_silence_ms=400,
            max_speech_ms=6000,
            partial_interval_ms=700,
            max_queued=2,
            reason=(
                f"{gpu_name} ({vram:.1f} GB) — high-VRAM profile: "
                "small + beam 2 + 3–6s VAD batching"
            ),
        )

    return CapabilityProfile(
        device="cpu",
        gpu_name=gpu_name or "CPU",
        vram_gb=0.0,
        model_size="tiny",
        beam_size=1,
        compute_type="int8",
        cpu_threads=threads,
        min_silence_ms=400,
        max_speech_ms=6000,
        partial_interval_ms=900,
        max_queued=1,
        reason="No usable CUDA — CPU profile: tiny + beam 1",
    )


def apply_profile_to_config(config: dict, profile: CapabilityProfile, force: bool = False) -> dict:
    """Merge the profile into config for near-real-time defaults.

    With ``latency.auto_tune: true`` (the default), latency-critical knobs
    (beam size, silence wait, max speech, live partials) are always aligned
    to the hardware profile. Heavier models the user explicitly chose are
    kept unless they are known-deadly on this GPU (e.g. beam≥2 on ≤4 GB).
    """
    model = config.setdefault("model", {})
    vad = config.setdefault("vad", {})
    latency = config.setdefault("latency", {})
    auto = force or bool(latency.get("auto_tune", True))

    # Always record what we detected so Settings / logs can show it.
    latency["profile"] = profile.as_dict()

    if force or not model.get("size"):
        model["size"] = profile.model_size
    if force or not model.get("device"):
        model["device"] = profile.device
    if force or not model.get("compute_type"):
        model["compute_type"] = profile.compute_type
    if force or not model.get("cpu_threads"):
        model["cpu_threads"] = profile.cpu_threads

    if auto:
        # Beam > 1 on low-VRAM GPUs is the #1 cause of multi-sentence lag.
        # Measured on MX450: base/beam1 ≈ 0.25 RTF, base/beam2 ≈ 3.0 RTF.
        current_beam = int(model.get("beam_size") or profile.beam_size)
        if force or current_beam > profile.beam_size:
            if current_beam != profile.beam_size:
                logger.info(
                    "Auto-tune: beam_size %s → %s (%s)",
                    current_beam, profile.beam_size, profile.gpu_name or profile.device,
                )
            model["beam_size"] = profile.beam_size
        elif not model.get("beam_size"):
            model["beam_size"] = profile.beam_size

        # Only downgrade models heavier than the profile default. User/config
        # size=small is kept on low-VRAM GPUs after Step 3 bench evidence.
        heavy = ("medium", "large-v3", "large", "large-v2", "large-v1")
        if force or (profile.vram_gb and profile.vram_gb < 3.5
                     and model.get("size") in heavy):
            if model.get("size") != profile.model_size:
                logger.info(
                    "Auto-tune: model %s → %s (%.1f GB VRAM)",
                    model.get("size"), profile.model_size, profile.vram_gb,
                )
            model["size"] = profile.model_size
        elif not model.get("size"):
            model["size"] = profile.model_size

        vad["min_silence_duration_ms"] = profile.min_silence_ms
        vad["max_speech_duration_ms"] = profile.max_speech_ms
        # Preserve batching knobs; only fill defaults when missing.
        vad.setdefault("merge_silence_ms", 400)
        vad.setdefault("target_min_speech_ms", 3000)
        if force or not vad.get("speech_pad_ms"):
            vad["speech_pad_ms"] = 250
        elif int(vad.get("speech_pad_ms", 250)) > 400:
            vad["speech_pad_ms"] = 250
        if force or int(vad.get("min_speech_duration_ms", 500)) > 600:
            vad["min_speech_duration_ms"] = 550
        elif force or not vad.get("min_speech_duration_ms"):
            vad["min_speech_duration_ms"] = 550

        latency["partial_interval_ms"] = profile.partial_interval_ms
        latency["max_queued"] = profile.max_queued
        latency["live_partials"] = latency.get("live_partials", True)
        latency["lag_governor"] = latency.get("lag_governor", True)
        latency["streaming"] = latency.get("streaming", True)
        latency["final_beam"] = int(latency.get("final_beam", 5))
        latency.setdefault("streaming_flush_chars", 80)
        latency.setdefault("streaming_flush_timeout_s", 6.0)
        latency["auto_tune"] = True

    return config
