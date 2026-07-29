"""WASAPI loopback audio capture via pyaudiowpatch.

Captures system audio output using Windows WASAPI loopback.
pyaudiowpatch wraps the Windows Core Audio API directly,
which is the supported way to do loopback capture on Windows.
"""

import threading
import numpy as np
from typing import Optional, Callable
from .buffer import AudioBuffer


def list_loopback_devices() -> list[dict]:
    """List all WASAPI loopback devices available (for display only).
    
    Note: Device indices may change between PyAudio instances.
    For actual capture, use AudioCapture which discovers devices
    within its own session.
    """
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    devices = []
    try:
        for info in p.get_loopback_device_info_generator():
            devices.append({
                'name': info['name'],
                'channels': info['maxInputChannels'],
                'sample_rate': int(info['defaultSampleRate']),
            })
    finally:
        p.terminate()
    return devices


class AudioCapture:
    """Captures system audio via WASAPI loopback.
    
    Discovers the default loopback device within its own PyAudio session,
    ensuring correct device indices. Audio is captured at native rate,
    downmixed to mono, and resampled to 16kHz.
    
    Usage:
        capture = AudioCapture()
        capture.start(on_audio=my_callback)
        ...
        capture.stop()
    """

    def __init__(self, sample_rate: int = 16000, blocksize: int = 1024):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.buffer = AudioBuffer(sample_rate=sample_rate, max_seconds=10.0)
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._on_audio: Optional[Callable[[np.ndarray], None]] = None

    def start(self, on_audio: Optional[Callable[[np.ndarray], None]] = None) -> None:
        """Start capturing system audio via WASAPI loopback.
        
        Args:
            on_audio: Callback receiving each 16kHz mono audio chunk (numpy array).
                      Called from worker thread — must be non-blocking.
        """
        if self._running:
            return
        self._on_audio = on_audio
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Worker thread: discovers loopback device and captures audio."""
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        try:
            # Discover the default loopback device within THIS session
            loopback = p.get_default_wasapi_loopback()
            native_rate = int(loopback['defaultSampleRate'])
            device_index = loopback['index']
            device_name = loopback.get('name', 'unknown')

            print(f"[audio] Loopback device: [{device_index}] {device_name} @ {native_rate}Hz")
            print(f"[audio] Capture thread started (blocksize={self.blocksize}, target_rate={self.sample_rate})")

            # Verify this is really a loopback device
            device_info = p.get_device_info_by_index(device_index)
            if not device_info.get('isLoopbackDevice'):
                print(f"[audio] Warning: device {device_index} is not a loopback device")
                return

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=2,
                rate=native_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.blocksize,
            )
            print("[audio] Stream opened, listening for audio...")

            while self._running:
                try:
                    raw = stream.read(self.blocksize, exception_on_overflow=False)
                    data = np.frombuffer(raw, dtype=np.float32)

                    if len(data) < 2:
                        continue

                    # Reshape stereo interleaved → mono
                    data = data.reshape(-1, 2).mean(axis=1).astype(np.float32)

                    # Resample native rate → 16kHz (simple decimation)
                    ratio = native_rate / self.sample_rate
                    indices = np.arange(0, len(data), ratio).astype(int)
                    indices = indices[indices < len(data)]
                    resampled = data[indices]

                    self.buffer.write(resampled)
                    if self._on_audio:
                        self._on_audio(resampled)

                except Exception:
                    pass  # benign overflow errors during loopback capture

            stream.close()
        except LookupError:
            print("[audio] No WASAPI loopback device found")
        except Exception as e:
            print(f"[audio] Capture error: {e}")
        finally:
            p.terminate()
            print("[audio] Capture thread stopped")

    def stop(self) -> None:
        """Stop capturing audio."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None
        self._on_audio = None

    @property
    def is_running(self) -> bool:
        return self._running
