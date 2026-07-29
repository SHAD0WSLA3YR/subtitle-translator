"""Optional Windows acrylic/blur backdrop for frameless overlays.

Uses the undocumented SetWindowCompositionAttribute API. Fails silently on
non-Windows or unsupported builds — callers fall back to a solid painted box.
"""

import ctypes
import logging
import sys
from ctypes import wintypes

logger = logging.getLogger(__name__)

_ACCENT_DISABLED = 0
_ACCENT_ENABLE_BLURBEHIND = 3
_ACCENT_ENABLE_ACRYLICBLURBEHIND = 4
_WCA_ACCENT_POLICY = 19


class _AccentPolicy(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_int),
        ("AccentFlags", ctypes.c_int),
        ("GradientColor", ctypes.c_uint),
        ("AnimationId", ctypes.c_int),
    ]


class _WinCompAttrData(ctypes.Structure):
    _fields_ = [
        ("Attribute", ctypes.c_int),
        ("Data", ctypes.POINTER(_AccentPolicy)),
        ("SizeOfData", ctypes.c_size_t),
    ]


def _set_composition(hwnd: int, state: int, gradient_color: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        user32 = ctypes.windll.user32
        set_attr = user32.SetWindowCompositionAttribute
    except (AttributeError, OSError):
        return False

    accent = _AccentPolicy(state, 2, gradient_color, 0)
    data = _WinCompAttrData(
        _WCA_ACCENT_POLICY,
        ctypes.pointer(accent),
        ctypes.sizeof(accent),
    )
    try:
        set_attr(wintypes.HWND(hwnd), ctypes.byref(data))
        return True
    except Exception as exc:
        logger.debug("Blur backdrop unavailable: %s", exc)
        return False


def enable_blur(hwnd: int, tint_rgb=(0, 0, 0), tint_alpha: int = 140) -> bool:
    """Turn on an acrylic blur backdrop behind the window.

    Args:
        hwnd: Native window handle.
        tint_rgb: Tint color painted over the blur.
        tint_alpha: 0-255 tint strength.

    Returns:
        True if the backdrop was applied.
    """
    r, g, b = tint_rgb
    # GradientColor is AABBGGRR
    color = (int(tint_alpha) << 24) | (int(b) << 16) | (int(g) << 8) | int(r)
    if _set_composition(hwnd, _ACCENT_ENABLE_ACRYLICBLURBEHIND, color):
        return True
    return _set_composition(hwnd, _ACCENT_ENABLE_BLURBEHIND, color)


def disable_blur(hwnd: int) -> bool:
    """Remove any blur backdrop from the window."""
    return _set_composition(hwnd, _ACCENT_DISABLED, 0)
