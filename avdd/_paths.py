"""
Windows long-path workaround.

Some compiled dependencies of Librosa (numba's `_box.pyd`, plus a few DLLs)
cannot be loaded when the absolute path of `site-packages` gets too long,
because Windows' classic 260-character MAX_PATH limit still applies to the
DLL search performed by the loader. The symptom is:

    ImportError: DLL load failed while importing _box:
    The filename or extension is too long.

On a normal installation (``C:\\Users\\you\\avdd``) this never happens. When the
project lives in a very deep folder we rewrite the import paths to their
Windows 8.3 *short* form (e.g. ``C:\\Users\\ABHILA~1\\...``), which is the same
directory under a shorter name, so the loader is happy again.

This is a no-op on Linux, macOS, and on any Windows path that is already short.
"""

from __future__ import annotations

import os
import sys

#: Only rewrite entries longer than this; shorter paths already load fine.
_MAX_SAFE_LEN = 120


def shorten_windows_sys_path(max_len: int = _MAX_SAFE_LEN) -> list[str]:
    """Replace long ``sys.path`` entries with their 8.3 short form (Windows only)."""
    if os.name != "nt":
        return []

    try:
        import ctypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        get_short.restype = ctypes.c_uint32
    except Exception:
        return []

    changed: list[str] = []
    for i, entry in enumerate(list(sys.path)):
        if not entry or len(entry) <= max_len or not os.path.isdir(entry):
            continue
        buf = ctypes.create_unicode_buffer(1024)
        n = get_short(entry, buf, 1024)
        if not n or n >= 1024:
            continue
        short = buf.value
        if short and len(short) < len(entry) and os.path.isdir(short):
            sys.path[i] = short
            changed.append(short)
    return changed
