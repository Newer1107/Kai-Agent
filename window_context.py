from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


@dataclass(frozen=True)
class WindowContext:
    title: str
    process_name: str
    process_id: int | None


def _get_foreground_hwnd() -> int | None:
    try:
        return int(ctypes.windll.user32.GetForegroundWindow())
    except Exception:
        return None


def _get_window_title(hwnd: int) -> str:
    if hwnd <= 0:
        return ""
    try:
        length = int(ctypes.windll.user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return ""
        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
        return str(buff.value or "")
    except Exception:
        return ""


def _get_window_pid(hwnd: int) -> int | None:
    if hwnd <= 0:
        return None
    try:
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        value = int(pid.value)
        return value if value > 0 else None
    except Exception:
        return None


def _get_process_name(pid: int | None) -> str:
    if pid is None or psutil is None:
        return "unknown"
    try:
        proc = psutil.Process(pid)
        return proc.name().lower()
    except Exception:
        return "unknown"


def get_active_window_context() -> WindowContext:
    hwnd = _get_foreground_hwnd() or 0
    title = _get_window_title(hwnd)
    pid = _get_window_pid(hwnd)
    process_name = _get_process_name(pid)
    return WindowContext(title=title or "unknown", process_name=process_name, process_id=pid)


def classify_app_context(process_name: str, title: str) -> str:
    proc = (process_name or "").lower()
    ttl = (title or "").lower()

    if any(token in proc for token in ["chrome", "msedge", "firefox", "brave", "opera"]):
        return "browser"
    if "explorer" in proc or "file explorer" in ttl:
        return "file_explorer"
    if "excel" in proc:
        return "excel"
    if "code" in proc or "visual studio code" in ttl:
        return "vscode"
    if "spotify" in proc:
        return "spotify"
    if "notepad" in proc:
        return "notepad"
    if "settings" in ttl:
        return "settings"
    return "desktop"


def as_dict(ctx: WindowContext) -> dict[str, Any]:
    return {
        "window_title": ctx.title,
        "process_name": ctx.process_name,
        "process_id": ctx.process_id,
        "app_context": classify_app_context(ctx.process_name, ctx.title),
    }
