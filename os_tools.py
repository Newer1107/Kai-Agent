from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import keyboard
except Exception:  # pragma: no cover
    keyboard = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None

try:
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None

try:
    import pygetwindow as gw
except Exception:  # pragma: no cover
    gw = None

try:
    from pywinauto import Application
except Exception:  # pragma: no cover
    Application = None


@dataclass(frozen=True)
class ToolResult:
    success: bool
    output: str
    error: str | None = None


class OSToolLayer:
    def __init__(self) -> None:
        self._app_aliases = {
            "chrome": ["chrome.exe", "start chrome"],
            "notepad": ["notepad.exe"],
            "vscode": ["code"],
            "spotify": ["spotify.exe"],
            "explorer": ["explorer.exe"],
            "excel": ["excel.exe"],
            "settings": ["start ms-settings:"],
        }

    def open_app(self, app_name: str) -> ToolResult:
        key = (app_name or "").strip().lower()
        commands = self._app_aliases.get(key, [key])
        for cmd in commands:
            try:
                if cmd.startswith("start "):
                    subprocess.Popen(cmd, shell=True)
                else:
                    subprocess.Popen(cmd, shell=True)
                return ToolResult(success=True, output=f"Opened {app_name}.")
            except Exception:
                continue
        return ToolResult(success=False, output="", error=f"Could not open app: {app_name}")

    def close_app(self, app_name: str) -> ToolResult:
        if psutil is None:
            return ToolResult(success=False, output="", error="psutil is unavailable")

        target = (app_name or "").lower().strip()
        killed = 0
        for proc in psutil.process_iter(attrs=["name"]):
            name = str(proc.info.get("name", "")).lower()
            if target in name:
                try:
                    proc.terminate()
                    killed += 1
                except Exception:
                    pass
        if killed > 0:
            return ToolResult(success=True, output=f"Closed {killed} process(es) for {app_name}.")
        return ToolResult(success=False, output="", error=f"No running process matched {app_name}")

    def switch_window(self, window_name: str) -> ToolResult:
        target = (window_name or "").lower().strip()
        if gw is not None:
            try:
                wins = gw.getAllTitles()
                for title in wins:
                    if title and target in title.lower():
                        candidate = gw.getWindowsWithTitle(title)
                        if candidate:
                            candidate[0].activate()
                            return ToolResult(success=True, output=f"Switched to window: {title}")
            except Exception:
                pass

        if Application is not None:
            try:
                app = Application(backend="uia").connect(title_re=f".*{window_name}.*")
                wnd = app.top_window()
                wnd.set_focus()
                return ToolResult(success=True, output=f"Switched to window: {window_name}")
            except Exception:
                pass

        return ToolResult(success=False, output="", error=f"Window not found: {window_name}")

    def search_start_menu(self, query: str) -> ToolResult:
        if keyboard is None:
            return ToolResult(success=False, output="", error="keyboard module unavailable")
        try:
            keyboard.send("windows")
            keyboard.write(query)
            return ToolResult(success=True, output=f"Searched start menu for: {query}")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

    def move_file(self, source: str, destination: str) -> ToolResult:
        try:
            src = Path(source).expanduser().resolve()
            dst = Path(destination).expanduser().resolve()
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.replace(dst)
            return ToolResult(success=True, output=f"Moved {src} to {dst}")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

    def create_folder(self, path: str) -> ToolResult:
        try:
            Path(path).expanduser().mkdir(parents=True, exist_ok=True)
            return ToolResult(success=True, output=f"Created folder: {path}")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

    def list_directory(self, path: str) -> ToolResult:
        try:
            base = Path(path).expanduser()
            items = [p.name for p in base.iterdir()]
            return ToolResult(success=True, output="\n".join(items[:200]))
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

    def open_settings(self, page: str = "") -> ToolResult:
        try:
            uri = "ms-settings:" + (page or "")
            os.startfile(uri)
            return ToolResult(success=True, output=f"Opened settings page: {uri}")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

    def adjust_volume(self, level: int) -> ToolResult:
        if keyboard is None:
            return ToolResult(success=False, output="", error="keyboard module unavailable")
        bounded = max(0, min(100, int(level)))
        try:
            for _ in range(60):
                keyboard.send("volume down")
            for _ in range(max(0, bounded // 2)):
                keyboard.send("volume up")
            return ToolResult(success=True, output=f"Adjusted volume to approximately {bounded}%")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

    def minimize_window(self) -> ToolResult:
        if pyautogui is None:
            return ToolResult(success=False, output="", error="pyautogui unavailable")
        try:
            pyautogui.hotkey("win", "down")
            return ToolResult(success=True, output="Minimized current window.")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

    def maximize_window(self) -> ToolResult:
        if pyautogui is None:
            return ToolResult(success=False, output="", error="pyautogui unavailable")
        try:
            pyautogui.hotkey("win", "up")
            return ToolResult(success=True, output="Maximized current window.")
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))
