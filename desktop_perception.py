from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pyautogui
except Exception:  # pragma: no cover
    pyautogui = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

from window_context import as_dict, get_active_window_context


@dataclass(frozen=True)
class DesktopPerceptionState:
    active_app: str
    window_title: str
    ui_elements: list[dict[str, Any]]
    accessibility_tree: dict[str, Any]
    screenshot: str | None
    detected_text: str


class DesktopPerception:
    def __init__(self) -> None:
        self._capture_dir = Path("debug")
        self._capture_dir.mkdir(parents=True, exist_ok=True)

    def capture(self) -> DesktopPerceptionState:
        ctx = get_active_window_context()
        ctx_data = as_dict(ctx)

        screenshot_path: str | None = None
        detected_text = ""
        if pyautogui is not None:
            try:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                out = self._capture_dir / f"desktop_state_{stamp}.png"
                shot = pyautogui.screenshot()
                shot.save(out)
                screenshot_path = str(out)
                if pytesseract is not None:
                    try:
                        detected_text = (pytesseract.image_to_string(shot) or "").strip()[:4000]
                    except Exception:
                        detected_text = ""
            except Exception:
                screenshot_path = None

        return DesktopPerceptionState(
            active_app=str(ctx_data.get("app_context", "desktop")),
            window_title=str(ctx_data.get("window_title", "unknown")),
            ui_elements=[],
            accessibility_tree={"window": ctx_data},
            screenshot=screenshot_path,
            detected_text=detected_text,
        )

    def to_dict(self, state: DesktopPerceptionState) -> dict[str, Any]:
        return {
            "active_app": state.active_app,
            "window_title": state.window_title,
            "ui_elements": state.ui_elements,
            "accessibility_tree": state.accessibility_tree,
            "screenshot": state.screenshot,
            "detected_text": state.detected_text,
        }
