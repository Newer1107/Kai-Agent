from __future__ import annotations

import re
from dataclasses import dataclass

from desktop_perception import DesktopPerception
from os_tools import OSToolLayer
from safety_policy import SafetyPolicy


@dataclass(frozen=True)
class DesktopRouteResult:
    handled: bool
    success: bool
    method: str
    output: str
    error: str | None = None


class ActionRouter:
    """Desktop-first router: tools -> desktop perception fallback decision."""

    def __init__(self) -> None:
        self.tools = OSToolLayer()
        self.policy = SafetyPolicy()
        self.perception = DesktopPerception()

    def route(self, command: str) -> DesktopRouteResult:
        raw = (command or "").strip()
        if not raw:
            return DesktopRouteResult(False, False, "desktop_router", "", "Empty command")

        lowered = raw.lower()

        # open app
        open_match = re.match(r"^(open|launch|start)\s+(.+)$", lowered)
        if open_match:
            app_name = open_match.group(2).strip()
            decision = self.policy.evaluate("open_app", raw)
            if not decision.allowed:
                return DesktopRouteResult(True, False, "safety", "", decision.reason)
            result = self.tools.open_app(app_name)
            return DesktopRouteResult(True, result.success, "os_tools.open_app", result.output, result.error)

        close_match = re.match(r"^(close|quit|exit)\s+(.+)$", lowered)
        if close_match:
            app_name = close_match.group(2).strip()
            decision = self.policy.evaluate("close_app", raw)
            if not decision.allowed:
                return DesktopRouteResult(True, False, "safety", "", decision.reason)
            result = self.tools.close_app(app_name)
            return DesktopRouteResult(True, result.success, "os_tools.close_app", result.output, result.error)

        switch_match = re.match(r"^(switch to|focus)\s+(.+)$", lowered)
        if switch_match:
            window_name = switch_match.group(2).strip()
            result = self.tools.switch_window(window_name)
            return DesktopRouteResult(True, result.success, "os_tools.switch_window", result.output, result.error)

        settings_match = re.match(r"^(open\s+)?settings(\s+(.+))?$", lowered)
        if settings_match:
            page = (settings_match.group(3) or "").strip().replace(" ", "-")
            result = self.tools.open_settings(page)
            return DesktopRouteResult(True, result.success, "os_tools.open_settings", result.output, result.error)

        create_folder_match = re.match(r"^(create|make)\s+folder\s+(.+)$", raw, flags=re.IGNORECASE)
        if create_folder_match:
            path = create_folder_match.group(2).strip()
            result = self.tools.create_folder(path)
            return DesktopRouteResult(True, result.success, "os_tools.create_folder", result.output, result.error)

        list_dir_match = re.match(r"^(list|show)\s+(files|directory|folder)\s+(.+)$", raw, flags=re.IGNORECASE)
        if list_dir_match:
            path = list_dir_match.group(3).strip()
            result = self.tools.list_directory(path)
            return DesktopRouteResult(True, result.success, "os_tools.list_directory", result.output, result.error)

        move_match = re.match(r"^move\s+file\s+(.+)\s+to\s+(.+)$", raw, flags=re.IGNORECASE)
        if move_match:
            source = move_match.group(1).strip()
            dest = move_match.group(2).strip()
            decision = self.policy.evaluate("move_file", raw)
            if not decision.allowed:
                return DesktopRouteResult(True, False, "safety", "", decision.reason)
            result = self.tools.move_file(source, dest)
            return DesktopRouteResult(True, result.success, "os_tools.move_file", result.output, result.error)

        volume_match = re.match(r"^(set\s+)?volume\s+(to\s+)?(\d{1,3})", lowered)
        if volume_match:
            level = int(volume_match.group(3))
            result = self.tools.adjust_volume(level)
            return DesktopRouteResult(True, result.success, "os_tools.adjust_volume", result.output, result.error)

        if lowered in {"minimize window", "minimize"}:
            result = self.tools.minimize_window()
            return DesktopRouteResult(True, result.success, "os_tools.minimize_window", result.output, result.error)

        if lowered in {"maximize window", "maximize"}:
            result = self.tools.maximize_window()
            return DesktopRouteResult(True, result.success, "os_tools.maximize_window", result.output, result.error)

        # Not handled by direct tools; expose desktop perception for upstream fallback.
        state = self.perception.capture()
        return DesktopRouteResult(
            handled=False,
            success=False,
            method="desktop_perception",
            output=(
                f"Desktop context ready: app={state.active_app}, title={state.window_title}, "
                f"ocr_chars={len(state.detected_text)}"
            ),
            error=None,
        )
