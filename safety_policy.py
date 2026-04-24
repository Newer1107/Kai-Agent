from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    requires_approval: bool
    reason: str


class SafetyPolicy:
    def __init__(self) -> None:
        self._blocked_keywords = {
            "registry",
            "regedit",
            "system32",
            "format",
            "wipe",
            "erase",
            "delete all",
            "shutdown",
            "restart",
            "power off",
            "uninstall",
        }
        self._approval_actions = {
            "delete_file",
            "remove_file",
            "shutdown",
            "restart",
            "install_app",
            "uninstall_app",
            "registry_edit",
        }

    def evaluate(self, action_name: str, action_text: str) -> SafetyDecision:
        text = (action_text or "").lower()
        if any(keyword in text for keyword in self._blocked_keywords):
            return SafetyDecision(
                allowed=False,
                requires_approval=False,
                reason="Blocked by safety policy: dangerous system action detected.",
            )

        if action_name in self._approval_actions:
            return SafetyDecision(
                allowed=True,
                requires_approval=True,
                reason="Action requires explicit user approval.",
            )

        return SafetyDecision(allowed=True, requires_approval=False, reason="Allowed by policy.")
