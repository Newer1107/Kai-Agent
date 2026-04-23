from __future__ import annotations

import threading
from typing import Optional

from schema import UIAction


current_goal: Optional[str] = None
last_action: Optional[UIAction] = None
autopilot_enabled: bool = False
max_steps: int = 5
current_step: int = 0

_STATE_LOCK = threading.Lock()


def set_goal(goal: str) -> None:
    """Set the current in-memory goal; blank goals are treated as no goal."""
    global current_goal
    cleaned = goal.strip()
    with _STATE_LOCK:
        current_goal = cleaned if cleaned else None


def clear_goal() -> None:
    """Clear the current in-memory goal."""
    global current_goal
    with _STATE_LOCK:
        current_goal = None


def get_goal() -> Optional[str]:
    """Return the current in-memory goal, if any."""
    with _STATE_LOCK:
        return current_goal


def set_last_action(action: Optional[UIAction]) -> None:
    """Store the last approved step in memory for contextual reasoning."""
    global last_action
    with _STATE_LOCK:
        last_action = action


def get_last_action() -> Optional[UIAction]:
    """Return the last approved action, if any."""
    with _STATE_LOCK:
        return last_action


def enable_autopilot(step_limit: int | None = None) -> None:
    """Enable autopilot and reset step counter."""
    global autopilot_enabled, current_step, max_steps
    with _STATE_LOCK:
        autopilot_enabled = True
        current_step = 0
        if step_limit is not None:
            max_steps = max(1, int(step_limit))


def disable_autopilot() -> None:
    """Disable autopilot and reset step counter."""
    global autopilot_enabled, current_step
    with _STATE_LOCK:
        autopilot_enabled = False
        current_step = 0


def is_autopilot_enabled() -> bool:
    with _STATE_LOCK:
        return autopilot_enabled


def get_max_steps() -> int:
    with _STATE_LOCK:
        return max_steps


def set_max_steps(value: int) -> None:
    global max_steps
    with _STATE_LOCK:
        max_steps = max(1, int(value))


def get_current_step() -> int:
    with _STATE_LOCK:
        return current_step


def set_current_step(value: int) -> None:
    global current_step
    with _STATE_LOCK:
        current_step = max(0, int(value))


def clear_state() -> None:
    """Clear goal, last action, and autopilot runtime state."""
    global current_goal, last_action, autopilot_enabled, current_step
    with _STATE_LOCK:
        current_goal = None
        last_action = None
        autopilot_enabled = False
        current_step = 0
