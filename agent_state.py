from __future__ import annotations

import re
import threading
from typing import Optional

from schema import UIAction


current_goal: Optional[str] = None
last_action: Optional[UIAction] = None
autopilot_enabled: bool = False
dynamic_max_steps: int = 5
current_step: int = 0

_STATE_LOCK = threading.Lock()


def estimate_step_budget(goal: str) -> int:
    text = (goal or "").strip().lower()
    if not text:
        return 5

    action_keywords = [
        "click",
        "type",
        "enter",
        "search",
        "open",
        "select",
        "scroll",
        "submit",
        "close",
        "launch",
        "press",
        "find",
    ]
    connectors = [" and ", " then ", " after "]
    ambiguous_tokens = [" maybe ", " or ", " either ", " if possible ", " try "]

    action_count = sum(len(re.findall(rf"\\b{re.escape(keyword)}\\b", text)) for keyword in action_keywords)
    connector_count = sum(text.count(token) for token in connectors)
    is_ambiguous = any(token in f" {text} " for token in ambiguous_tokens)

    if action_count <= 1 and connector_count == 0:
        budget = 3
    elif action_count == 2 or connector_count >= 1:
        budget = 6
    else:
        budget = 10

    if action_count >= 3 or connector_count >= 2 or is_ambiguous:
        budget = max(budget, 10)

    if "open" in text and "search" in text and " and " in text:
        budget = max(budget, 4)

    budget += connector_count * 2
    return max(3, min(15, budget))


def set_goal(goal: str) -> None:
    """Set the current in-memory goal; blank goals are treated as no goal."""
    global current_goal, dynamic_max_steps
    cleaned = goal.strip()
    with _STATE_LOCK:
        current_goal = cleaned if cleaned else None
        dynamic_max_steps = estimate_step_budget(cleaned) if cleaned else 5


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
    global autopilot_enabled, current_step, dynamic_max_steps
    with _STATE_LOCK:
        autopilot_enabled = True
        current_step = 0
        if step_limit is not None:
            dynamic_max_steps = max(1, int(step_limit))


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
        return dynamic_max_steps


def set_max_steps(value: int) -> None:
    global dynamic_max_steps
    with _STATE_LOCK:
        dynamic_max_steps = max(1, int(value))


def get_dynamic_max_steps() -> int:
    with _STATE_LOCK:
        return dynamic_max_steps


def get_current_step() -> int:
    with _STATE_LOCK:
        return current_step


def set_current_step(value: int) -> None:
    global current_step
    with _STATE_LOCK:
        current_step = max(0, int(value))


def get_remaining_budget() -> int:
    with _STATE_LOCK:
        return max(0, dynamic_max_steps - current_step)


def clear_state() -> None:
    """Clear goal, last action, and autopilot runtime state."""
    global current_goal, last_action, autopilot_enabled, current_step, dynamic_max_steps
    with _STATE_LOCK:
        current_goal = None
        last_action = None
        autopilot_enabled = False
        current_step = 0
        dynamic_max_steps = 5
