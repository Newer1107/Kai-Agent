from __future__ import annotations

from typing import Optional

from schema import UIAction


current_goal: Optional[str] = None
last_action: Optional[UIAction] = None


def set_goal(goal: str) -> None:
    """Set the current in-memory goal; blank goals are treated as no goal."""
    global current_goal
    cleaned = goal.strip()
    current_goal = cleaned if cleaned else None


def clear_goal() -> None:
    """Clear the current in-memory goal."""
    global current_goal
    current_goal = None


def get_goal() -> Optional[str]:
    """Return the current in-memory goal, if any."""
    return current_goal


def set_last_action(action: Optional[UIAction]) -> None:
    """Store the last approved step in memory for contextual reasoning."""
    global last_action
    last_action = action


def get_last_action() -> Optional[UIAction]:
    """Return the last approved action, if any."""
    return last_action


def clear_state() -> None:
    """Clear both goal and last action state."""
    global current_goal, last_action
    current_goal = None
    last_action = None
