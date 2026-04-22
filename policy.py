from __future__ import annotations

from typing import Literal


def get_action_risk(action: str) -> Literal["low", "medium", "high"]:
    """Classify action by inherent risk level."""
    action = action.lower().strip()
    
    # Type is low-risk: wrong field is annoying but not dangerous
    if action == "type":
        return "low"
    
    # Click is medium-risk: might trigger unwanted action but usually recoverable
    if action == "click":
        return "medium"
    
    # Scroll and wait are high-risk (scroll could lose focus, wait wastes time)
    if action == "scroll":
        return "high"
    
    return "high"


def is_intent_clear(goal: str) -> bool:
    """Check if user's intent is explicit enough to justify lower confidence threshold."""
    if not goal:
        return False
    
    goal_lower = goal.lower()
    action_keywords = [
        "search",
        "type",
        "enter",
        "input",
        "submit",
        "click",
        "login",
        "sign in",
        "send",
        "find",
        "look for",
        "go to",
    ]
    
    return any(kw in goal_lower for kw in action_keywords)


def get_min_confidence_for_action(
    action: str,
    source: str,
    goal: str,
) -> float:
    """
    Get minimum required confidence for action execution.
    
    Risk-based policy:
    - Low-risk actions (type) need lower confidence
    - Medium-risk actions (click) need moderate confidence
    - High-risk actions (scroll) need high confidence
    - Heuristic sources always allowed (confidence floor: 0.25)
    - Clear intent lowers threshold for low/medium risk
    """
    # Heuristic sources: always allowed (but min floor of 0.25 for sanity)
    if source.startswith("heuristic"):
        return 0.25
    
    # YOLO sources: use risk-based policy
    action_lower = action.lower()
    risk = get_action_risk(action)
    clear_intent = is_intent_clear(goal)
    
    if risk == "low":
        # Type: allow 0.3 with clear intent, 0.5 otherwise
        return 0.3 if clear_intent else 0.5
    
    if risk == "medium":
        # Click: allow 0.4 with clear intent, 0.5 otherwise
        return 0.4 if clear_intent else 0.5
    
    # High-risk: always require 0.7
    return 0.7


def is_action_allowed(
    action: str,
    confidence: float,
    source: str,
    goal: str,
) -> bool:
    """
    Determine if action should be executed based on risk profile.
    
    Allows actions with lower confidence when:
    1. Action is low-risk (type)
    2. Source is heuristic fallback
    3. User intent is clear
    """
    min_confidence = get_min_confidence_for_action(action, source, goal)
    allowed = confidence >= min_confidence
    
    # Debug
    print(f"[POLICY] Action: {action} | Risk: {get_action_risk(action)} | Confidence: {confidence:.2f} | Min: {min_confidence:.2f} | Source: {source} | Intent clear: {is_intent_clear(goal)} | ALLOWED: {allowed}")
    
    return allowed
