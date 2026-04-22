from __future__ import annotations

import math
from typing import Any, Optional


def _center(element: dict[str, Any]) -> tuple[int, int]:
    """Extract center coordinates from element."""
    center = element.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return int(center[0]), int(center[1])
    bbox = element.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _aspect_ratio(element: dict[str, Any]) -> float:
    """Calculate width/height ratio of element."""
    bbox = element.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return width / height


def _is_wide_element(element: dict[str, Any]) -> bool:
    """Check if element is horizontally stretched (likely input/search)."""
    return _aspect_ratio(element) > 3.5


def _is_center_region(element: dict[str, Any], screen_w: int, screen_h: int) -> bool:
    """Check if element is in center 60% of screen."""
    cx, cy = _center(element)
    x_margin = screen_w * 0.2
    y_margin = screen_h * 0.2
    return x_margin <= cx <= (screen_w - x_margin) and y_margin <= cy <= (screen_h - y_margin)


def infer_input_field(screen_width: int, screen_height: int) -> dict[str, Any]:
    """Generate heuristic input field at screen center (typical position for search/input)."""
    cx = screen_width // 2
    cy = int(screen_height * 0.35)
    width = int(screen_width * 0.6)
    height = int(screen_height * 0.08)
    
    x1 = cx - width // 2
    y1 = cy - height // 2
    x2 = x1 + width
    y2 = y1 + height
    
    return {
        "type": "input",
        "bbox": [max(0, x1), max(0, y1), min(screen_width, x2), min(screen_height, y2)],
        "center": [cx, cy],
        "confidence": 0.35,
        "region": "CENTER",
        "semantic_label": "input_field",
        "text": None,
        "source": "heuristic_center",
        "ambiguous": False,
        "resolution_score": 0.35,
        "score_breakdown": {
            "confidence": 0.35,
            "area": 0.3,
            "type_match": 1.0,
            "position": 0.9,
            "cluster": 0.0,
        },
    }


def detect_input_by_shape(
    elements: list[dict[str, Any]],
    screen_width: int,
    screen_height: int,
) -> Optional[dict[str, Any]]:
    """Find input-like element by aspect ratio and position."""
    candidates = []
    for element in elements:
        if _is_wide_element(element) and _is_center_region(element, screen_width, screen_height):
            confidence = float(element.get("confidence", 0.0))
            candidates.append((element, confidence))
    
    if not candidates:
        return None
    
    # Sort by confidence, return highest
    candidates.sort(key=lambda item: item[1], reverse=True)
    best, conf = candidates[0]
    
    # Tag with heuristic source
    heuristic_element = dict(best)
    heuristic_element["source"] = "heuristic_shape"
    heuristic_element["confidence"] = max(conf, 0.35)
    return heuristic_element


def detect_button_by_label(
    elements: list[dict[str, Any]],
    label_keywords: list[str],
) -> Optional[dict[str, Any]]:
    """Find button matching keyword in text or semantic label."""
    keywords_lower = [kw.lower() for kw in label_keywords]
    
    candidates = []
    for element in elements:
        text = str(element.get("text", "")).lower()
        semantic = str(element.get("semantic_label", "")).lower()
        etype = str(element.get("type", "")).lower()
        
        matched = any(kw in text for kw in keywords_lower) or any(kw in semantic for kw in keywords_lower)
        if matched and ("button" in etype or "link" in etype or "action" in semantic):
            confidence = float(element.get("confidence", 0.0))
            candidates.append((element, confidence))
    
    if not candidates:
        return None
    
    candidates.sort(key=lambda item: item[1], reverse=True)
    best, conf = candidates[0]
    
    heuristic_element = dict(best)
    heuristic_element["source"] = "heuristic_label"
    heuristic_element["confidence"] = max(conf, 0.4)
    return heuristic_element


def detect_by_goal_heuristic(
    elements: list[dict[str, Any]],
    goal: str,
    screen_width: int,
    screen_height: int,
) -> Optional[dict[str, Any]]:
    """Use goal context to infer best fallback element."""
    goal_lower = goal.lower()
    
    # Search/input intent
    if any(kw in goal_lower for kw in ["search", "type", "enter", "input", "find"]):
        # Try shape-based first
        candidate = detect_input_by_shape(elements, screen_width, screen_height)
        if candidate is not None:
            return candidate
        # Otherwise, use center heuristic
        return infer_input_field(screen_width, screen_height)
    
    # Submit/send intent
    if any(kw in goal_lower for kw in ["submit", "send", "post", "login", "sign"]):
        button = detect_button_by_label(
            elements,
            ["submit", "send", "post", "login", "sign in", "ok", "yes"],
        )
        if button is not None:
            return button
    
    # Click/navigate intent (no specific fallback needed)
    
    return None


def is_safe_heuristic_location(
    element: dict[str, Any],
    screen_width: int,
    screen_height: int,
) -> bool:
    """Ensure heuristic element center is in reasonable UI area, not edges."""
    cx, cy = _center(element)
    
    # Reject if too close to screen edges
    margin = min(screen_width, screen_height) * 0.1
    if cx <= margin or cx >= (screen_width - margin):
        return False
    if cy <= margin or cy >= (screen_height - margin):
        return False
    
    # Accept if in any valid UI region (not just random points)
    # Must be in top 90% of screen (avoid bottom-most taskbar area)
    return cy < screen_height * 0.9
