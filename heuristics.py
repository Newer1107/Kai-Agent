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


def _aspect_ratio(element: dict[str, Any]) -> float:
    bbox = element.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return width / height


def _area_ratio(element: dict[str, Any], screen_width: int, screen_height: int) -> float:
    bbox = element.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [int(v) for v in bbox]
    area = max(1, (x2 - x1) * (y2 - y1))
    return float(area) / max(1.0, float(screen_width * screen_height))


def compute_affordances(element: dict[str, Any], screen_size: tuple[int, int]) -> dict[str, Any]:
    screen_width, screen_height = int(screen_size[0]), int(screen_size[1])
    aspect_ratio = _aspect_ratio(element)
    area_ratio = _area_ratio(element, screen_width, screen_height)
    cx, cy = _center(element)
    center_dx = abs(cx - (screen_width / 2.0)) / max(1.0, screen_width / 2.0)
    center_dy = abs(cy - (screen_height / 2.0)) / max(1.0, screen_height / 2.0)
    center_bias = max(0.0, 1.0 - ((center_dx + center_dy) / 2.0))

    text = " ".join(
        [
            str(element.get("text", "")),
            str(element.get("semantic_label", "")),
            str(element.get("type", "")),
        ]
    ).lower()
    text_hint = any(token in text for token in ["search", "enter", "type", "input", "find", "login", "submit"])

    can_type = aspect_ratio > 3.0 and screen_width > 0 and screen_height > 0 and _area_ratio(element, screen_width, screen_height) > 0.002
    if aspect_ratio > 3.0 and (aspect_ratio > 3.0 and int(element.get("bbox", [0, 0, 0, 0])[2]) - int(element.get("bbox", [0, 0, 0, 0])[0]) > 150):
        can_type = True
    if text_hint:
        can_type = True

    size_bias = max(0.0, min(1.0, area_ratio / 0.08))
    text_bias = 1.0 if text_hint else 0.0
    importance_score = max(0.0, min(1.0, (0.45 * center_bias) + (0.35 * size_bias) + (0.20 * text_bias)))
    if can_type:
        importance_score = min(1.0, importance_score + 0.12)

    return {
        "can_type": can_type,
        "can_click": True,
        "importance_score": round(importance_score, 4),
        "center_bias": round(center_bias, 4),
        "size_bias": round(size_bias, 4),
        "text_bias": round(text_bias, 4),
    }


def _annotate_affordances(element: dict[str, Any], screen_size: tuple[int, int]) -> dict[str, Any]:
    item = dict(element)
    item.update(compute_affordances(item, screen_size))
    return item


def _dedupe_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for element in elements:
        bbox = tuple(int(v) for v in element.get("bbox", [0, 0, 0, 0]))
        key = (bbox, str(element.get("type", "")), str(element.get("source", "")), str(element.get("text", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(element)
    return deduped


def detect_button_by_shape(
    elements: list[dict[str, Any]],
    screen_width: int,
    screen_height: int,
) -> Optional[dict[str, Any]]:
    candidates = []
    for element in elements:
        bbox = element.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect = width / height
        area = (width * height) / max(1.0, float(screen_width * screen_height))
        if 1.1 <= aspect <= 7.5 and 0.0007 <= area <= 0.14 and height >= 18:
            candidates.append((element, float(element.get("confidence", 0.0))))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[1], reverse=True)
    best, conf = candidates[0]
    heuristic_element = dict(best)
    heuristic_element["source"] = "heuristic_button_shape"
    heuristic_element["confidence"] = max(conf, 0.35)
    heuristic_element.update(compute_affordances(heuristic_element, (screen_width, screen_height)))
    return heuristic_element


def build_hybrid_candidates(elements: list[dict[str, Any]], screen_size: tuple[int, int]) -> list[dict[str, Any]]:
    screen_width, screen_height = int(screen_size[0]), int(screen_size[1])
    enriched = [_annotate_affordances(element, screen_size) for element in elements]

    if not any(bool(item.get("can_type", False)) for item in enriched):
        synthetic_input = infer_input_field(screen_width, screen_height)
        synthetic_input["source"] = "heuristic_center"
        synthetic_input.update(compute_affordances(synthetic_input, screen_size))
        enriched.append(synthetic_input)

    button_candidate = detect_button_by_shape(enriched, screen_width, screen_height)
    if button_candidate is not None:
        enriched.append(button_candidate)

    return _dedupe_elements(enriched)


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
