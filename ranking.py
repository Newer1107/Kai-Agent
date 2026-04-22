from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RankedElement:
    element: dict[str, Any]
    score: float
    score_breakdown: dict[str, float]


def _center_of(element: dict[str, Any]) -> tuple[float, float]:
    center = element.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return float(center[0]), float(center[1])
    bbox = element.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_area(element: dict[str, Any]) -> float:
    bbox = element.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(1.0, (x2 - x1) * (y2 - y1))


def _infer_screen_size(elements: Iterable[dict[str, Any]]) -> tuple[float, float]:
    max_x, max_y = 1.0, 1.0
    for element in elements:
        bbox = element.get("bbox", [0, 0, 0, 0])
        _, _, x2, y2 = [float(v) for v in bbox]
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
    return max_x, max_y


def _type_match_score(element_type: str, goal: str, action_type: str) -> float:
    t = element_type.lower()
    g = goal.lower()
    a = action_type.lower()

    if a == "type":
        if "input" in t or "textbox" in t or "field" in t:
            return 1.0
        if "search" in g and ("input" in t or "bar" in t):
            return 1.0
        if "button" in t:
            return 0.25

    if a == "click":
        if "button" in t or "link" in t or "tab" in t or "menu" in t:
            return 1.0
        if "input" in t:
            return 0.55

    if a == "scroll":
        if "panel" in t or "list" in t or "feed" in t:
            return 1.0
        return 0.4

    return 0.5


def _position_score(element: dict[str, Any], screen_w: float, screen_h: float, goal: str, action_type: str) -> float:
    x, y = _center_of(element)
    a = action_type.lower()
    g = goal.lower()

    x_norm = min(1.0, max(0.0, x / max(1.0, screen_w)))
    y_norm = min(1.0, max(0.0, y / max(1.0, screen_h)))

    if a == "type" or "search" in g:
        x_pref = 1.0 - abs(x_norm - 0.5) * 2.0
        y_pref = 1.0 - abs(y_norm - 0.25) * 1.4
        return max(0.0, min(1.0, 0.65 * x_pref + 0.35 * y_pref))

    if "menu" in g or "nav" in g:
        return max(0.0, 1.0 - y_norm * 1.6)

    return max(0.0, 1.0 - y_norm)


def _cluster_score(index: int, elements: list[dict[str, Any]], radius: float = 220.0) -> float:
    x0, y0 = _center_of(elements[index])
    neighbors = 0
    for idx, other in enumerate(elements):
        if idx == index:
            continue
        x1, y1 = _center_of(other)
        if math.hypot(x1 - x0, y1 - y0) <= radius:
            neighbors += 1
    return min(1.0, neighbors / 8.0)


def rank_elements(elements: list[dict[str, Any]], goal: str, action_type: str) -> list[RankedElement]:
    if not elements:
        return []

    screen_w, screen_h = _infer_screen_size(elements)
    max_area = max(_bbox_area(el) for el in elements)

    ranked: list[RankedElement] = []
    for idx, element in enumerate(elements):
        confidence = max(0.0, min(1.0, float(element.get("confidence", 0.0))))
        area_score = min(1.0, _bbox_area(element) / max_area)
        type_score = _type_match_score(str(element.get("type", "")), goal, action_type)
        position_score = _position_score(element, screen_w, screen_h, goal, action_type)
        cluster_score = _cluster_score(idx, elements)
        
        # Add CENTER region bias: prioritize center-screen elements
        center_bias = 0.0
        region = str(element.get("region", "")).lower()
        if region == "center":
            center_bias = 0.15  # Boost for center region
        
        # Wide boxes in CENTER region are likely inputs/search bars
        bbox = element.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]
        width = x2 - x1
        height = max(1, y2 - y1)
        if region == "center" and width / height > 2.5:
            center_bias += 0.15  # Extra boost for wide center elements

        score = (
            0.30 * confidence
            + 0.20 * area_score
            + 0.25 * type_score
            + 0.15 * position_score
            + 0.10 * cluster_score
        ) + center_bias

        ranked.append(
            RankedElement(
                element=element,
                score=score,
                score_breakdown={
                    "confidence": confidence,
                    "area": area_score,
                    "type_match": type_score,
                    "position": position_score,
                    "cluster": cluster_score,
                    "center_bias": center_bias,
                },
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked
