from __future__ import annotations

from typing import Any, Iterable, Tuple

TOP_BAR = "TOP_BAR"
CENTER = "CENTER"
LEFT_PANEL = "LEFT_PANEL"
RIGHT_PANEL = "RIGHT_PANEL"
FOOTER = "FOOTER"


def _safe_center(element: dict[str, Any]) -> tuple[float, float]:
    center = element.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        return float(center[0]), float(center[1])

    bbox = element.get("bbox", [0, 0, 0, 0])
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def assign_region(element: dict[str, Any], screen_size: Tuple[int, int]) -> str:
    width, height = int(screen_size[0]), int(screen_size[1])
    if width <= 0 or height <= 0:
        return CENTER

    x, y = _safe_center(element)

    if y >= 0.88 * height:
        return FOOTER
    if y <= 0.18 * height:
        return TOP_BAR
    if x <= 0.22 * width:
        return LEFT_PANEL
    if x >= 0.78 * width:
        return RIGHT_PANEL
    return CENTER


def attach_regions(elements: Iterable[dict[str, Any]], screen_size: Tuple[int, int]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for element in elements:
        item = dict(element)
        item["region"] = assign_region(item, screen_size)
        enriched.append(item)
    return enriched
