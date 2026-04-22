from __future__ import annotations

import math
import os
import re
from typing import Any

from PIL import Image

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_text_regions(
    image: Image.Image,
    enabled: bool | None = None,
    min_confidence: int = 40,
) -> list[dict[str, Any]]:
    if enabled is None:
        enabled = os.getenv("KAI_ENABLE_OCR", "0").strip().lower() in {"1", "true", "yes", "on"}

    if not enabled or pytesseract is None:
        return []

    rgb = image.convert("RGB")
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)

    regions: list[dict[str, Any]] = []
    total = len(data.get("text", []))
    for idx in range(total):
        raw_text = str(data["text"][idx])
        text = _normalize_text(raw_text)
        if not text:
            continue

        try:
            conf = float(data["conf"][idx])
        except Exception:
            conf = -1.0

        if conf < float(min_confidence):
            continue

        x = int(data["left"][idx])
        y = int(data["top"][idx])
        w = int(data["width"][idx])
        h = int(data["height"][idx])
        regions.append(
            {
                "text": text,
                "confidence": conf / 100.0,
                "bbox": [x, y, x + w, y + h],
            }
        )

    return regions


def _boxes_overlap(box_a: list[int], box_b: list[int]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _center(box: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _distance(box_a: list[int], box_b: list[int]) -> float:
    ax, ay = _center(box_a)
    bx, by = _center(box_b)
    return math.hypot(ax - bx, ay - by)


def _semantic_label_from_text(element_type: str, text: str) -> str:
    t = text.lower()
    e = element_type.lower()

    if "search" in t and ("input" in e or "field" in e or "bar" in e):
        return "search_input"
    if any(token in t for token in ["sign in", "login", "log in"]):
        return "auth_action"
    if any(token in t for token in ["submit", "send", "post"]):
        return "submit_action"
    if any(token in t for token in ["menu", "home", "profile", "settings"]):
        return "navigation"
    if any(token in t for token in ["next", "continue", "ok", "yes"]):
        return "forward_action"

    if "input" in e or "field" in e:
        return "input_field"
    if "button" in e:
        return "button_action"
    if "link" in e:
        return "link_action"
    return "generic_ui"


def attach_semantics(
    elements: list[dict[str, Any]],
    text_regions: list[dict[str, Any]],
    radius: float = 120.0,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []

    for element in elements:
        item = dict(element)
        bbox = [int(v) for v in item.get("bbox", [0, 0, 0, 0])]

        best_region: dict[str, Any] | None = None
        best_score = -1.0
        for region in text_regions:
            region_bbox = [int(v) for v in region.get("bbox", [0, 0, 0, 0])]
            overlap = _boxes_overlap(bbox, region_bbox)
            dist = _distance(bbox, region_bbox)
            if not overlap and dist > radius:
                continue

            proximity = 1.0 if overlap else max(0.0, 1.0 - (dist / radius))
            score = 0.7 * proximity + 0.3 * float(region.get("confidence", 0.0))
            if score > best_score:
                best_score = score
                best_region = region

        if best_region is not None:
            text = str(best_region.get("text", "")).strip()
            item["text"] = text
            item["semantic_label"] = _semantic_label_from_text(str(item.get("type", "")), text)
            item["text_confidence"] = float(best_region.get("confidence", 0.0))
        else:
            item["text"] = None
            item["semantic_label"] = _semantic_label_from_text(
                str(item.get("type", "")),
                "",
            )
            item["text_confidence"] = 0.0

        enriched.append(item)

    return enriched
