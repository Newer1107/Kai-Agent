from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw


def draw_target_preview(
    image: Image.Image,
    element: Optional[dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    if element is None:
        return None

    preview = image.convert("RGB").copy()
    draw = ImageDraw.Draw(preview)

    bbox = [int(v) for v in element.get("bbox", [0, 0, 0, 0])]
    x1, y1, x2, y2 = bbox
    center = element.get("center", [int((x1 + x2) / 2), int((y1 + y2) / 2)])
    cx, cy = int(center[0]), int(center[1])

    draw.rectangle([(x1, y1), (x2, y2)], outline=(30, 220, 120), width=3)
    draw.ellipse([(cx - 6, cy - 6), (cx + 6, cy + 6)], outline=(255, 230, 60), width=3)

    label = str(element.get("type", "unknown"))
    confidence = float(element.get("confidence", 0.0))
    semantic = str(element.get("semantic_label", ""))
    text = str(element.get("text", ""))
    region = str(element.get("region", ""))
    score = float(element.get("resolution_score", 0.0))
    can_type = bool(element.get("can_type", False))
    can_click = bool(element.get("can_click", False))
    importance = float(element.get("importance_score", 0.0))

    caption = (
        f"TARGET: {label} conf={confidence:.2f} score={score:.2f} "
        f"region={region} semantic={semantic} text={text[:40]} "
        f"T:{int(can_type)} C:{int(can_click)} I:{importance:.2f}"
    )
    draw.text((x1 + 2, max(2, y1 - 14)), caption, fill=(30, 220, 120))

    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        preview.save(output)
        if show:
            preview.show()
        return str(output)

    if show:
        preview.show()
    return None
