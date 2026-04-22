from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image

try:
    import cv2
except Exception:  # pragma: no cover - handled safely at runtime
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - handled safely at runtime
    np = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - handled safely at runtime
    YOLO = None


class VisionError(RuntimeError):
    """Raised when the YOLO vision pipeline cannot produce detections."""


_MODEL: YOLO | None = None


def _require_runtime_deps() -> None:
    if np is None or cv2 is None:
        raise VisionError("opencv-python and numpy are required for vision detection")


def _resolve_model_name() -> str:
    custom = Path("best.pt")
    if custom.exists():
        return str(custom)
    return "yolov8n.pt"


def _get_model() -> YOLO:
    global _MODEL
    if YOLO is None:
        raise VisionError("ultralytics is not installed. Install dependencies from requirements.txt")

    if _MODEL is None:
        _MODEL = YOLO(_resolve_model_name())
        _MODEL.to("cpu")

    return _MODEL


def _to_numpy_rgb(image_or_path: Any) -> np.ndarray:
    _require_runtime_deps()

    if isinstance(image_or_path, (str, Path)):
        image = cv2.imread(str(image_or_path))
        if image is None:
            raise VisionError(f"Could not read image path: {image_or_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if isinstance(image_or_path, Image.Image):
        return np.asarray(image_or_path.convert("RGB"))

    if isinstance(image_or_path, np.ndarray):
        if image_or_path.ndim != 3:
            raise VisionError("Expected HxWxC image array for detection")
        # Assume RGB if caller passes array directly.
        return image_or_path

    raise VisionError("Unsupported image input type for YOLO detection")


def detect_ui_elements(image_or_path: Any, conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """Detect UI elements and return label/confidence/bbox/center records."""
    _require_runtime_deps()
    model = _get_model()
    rgb = _to_numpy_rgb(image_or_path)
    results = model.predict(source=rgb, conf=conf_threshold, verbose=False, device="cpu")

    detections: List[Dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        names = result.names
        for box in boxes:
            cls_id = int(box.cls.item())
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            label: str
            if isinstance(names, dict):
                label = str(names.get(cls_id, cls_id))
            else:
                label = str(names[cls_id])

            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                }
            )

    return detections


def save_detection_debug_image(
    image_or_path: Any,
    detections: Sequence[Dict[str, Any]],
    output_path: str,
    highlight_label: str | None = None,
) -> str:
    """Draw detection boxes and labels and save a debug image."""
    _require_runtime_deps()
    rgb = _to_numpy_rgb(image_or_path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    highlight = (highlight_label or "").lower()
    for detection in detections:
        label = str(detection.get("label", "unknown"))
        confidence = float(detection.get("confidence", 0.0))
        x1, y1, x2, y2 = detection.get("bbox", (0, 0, 0, 0))

        color = (50, 210, 50) if label.lower() == highlight else (70, 170, 255)
        cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(
            bgr,
            f"{label} {confidence:.2f}",
            (int(x1), max(18, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), bgr)
    return str(output)
