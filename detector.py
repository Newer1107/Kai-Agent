from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Sequence

from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class DetectorError(RuntimeError):
    """Raised when the detector cannot load or run safely."""


_MODEL: YOLO | None = None
_MODEL_LOCK = threading.Lock()
_TRAINED_MODEL_PATH = Path(r"C:\Users\Raunak\Documents\Kai\runs\detect\train-14\weights\best.pt")
_FALLBACK_MODEL_PATH = Path("yolov8n.pt")
_MIN_CONFIDENCE = 0.5
_LAST_INFERENCE_MS: float = 0.0


def _ensure_runtime_deps() -> None:
    if YOLO is None:
        raise DetectorError("ultralytics is required for detection")
    if np is None or cv2 is None:
        raise DetectorError("numpy and opencv-python are required for detection")


def _resolve_model_path() -> str:
    if _TRAINED_MODEL_PATH.exists():
        return str(_TRAINED_MODEL_PATH)
    if _FALLBACK_MODEL_PATH.exists():
        return str(_FALLBACK_MODEL_PATH)
    return "yolov8n.pt"


def load_model() -> YOLO:
    """Load and cache YOLO model globally to avoid repeated startup overhead."""
    global _MODEL
    _ensure_runtime_deps()

    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                model_path = _resolve_model_path()
                try:
                    _MODEL = YOLO(model_path)
                except Exception as exc:
                    raise DetectorError(f"failed to load YOLO model from '{model_path}': {exc}") from exc

    return _MODEL


def _resolve_inference_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _to_numpy_rgb(image: Any) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))

    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise DetectorError(f"could not read image from path: {image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if isinstance(image, np.ndarray):
        if image.ndim != 3:
            raise DetectorError("expected HxWxC image array")
        return image

    raise DetectorError("unsupported image type for detection")


def get_center(box: Sequence[int]) -> List[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    return [int((x1 + x2) / 2), int((y1 + y2) / 2)]


def filter_by_type(elements: List[dict[str, Any]], type_name: str) -> List[dict[str, Any]]:
    wanted = type_name.strip().lower()
    return [e for e in elements if str(e.get("type", "")).lower() == wanted]


def detect_ui_elements(image: Image.Image) -> List[dict[str, Any]]:
    """Detect UI elements and return clean structured detections."""
    global _LAST_INFERENCE_MS
    _ensure_runtime_deps()
    model = load_model()
    rgb = _to_numpy_rgb(image)
    device = _resolve_inference_device()

    start = time.perf_counter()
    try:
        results = model.predict(source=rgb, conf=_MIN_CONFIDENCE, verbose=False, device=device)
    except Exception as exc:
        if device != "cpu":
            try:
                results = model.predict(source=rgb, conf=_MIN_CONFIDENCE, verbose=False, device="cpu")
                device = "cpu"
            except Exception:
                raise DetectorError(f"YOLO inference failed: {exc}") from exc
        else:
            raise DetectorError(f"YOLO inference failed: {exc}") from exc
    _LAST_INFERENCE_MS = (time.perf_counter() - start) * 1000.0

    elements: List[dict[str, Any]] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        names = result.names
        for box in boxes:
            confidence = float(box.conf.item())
            if confidence < _MIN_CONFIDENCE:
                continue

            cls_id = int(box.cls.item())
            if isinstance(names, dict):
                label = str(names.get(cls_id, cls_id))
            else:
                label = str(names[cls_id])

            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            bbox = [x1, y1, x2, y2]
            center = get_center(bbox)

            elements.append(
                {
                    "type": label,
                    "bbox": bbox,
                    "center": center,
                    "confidence": confidence,
                }
            )

    elements.sort(key=lambda item: float(item["confidence"]), reverse=True)
    return elements


def draw_detections(
    image: Image.Image,
    elements: List[dict[str, Any]],
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """Draw boxes + labels for debug visibility and optionally save to file."""
    debug_img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(debug_img)

    for element in elements:
        bbox = element.get("bbox", [0, 0, 0, 0])
        label = str(element.get("type", "unknown"))
        confidence = float(element.get("confidence", 0.0))

        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([(x1, y1), (x2, y2)], outline=(70, 180, 255), width=2)
        draw.text((x1 + 2, max(2, y1 - 14)), f"{label} {confidence:.2f}", fill=(70, 180, 255))

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        debug_img.save(out)
        if show:
            debug_img.show()
        return str(out)

    if show:
        debug_img.show()

    return None


def get_last_inference_ms() -> float:
    """Expose last measured inference latency in milliseconds for diagnostics."""
    return _LAST_INFERENCE_MS
