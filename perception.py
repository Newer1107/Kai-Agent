from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple

import pyautogui
from PIL import Image, ImageGrab

from detector import DetectorError, detect_ui_elements, get_last_fallback_used, get_last_inference_ms
from heuristics import build_hybrid_candidates

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

try:
    import win32gui
except Exception:  # pragma: no cover
    win32gui = None


@dataclass(frozen=True)
class AppContext:
    app_name: str
    window_title: str
    confidence: float
    detected_at: float


@dataclass(frozen=True)
class ScreenInferencePayload:
    """Image payload and coordinate-space metadata used for model inference."""

    image: Image.Image
    image_base64: str
    ui_elements: List[dict[str, Any]]
    text_regions: List[dict[str, Any]]
    ocr_enabled: bool
    app_context: AppContext
    original_size: Tuple[int, int]
    resized_size: Tuple[int, int]
    region_offset: Tuple[int, int] = (0, 0)

    @property
    def base64_image(self) -> str:
        """Backward-compatible alias for existing call sites."""
        return self.image_base64


_APP_CONTEXT_LOCK = threading.Lock()
_APP_CONTEXT_CACHE: AppContext | None = None
_APP_CONTEXT_CACHE_TTL = 0.5

_APP_TITLE_LOOKUP: dict[str, list[str]] = {
    "chrome": ["Google Chrome", "Chrome"],
    "firefox": ["Mozilla Firefox", "Firefox"],
    "notepad": ["Notepad"],
    "explorer": ["File Explorer", "Windows Explorer", "This PC"],
    "vscode": ["Visual Studio Code", "Code"],
    "excel": ["Microsoft Excel", "Excel"],
    "word": ["Microsoft Word", "Word"],
    "settings": ["Settings"],
    "calculator": ["Calculator"],
}


def _normalize_app_from_title(window_title: str) -> tuple[str, float]:
    title = window_title.lower()
    for app_name, patterns in _APP_TITLE_LOOKUP.items():
        for pattern in patterns:
            if pattern.lower() in title:
                return app_name, 0.9
    return "unknown", 0.3


def get_active_app_context() -> AppContext:
    global _APP_CONTEXT_CACHE
    now = time.time()
    with _APP_CONTEXT_LOCK:
        if _APP_CONTEXT_CACHE is not None and (now - _APP_CONTEXT_CACHE.detected_at) <= _APP_CONTEXT_CACHE_TTL:
            return _APP_CONTEXT_CACHE

    window_title = ""
    if win32gui is not None:
        try:
            hwnd = win32gui.GetForegroundWindow()
            window_title = win32gui.GetWindowText(hwnd) or ""
        except Exception:
            window_title = ""

    app_name, confidence = _normalize_app_from_title(window_title)
    context = AppContext(
        app_name=app_name,
        window_title=window_title,
        confidence=confidence,
        detected_at=now,
    )

    with _APP_CONTEXT_LOCK:
        _APP_CONTEXT_CACHE = context
    return context


class ConfidenceTracker:
    def __init__(self, window_size: int = 50) -> None:
        self._runs: deque[dict[str, Any]] = deque(maxlen=max(1, window_size))
        self._lock = threading.Lock()

    def update_run(
        self,
        elements: list[dict[str, Any]],
        inference_ms: float,
        fallback_used: bool,
    ) -> dict[str, Any]:
        label_counts: dict[str, int] = {}
        label_conf_sum: dict[str, float] = {}
        confidences: list[float] = []

        for element in elements:
            label = str(element.get("type", "unknown")).strip().lower() or "unknown"
            confidence = max(0.0, min(1.0, float(element.get("confidence", 0.0))))
            label_counts[label] = label_counts.get(label, 0) + 1
            label_conf_sum[label] = label_conf_sum.get(label, 0.0) + confidence
            confidences.append(confidence)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        min_conf = min(confidences) if confidences else 0.0
        max_conf = max(confidences) if confidences else 0.0

        per_label_avg = {
            label: (label_conf_sum[label] / max(1, count)) for label, count in label_counts.items()
        }
        run_payload = {
            "timestamp": time.time(),
            "label_counts": label_counts,
            "per_label_avg": per_label_avg,
            "avg_conf": avg_conf,
            "min_conf": min_conf,
            "max_conf": max_conf,
            "inference_ms": float(inference_ms),
            "fallback_used": bool(fallback_used),
        }

        with self._lock:
            self._runs.append(run_payload)

        return run_payload

    def _aggregate(self) -> tuple[dict[str, float], dict[str, int], dict[str, float] | None]:
        with self._lock:
            runs = list(self._runs)

        conf_sum: dict[str, float] = {}
        count_sum: dict[str, int] = {}
        for run in runs:
            label_counts = run.get("label_counts", {})
            per_label_avg = run.get("per_label_avg", {})
            for label, count in label_counts.items():
                count_int = int(count)
                count_sum[label] = count_sum.get(label, 0) + count_int
                conf_sum[label] = conf_sum.get(label, 0.0) + (float(per_label_avg.get(label, 0.0)) * count_int)

        avg_by_label: dict[str, float] = {}
        for label, count in count_sum.items():
            avg_by_label[label] = conf_sum.get(label, 0.0) / max(1, count)

        last_run = runs[-1] if runs else None
        return avg_by_label, count_sum, last_run

    def get_weak_labels(self) -> list[str]:
        avg_by_label, _, _ = self._aggregate()
        return sorted([label for label, avg in avg_by_label.items() if avg < 0.45])

    def get_telemetry_summary(self) -> dict[str, Any]:
        avg_by_label, count_sum, last_run = self._aggregate()
        with self._lock:
            run_count = len(self._runs)
        labels = []
        for label in sorted(avg_by_label.keys()):
            avg = avg_by_label[label]
            detections = count_sum.get(label, 0)
            labels.append(
                {
                    "label": label,
                    "avg_conf": round(avg, 4),
                    "detections": int(detections),
                    "status": "WEAK" if avg < 0.45 else "OK",
                }
            )

        return {
            "window_size": run_count,
            "labels": labels,
            "weak_labels": self.get_weak_labels(),
            "last_run": last_run,
        }


_CONFIDENCE_TRACKER = ConfidenceTracker(window_size=50)


def get_weak_labels() -> list[str]:
    return _CONFIDENCE_TRACKER.get_weak_labels()


def get_telemetry_summary() -> dict[str, Any]:
    return _CONFIDENCE_TRACKER.get_telemetry_summary()


def _telemetry_logging_enabled() -> bool:
    return os.getenv("KAI_TELEMETRY_LOG", "0").strip().lower() in {"1", "true", "yes", "on"}


def _append_telemetry_log(payload: dict[str, Any]) -> None:
    if not _telemetry_logging_enabled():
        return

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%Y-%m-%d")
    out_path = logs_dir / f"telemetry_{date_tag}.jsonl"
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def capture_primary_screenshot() -> Image.Image:
    """Capture the primary monitor using pyautogui, with a Pillow fallback."""
    try:
        return pyautogui.screenshot()
    except Exception:
        return ImageGrab.grab(all_screens=False)


def _capture_with_retry() -> Image.Image | None:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            return capture_primary_screenshot()
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(0.1)
    print(f"[CAPTURE_FAIL] reason={type(last_error).__name__ if last_error else 'unknown'}: {last_error}")
    return None


def _blank_payload(max_width: int, app_context: AppContext) -> ScreenInferencePayload:
    try:
        width, height = pyautogui.size()
    except Exception:
        width, height = (1280, 720)

    fallback_image = Image.new("RGB", (max(1, width), max(1, height)), color=(0, 0, 0))
    resized = resize_for_inference(fallback_image, max_width=max_width)
    return ScreenInferencePayload(
        image=fallback_image,
        image_base64=encode_image_to_base64(resized),
        ui_elements=[],
        text_regions=[],
        ocr_enabled=False,
        app_context=app_context,
        original_size=(fallback_image.width, fallback_image.height),
        resized_size=(resized.width, resized.height),
        region_offset=(0, 0),
    )


def _assign_region(element: dict[str, Any], screen_size: tuple[int, int]) -> str:
    width, height = int(screen_size[0]), int(screen_size[1])
    center = element.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        x, y = float(center[0]), float(center[1])
    else:
        bbox = element.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [float(v) for v in bbox]
        x, y = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    if y >= 0.88 * height:
        return "FOOTER"
    if y <= 0.18 * height:
        return "TOP_BAR"
    if x <= 0.22 * width:
        return "LEFT_PANEL"
    if x >= 0.78 * width:
        return "RIGHT_PANEL"
    return "CENTER"


def _attach_regions(elements: list[dict[str, Any]], screen_size: tuple[int, int]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for element in elements:
        item = dict(element)
        item["region"] = _assign_region(item, screen_size)
        enriched.append(item)
    return enriched


def _extract_text_regions(
    image: Image.Image,
    enabled: bool,
    min_confidence: int = 40,
) -> list[dict[str, Any]]:
    if not enabled or pytesseract is None:
        return []

    rgb = image.convert("RGB")
    data = pytesseract.image_to_data(rgb, output_type=pytesseract.Output.DICT)
    regions: list[dict[str, Any]] = []
    total = len(data.get("text", []))
    for idx in range(total):
        text = str(data["text"][idx]).strip()
        if not text:
            continue

        try:
            conf = float(data["conf"][idx])
        except Exception:
            conf = -1.0
        if conf < float(min_confidence):
            continue

        left = int(data["left"][idx])
        top = int(data["top"][idx])
        width = int(data["width"][idx])
        height = int(data["height"][idx])
        regions.append(
            {
                "text": " ".join(text.split()),
                "confidence": conf / 100.0,
                "bbox": [left, top, left + width, top + height],
            }
        )
    return regions


def _bbox_overlap(box_a: list[int], box_b: list[int]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _bbox_center(box: list[int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_distance(box_a: list[int], box_b: list[int]) -> float:
    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def _semantic_label(element_type: str, text: str) -> str:
    lowered_text = text.lower()
    lowered_type = element_type.lower()
    if "search" in lowered_text:
        return "search_input"
    if any(token in lowered_text for token in ["submit", "send", "post", "confirm"]):
        return "submit_action"
    if any(token in lowered_text for token in ["login", "sign in", "log in"]):
        return "auth_action"
    if any(token in lowered_text for token in ["next", "continue", "ok", "yes"]):
        return "forward_action"
    if "input" in lowered_type or "field" in lowered_type or "textbox" in lowered_type:
        return "input_field"
    if "button" in lowered_type:
        return "button_action"
    return "generic_ui"


def _attach_text_to_elements(
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
            overlap = _bbox_overlap(bbox, region_bbox)
            distance = _bbox_distance(bbox, region_bbox)
            if not overlap and distance > radius:
                continue
            proximity = 1.0 if overlap else max(0.0, 1.0 - (distance / radius))
            score = (0.7 * proximity) + (0.3 * float(region.get("confidence", 0.0)))
            if score > best_score:
                best_score = score
                best_region = region

        if best_region is not None:
            text = str(best_region.get("text", "")).strip()
            item["text"] = text
            item["text_confidence"] = float(best_region.get("confidence", 0.0))
            item["semantic_label"] = _semantic_label(str(item.get("type", "")), text)
        else:
            item["text"] = None
            item["text_confidence"] = 0.0
            item["semantic_label"] = _semantic_label(str(item.get("type", "")), "")

        enriched.append(item)
    return enriched


def resize_for_inference(image: Image.Image, max_width: int = 1280) -> Image.Image:
    """Resize while preserving aspect ratio to reduce inference cost."""
    if image.width <= max_width:
        return image.copy()

    ratio = max_width / float(image.width)
    new_height = max(1, int(image.height * ratio))
    resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
    return image.resize((max_width, new_height), resampling)


def encode_image_to_base64(image: Image.Image, image_format: str = "PNG") -> str:
    """Convert a Pillow image into base64 text suitable for multimodal API payloads."""
    with BytesIO() as buffer:
        image.save(buffer, format=image_format, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def capture_screen_as_base64(max_width: int = 1280) -> str:
    """Capture, resize, and encode the current primary screen."""
    return capture_screen_for_inference(max_width=max_width).image_base64


def capture_screen_for_inference(
    max_width: int = 1280,
    enable_ocr: bool | None = None,
    enriched: bool = True,
) -> ScreenInferencePayload:
    """Capture screen and return image, encoded image, and fast UI detections."""
    app_context = get_active_app_context()
    print(f"[CONTEXT] app={app_context.app_name} window=\"{app_context.window_title}\"")

    screenshot = _capture_with_retry()
    if screenshot is None:
        return _blank_payload(max_width=max_width, app_context=app_context)

    resized = resize_for_inference(screenshot, max_width=max_width)

    detector_failed = False
    try:
        elements = detect_ui_elements(screenshot, max_width=max_width)
    except DetectorError:
        elements = []
        detector_failed = True

    text_regions: List[dict[str, Any]] = []
    final_elements = _attach_regions(elements, (screenshot.width, screenshot.height))
    ocr_enabled = False

    if enriched:
        if enable_ocr is None:
            enable_ocr = os.getenv("KAI_ENABLE_OCR", "0").strip().lower() in {"1", "true", "yes", "on"}

        try:
            text_regions = _extract_text_regions(screenshot, enabled=bool(enable_ocr))
            final_elements = _attach_text_to_elements(final_elements, text_regions)
            ocr_enabled = bool(enable_ocr)
        except Exception:
            text_regions = []
            final_elements = _attach_regions(elements, (screenshot.width, screenshot.height))
            ocr_enabled = False

    try:
        final_elements = build_hybrid_candidates(
            final_elements,
            (screenshot.width, screenshot.height),
        )
    except Exception:
        final_elements = _attach_regions(elements, (screenshot.width, screenshot.height))

    telemetry = _CONFIDENCE_TRACKER.update_run(
        elements=elements,
        inference_ms=get_last_inference_ms(),
        fallback_used=bool(detector_failed or get_last_fallback_used()),
    )
    print(
        "[YOLO_TELEMETRY] "
        f"labels={telemetry['label_counts']} "
        f"avg_conf={telemetry['avg_conf']:.3f} "
        f"min_conf={telemetry['min_conf']:.3f} "
        f"max_conf={telemetry['max_conf']:.3f} "
        f"inference_ms={telemetry['inference_ms']:.1f} "
        f"fallback_used={telemetry['fallback_used']}"
    )
    _append_telemetry_log(telemetry)

    return ScreenInferencePayload(
        image=screenshot.copy(),
        image_base64=encode_image_to_base64(resized),
        ui_elements=final_elements,
        text_regions=text_regions,
        ocr_enabled=ocr_enabled,
        app_context=app_context,
        original_size=(screenshot.width, screenshot.height),
        resized_size=(resized.width, resized.height),
        region_offset=(0, 0),
    )


def capture_structured_perception(max_width: int = 1280, enriched: bool = False) -> dict[str, Any]:
    """Capture screen and return multimodal payload for reasoning."""
    payload = capture_screen_for_inference(max_width=max_width, enriched=enriched)
    return {
        "image_base64": payload.image_base64,
        "ui_elements": payload.ui_elements,
        "text_regions": payload.text_regions,
        "ocr_enabled": payload.ocr_enabled,
    }
