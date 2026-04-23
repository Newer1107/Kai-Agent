from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, List, Tuple

import pyautogui
from PIL import Image, ImageGrab

from detector import DetectorError, detect_ui_elements
from heuristics import build_hybrid_candidates


@dataclass(frozen=True)
class ScreenInferencePayload:
    """Image payload and coordinate-space metadata used for model inference."""

    image: Image.Image
    image_base64: str
    ui_elements: List[dict[str, Any]]
    text_regions: List[dict[str, Any]]
    ocr_enabled: bool
    original_size: Tuple[int, int]
    resized_size: Tuple[int, int]
    region_offset: Tuple[int, int] = (0, 0)

    @property
    def base64_image(self) -> str:
        """Backward-compatible alias for existing call sites."""
        return self.image_base64


def capture_primary_screenshot() -> Image.Image:
    """Capture the primary monitor using pyautogui, with a Pillow fallback."""
    try:
        return pyautogui.screenshot()
    except Exception:
        return ImageGrab.grab(all_screens=False)


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
    screenshot = capture_primary_screenshot()
    resized = resize_for_inference(screenshot, max_width=max_width)

    try:
        elements = detect_ui_elements(screenshot, max_width=max_width)
    except DetectorError:
        elements = []

    text_regions: List[dict[str, Any]] = []
    final_elements = elements
    ocr_enabled = False

    if enriched:
        if enable_ocr is None:
            enable_ocr = os.getenv("KAI_ENABLE_OCR", "0").strip().lower() in {"1", "true", "yes", "on"}

        try:
            from layout import attach_regions
            from semantics import attach_semantics, extract_text_regions

            regions_attached = attach_regions(elements, (screenshot.width, screenshot.height))
            text_regions = extract_text_regions(screenshot, enabled=enable_ocr)
            final_elements = attach_semantics(regions_attached, text_regions)
            ocr_enabled = bool(enable_ocr)
        except Exception:
            # Fall back to raw detections if enrichment fails.
            text_regions = []
            final_elements = elements
            ocr_enabled = False

    try:
        from layout import attach_regions

        final_elements = build_hybrid_candidates(
            attach_regions(final_elements, (screenshot.width, screenshot.height)),
            (screenshot.width, screenshot.height),
        )
    except Exception:
        final_elements = elements

    return ScreenInferencePayload(
        image=screenshot.copy(),
        image_base64=encode_image_to_base64(resized),
        ui_elements=final_elements,
        text_regions=text_regions,
        ocr_enabled=ocr_enabled,
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
