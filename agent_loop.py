from __future__ import annotations

import math
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Optional

from PIL import Image, ImageChops, ImageStat

from execution import execute_action
from perception import ScreenInferencePayload, capture_primary_screenshot, capture_screen_for_inference
from reasoning import GoalIntent, ParsedIntent, parse_goal, parse_intent
from schema import ActionEnum, UIAction, safe_wait_action
from heuristics import compute_affordances, build_hybrid_candidates


@dataclass(frozen=True)
class LoopState:
    payload: ScreenInferencePayload
    capture_ms: float


@dataclass(frozen=True)
class RankedCandidate:
    element: dict[str, Any]
    score: float
    score_breakdown: dict[str, float]


@dataclass(frozen=True)
class CandidatePlan:
    action: UIAction
    element: dict[str, Any] | None
    score: float
    score_breakdown: dict[str, float]
    source: str
    planned_actions: list[UIAction]


@dataclass(frozen=True)
class DecisionResult:
    action: UIAction
    selected_element: dict[str, Any] | None
    intent: GoalIntent | ParsedIntent
    used_fallback: bool
    decision_ms: float
    reason: str
    ranked_candidates: list[RankedCandidate]
    candidate_plans: list[CandidatePlan]
    planned_actions: list[UIAction]


@dataclass(frozen=True)
class StepResult:
    step_index: int
    decision: DecisionResult
    executed: bool
    step_ms: float


def observe_state(max_width: int = 1280) -> LoopState:
    start = time.perf_counter()
    payload = capture_screen_for_inference(max_width=max_width, enable_ocr=None, enriched=True)
    capture_ms = (time.perf_counter() - start) * 1000.0
    return LoopState(payload=payload, capture_ms=capture_ms)


def _normalized_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    cleaned = "".join(ch if ch.isalnum() else " " for ch in value.lower())
    return [token for token in cleaned.split() if token]


def _goal_mode(goal: str) -> str:
    text = (goal or "").lower()
    if any(token in text for token in ["search", "type", "enter", "input", "find", "write"]):
        return "input"
    if any(token in text for token in ["click", "select", "tap", "submit", "send", "open", "press"]):
        return "click"
    if "scroll" in text:
        return "scroll"
    return "generic"


def _text_similarity(left: str, right: str) -> float:
    left_text = (left or "").strip().lower()
    right_text = (right or "").strip().lower()
    if not left_text or not right_text:
        return 0.0
    if left_text in right_text or right_text in left_text:
        return 1.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _get_bbox(element: dict[str, Any], screen_size: tuple[int, int]) -> tuple[int, int, int, int]:
    bbox = element.get("bbox", [0, 0, 0, 0])
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2

    cx, cy = _get_center(element, fallback=(screen_size[0] // 2, screen_size[1] // 2))
    half_w = max(12, int(screen_size[0] * 0.02))
    half_h = max(10, int(screen_size[1] * 0.015))
    return cx - half_w, cy - half_h, cx + half_w, cy + half_h


def _get_center(element: dict[str, Any], fallback: tuple[int, int]) -> tuple[int, int]:
    center = element.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        try:
            return int(center[0]), int(center[1])
        except Exception:
            return fallback
    return fallback


def _aspect_ratio(element: dict[str, Any], screen_size: tuple[int, int]) -> float:
    x1, y1, x2, y2 = _get_bbox(element, screen_size)
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    return width / height


def _area_ratio(element: dict[str, Any], screen_size: tuple[int, int]) -> float:
    x1, y1, x2, y2 = _get_bbox(element, screen_size)
    area = max(1, (x2 - x1) * (y2 - y1))
    total = max(1, screen_size[0] * screen_size[1])
    return float(area) / float(total)


def _center_bias(element: dict[str, Any], screen_size: tuple[int, int]) -> float:
    width, height = screen_size
    cx, cy = _get_center(element, fallback=(width // 2, height // 2))
    dx = (cx - (width / 2.0)) / max(1.0, width / 2.0)
    dy = (cy - (height / 2.0)) / max(1.0, height / 2.0)
    distance = math.sqrt((dx * dx) + (dy * dy))
    return max(0.0, min(1.0, 1.0 - distance))


def is_probable_input(element: dict[str, Any], screen_size: tuple[int, int]) -> bool:
    aspect = _aspect_ratio(element, screen_size)
    center = _center_bias(element, screen_size)
    area = _area_ratio(element, screen_size)
    return aspect > 4.0 and center > 0.45 and 0.003 <= area <= 0.35


def is_probable_button(element: dict[str, Any], screen_size: tuple[int, int]) -> bool:
    aspect = _aspect_ratio(element, screen_size)
    area = _area_ratio(element, screen_size)
    x1, y1, x2, y2 = _get_bbox(element, screen_size)
    height = max(1, y2 - y1)
    return 1.1 <= aspect <= 6.5 and 0.0007 <= area <= 0.10 and height >= 18


def _goal_alignment(element: dict[str, Any], goal: str, target_hint: str | None = None) -> float:
    tokens = _normalized_tokens(goal)
    hint_tokens = _normalized_tokens(target_hint)
    combined_tokens = list(dict.fromkeys(tokens + hint_tokens))
    if not combined_tokens:
        return 0.0

    haystack = " ".join(
        [
            str(element.get("type", "")),
            str(element.get("semantic_label", "")),
            str(element.get("text", "")),
            str(element.get("region", "")),
        ]
    ).lower()
    if not haystack.strip():
        return 0.0

    matched = sum(1 for token in combined_tokens if token in haystack)
    return max(0.0, min(1.0, matched / max(1, len(combined_tokens))))


def _size_relevance(element: dict[str, Any], mode: str, screen_size: tuple[int, int]) -> float:
    area = _area_ratio(element, screen_size)
    if mode == "input":
        # Inputs are often medium to large horizontal controls.
        return max(0.0, 1.0 - abs(area - 0.03) / 0.05)
    if mode == "click":
        # Click targets are often medium controls.
        return max(0.0, 1.0 - abs(area - 0.01) / 0.03)
    return max(0.0, 1.0 - abs(area - 0.015) / 0.05)


def _shape_bias(element: dict[str, Any], mode: str, screen_size: tuple[int, int]) -> float:
    if mode == "input":
        if is_probable_input(element, screen_size):
            return 1.0
        aspect = _aspect_ratio(element, screen_size)
        return max(0.0, min(1.0, (aspect - 1.5) / 4.0))

    if mode == "click":
        if is_probable_button(element, screen_size):
            return 1.0
        aspect = _aspect_ratio(element, screen_size)
        if 0.8 <= aspect <= 8.0:
            return 0.55
        return 0.2

    return 0.35


def score_element(
    element: dict[str, Any],
    goal: str,
    screen_size: tuple[int, int],
    target_hint: str | None = None,
) -> tuple[float, dict[str, float]]:
    mode = _goal_mode(goal)
    confidence = max(0.0, min(1.0, float(element.get("confidence", 0.0))))
    center = _center_bias(element, screen_size)
    shape = _shape_bias(element, mode, screen_size)
    size = _size_relevance(element, mode, screen_size)
    goal_match = _goal_alignment(element, goal, target_hint=target_hint)
    affordances = compute_affordances(element, screen_size)
    text_similarity = max(
        _text_similarity(goal, str(element.get("text", ""))),
        _text_similarity(goal, str(element.get("semantic_label", ""))),
        _text_similarity(goal, str(element.get("type", ""))),
    )
    affordance_score = float(affordances.get("importance_score", 0.0))

    score = (
        (confidence * 0.3)
        + (center * 0.18)
        + (shape * 0.14)
        + (size * 0.10)
        + (goal_match * 0.10)
        + (text_similarity * 0.12)
        + (affordance_score * 0.06)
    )

    if bool(affordances.get("can_type", False)) and mode == "input":
        score += 0.06
    if bool(affordances.get("can_click", False)) and mode == "click":
        score += 0.03

    score = max(0.0, min(1.0, score))

    return score, {
        "confidence": round(confidence, 4),
        "center_bias": round(center, 4),
        "shape_bias": round(shape, 4),
        "size_relevance": round(size, 4),
        "goal_match": round(goal_match, 4),
        "text_similarity": round(text_similarity, 4),
        "affordance_score": round(affordance_score, 4),
        "can_type": 1.0 if bool(affordances.get("can_type", False)) else 0.0,
    }


def rank_elements(
    elements: list[dict[str, Any]],
    goal: str,
    screen_size: tuple[int, int],
    top_k: int = 8,
    target_hint: str | None = None,
) -> list[RankedCandidate]:
    ranked: list[RankedCandidate] = []
    for element in elements:
        score, breakdown = score_element(element, goal, screen_size, target_hint=target_hint)
        ranked.append(
            RankedCandidate(
                element=dict(element),
                score=score,
                score_breakdown=breakdown,
            )
        )

    ranked.sort(key=lambda item: item.score, reverse=True)
    if top_k > 0:
        return ranked[:top_k]
    return ranked


def _make_center_input(screen_size: tuple[int, int]) -> dict[str, Any]:
    width, height = screen_size
    cx = width // 2
    cy = int(height * 0.35)
    box_w = int(width * 0.6)
    box_h = max(24, int(height * 0.07))

    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(width, x1 + box_w)
    y2 = min(height, y1 + box_h)

    return {
        "type": "input",
        "bbox": [x1, y1, x2, y2],
        "center": [cx, cy],
        "confidence": 0.35,
        "source": "heuristic_center",
    }


def _goal_is_clear(goal: str) -> bool:
    text = (goal or "").lower()
    return any(
        token in text
        for token in [
            "search",
            "type",
            "enter",
            "input",
            "find",
            "click",
            "open",
            "select",
            "submit",
            "send",
            "scroll",
        ]
    )


def _fallback_ranked_candidates(
    intent: ParsedIntent,
    elements: list[dict[str, Any]],
    screen_size: tuple[int, int],
) -> list[RankedCandidate]:
    fallback: list[RankedCandidate] = []

    if intent.action == ActionEnum.TYPE:
        center_input = _make_center_input(screen_size)
        score, breakdown = score_element(
            center_input,
            goal="search input field",
            screen_size=screen_size,
            target_hint="input",
        )
        fallback.append(
            RankedCandidate(
                element=center_input,
                score=max(score, 0.35),
                score_breakdown=breakdown,
            )
        )
        return fallback

    if intent.action == ActionEnum.CLICK and elements:
        best = dict(max(elements, key=lambda item: float(item.get("confidence", 0.0))))
        best["source"] = "heuristic_click"
        best["confidence"] = max(0.35, float(best.get("confidence", 0.0)))
        score, breakdown = score_element(
            best,
            goal="generic click",
            screen_size=screen_size,
            target_hint=intent.target,
        )
        fallback.append(
            RankedCandidate(
                element=best,
                score=max(score, 0.35),
                score_breakdown=breakdown,
            )
        )

    return fallback


def _build_action(
    intent: ParsedIntent,
    selected: dict[str, Any] | None,
    last_action: Optional[UIAction],
    screen_size: tuple[int, int],
    used_fallback: bool,
    goal: str,
    score_breakdown: dict[str, float] | None = None,
) -> UIAction:
    if intent.action == ActionEnum.WAIT:
        return safe_wait_action(
            reason="Intent parser returned wait.",
            intent_summary="No immediate action proposed.",
            next_step_summary="Wait for a clearer goal.",
        )

    center = _get_center(selected or {}, fallback=(screen_size[0] // 2, screen_size[1] // 2))
    if intent.action == ActionEnum.ENTER:
        return UIAction(
            intent_summary="Press enter to confirm the current UI state.",
            next_step_summary="Press enter and verify the result.",
            action=ActionEnum.ENTER,
            target_label=str((selected or {}).get("type", intent.target or "enter")).strip().lower() or "enter",
            target_description=f"Enter key press from {str((selected or {}).get('source', 'rule'))} at {center}",
            target_coordinates=center,
            text_to_type=None,
            confidence_score=max(0.35, min(0.95, float((selected or {}).get("confidence", 0.35)))),
            uncertainty_reason="Enter action is intentionally low-risk and bounded by verification.",
        )

    base_confidence = float((selected or {}).get("confidence", 0.35))
    base_confidence = max(0.1, min(0.95, base_confidence))

    shape_strength = (score_breakdown or {}).get("shape_bias", 0.0)
    goal_strength = (score_breakdown or {}).get("goal_match", 0.0)
    clear_intent = _goal_is_clear(goal)

    confidence = base_confidence
    if shape_strength >= 0.75:
        confidence = max(confidence, 0.52)
    if clear_intent and intent.action in {ActionEnum.CLICK, ActionEnum.TYPE}:
        confidence = max(confidence, 0.35)
    if clear_intent and goal_strength >= 0.5:
        confidence = max(confidence, 0.55)
    if used_fallback:
        confidence = max(confidence, 0.35)
    confidence = max(0.2, min(0.95, confidence))

    target_label = str((selected or {}).get("type", intent.target or "button")).strip().lower() or "button"
    source = str((selected or {}).get("source", "yolo"))
    uncertainty = "Decision chosen from ranked candidates with post-action verification enabled."

    if intent.action == ActionEnum.TYPE:
        if not intent.text:
            return safe_wait_action(
                reason="Type intent has no text payload.",
                intent_summary="Cannot type without text.",
                next_step_summary="Update goal with explicit text.",
            )

        if selected is None:
            return safe_wait_action(
                reason="No target available for type action.",
                intent_summary="No candidate input found.",
                next_step_summary="Observe again and try fallback input location.",
            )

        already_typed = (
            last_action is not None
            and last_action.action == ActionEnum.TYPE
            and (last_action.text_to_type or "") == intent.text
        )
        if already_typed:
            return safe_wait_action(
                reason="Type step already executed for this intent.",
                intent_summary="Likely reached end of immediate goal steps.",
                next_step_summary="Wait for the next user goal.",
            )

        clicked_input_recently = (
            last_action is not None
            and last_action.action == ActionEnum.CLICK
            and (last_action.target_label or "") in {"input", "search_input", "search", "textbox", "text"}
        )

        if clicked_input_recently:
            return UIAction(
                intent_summary="Type parsed text into focused field.",
                next_step_summary="Type intent payload into current input.",
                action=ActionEnum.TYPE,
                target_label=target_label,
                target_description=f"Type target from {source} at {center}",
                target_coordinates=center,
                text_to_type=intent.text,
                confidence_score=confidence,
                uncertainty_reason=uncertainty,
            )

        return UIAction(
            intent_summary="Prepare input focus before typing.",
            next_step_summary="Click inferred input field, then type on next iteration.",
            action=ActionEnum.CLICK,
            target_label="input",
            target_description=f"Focus input from {source} at {center}",
            target_coordinates=center,
            text_to_type=None,
            confidence_score=confidence,
            uncertainty_reason=uncertainty,
        )

    if intent.action == ActionEnum.SCROLL:
        direction = intent.target or "down"
        return UIAction(
            intent_summary="Scroll action parsed from goal.",
            next_step_summary=f"Scroll {direction} and re-observe.",
            action=ActionEnum.SCROLL,
            target_label="scroll_area",
            target_description=f"Scroll {direction} around {center}",
            target_coordinates=center,
            text_to_type=None,
            confidence_score=max(0.5, confidence),
            uncertainty_reason=uncertainty,
        )

    if selected is None:
        return safe_wait_action(
            reason="No clickable target was resolved.",
            intent_summary="No target found for click intent.",
            next_step_summary="Re-observe screen for clearer clickable elements.",
        )

    repeated_click = (
        last_action is not None
        and last_action.action == ActionEnum.CLICK
        and (last_action.target_label or "") == target_label
        and source == "yolo"
    )
    if repeated_click:
        return safe_wait_action(
            reason="Click step already executed for current intent.",
            intent_summary="Likely reached the end of this click-only intent.",
            next_step_summary="Wait for updated UI or a new goal.",
        )

    return UIAction(
        intent_summary="Click target parsed from goal.",
        next_step_summary="Click ranked candidate and verify outcome.",
        action=ActionEnum.CLICK,
        target_label=target_label,
        target_description=f"Click target from {source} at {center}",
        target_coordinates=center,
        text_to_type=None,
        confidence_score=confidence,
        uncertainty_reason=uncertainty,
    )


def _build_enter_action(
    selected: dict[str, Any] | None,
    screen_size: tuple[int, int],
    confidence: float,
    source: str,
) -> UIAction:
    center = _get_center(selected or {}, fallback=(screen_size[0] // 2, screen_size[1] // 2))
    target_label = str((selected or {}).get("type", "enter")).strip().lower() or "enter"
    return UIAction(
        intent_summary="Press enter to commit the current UI state.",
        next_step_summary="Press enter and verify the page response.",
        action=ActionEnum.ENTER,
        target_label=target_label,
        target_description=f"Enter key press from {source} at {center}",
        target_coordinates=center,
        text_to_type=None,
        confidence_score=max(0.35, min(0.95, confidence)),
        uncertainty_reason="Enter is a lightweight confirm step in the planned sequence.",
    )


def _build_planned_actions(
    intent: ParsedIntent,
    primary_action: UIAction,
    selected: dict[str, Any] | None,
    last_action: Optional[UIAction],
    screen_size: tuple[int, int],
    goal: str,
    score_breakdown: dict[str, float] | None = None,
) -> list[UIAction]:
    if primary_action.action == ActionEnum.WAIT:
        return [primary_action]

    confidence = primary_action.confidence_score
    source = str((selected or {}).get("source", "yolo"))
    planned: list[UIAction] = [primary_action]

    if intent.action == ActionEnum.TYPE:
        if primary_action.action == ActionEnum.CLICK:
            typed_action = UIAction(
                intent_summary="Type parsed text into focused field.",
                next_step_summary="Type intent payload into current input.",
                action=ActionEnum.TYPE,
                target_label=str((selected or {}).get("type", intent.target or "input")).strip().lower() or "input",
                target_description=f"Type target from {source} at {_get_center(selected or {}, fallback=(screen_size[0] // 2, screen_size[1] // 2))}",
                target_coordinates=_get_center(selected or {}, fallback=(screen_size[0] // 2, screen_size[1] // 2)),
                text_to_type=intent.text,
                confidence_score=max(confidence, 0.35),
                uncertainty_reason="Type follows the click-to-focus step.",
            )
            planned.append(typed_action)
        if intent.text:
            planned.append(_build_enter_action(selected, screen_size, confidence, source))
        return planned

    if intent.action == ActionEnum.CLICK and any(keyword in goal.lower() for keyword in ["submit", "send", "login", "continue", "next"]):
        planned.append(_build_enter_action(selected, screen_size, confidence, source))
        return planned

    if intent.action == ActionEnum.ENTER:
        return [_build_enter_action(selected, screen_size, confidence, source)]

    return planned


def _build_candidate_plans(
    intent: ParsedIntent,
    ranked_candidates: list[RankedCandidate],
    goal: str,
    screen_size: tuple[int, int],
    last_action: Optional[UIAction],
    top_k: int = 3,
) -> list[CandidatePlan]:
    plans: list[CandidatePlan] = []

    for ranked in ranked_candidates[: max(1, top_k)]:
        element = dict(ranked.element)
        source = str(element.get("source", "yolo"))
        used_fallback = source.startswith("heuristic")

        primary_action = _build_action(
            intent=intent,
            selected=element,
            last_action=last_action,
            screen_size=screen_size,
            used_fallback=used_fallback,
            goal=goal,
            score_breakdown=ranked.score_breakdown,
        )
        planned_actions = _build_planned_actions(
            intent=intent,
            primary_action=primary_action,
            selected=element,
            last_action=last_action,
            screen_size=screen_size,
            goal=goal,
            score_breakdown=ranked.score_breakdown,
        )

        plans.append(
            CandidatePlan(
                action=primary_action,
                element=element,
                score=ranked.score,
                score_breakdown=ranked.score_breakdown,
                source=source,
                planned_actions=planned_actions,
            )
        )

    return plans


def build_retry_fallback_plan(
    state: LoopState,
    goal: str,
    last_action: Optional[UIAction] = None,
) -> CandidatePlan | None:
    intent = parse_intent(goal)
    screen_size = state.payload.original_size
    fallback_ranked = _fallback_ranked_candidates(intent, list(state.payload.ui_elements or []), screen_size)
    if not fallback_ranked:
        return None

    ranked = fallback_ranked[0]
    primary_action = _build_action(
        intent=intent,
        selected=ranked.element,
        last_action=last_action,
        screen_size=screen_size,
        used_fallback=True,
        goal=goal,
        score_breakdown=ranked.score_breakdown,
    )
    planned_actions = _build_planned_actions(
        intent=intent,
        primary_action=primary_action,
        selected=ranked.element,
        last_action=last_action,
        screen_size=screen_size,
        goal=goal,
        score_breakdown=ranked.score_breakdown,
    )

    return CandidatePlan(
        action=primary_action,
        element=dict(ranked.element),
        score=ranked.score,
        score_breakdown=ranked.score_breakdown,
        source=str(ranked.element.get("source", "heuristic")),
        planned_actions=planned_actions,
    )


def _ui_signature_from_image(image: Image.Image) -> tuple[int, ...]:
    gray = image.convert("L").resize((72, 40))
    data = list(gray.getdata())

    row_sums: list[int] = []
    width = 72
    height = 40
    for row in range(height):
        start = row * width
        row_data = data[start : start + width]
        row_sums.append(sum(row_data) // width)

    col_sums: list[int] = []
    for col in range(width):
        total = 0
        for row in range(height):
            total += data[(row * width) + col]
        col_sums.append(total // height)

    # Quantize to reduce noise.
    compressed = tuple((value // 8) for value in (row_sums[::4] + col_sums[::8]))
    return compressed


def _local_change_ratio(
    prev_image: Image.Image,
    new_image: Image.Image,
    center: tuple[int, int],
    radius: int = 26,
) -> float:
    x, y = center
    width, height = prev_image.size

    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(width, x + radius)
    y2 = min(height, y + radius)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    prev_crop = prev_image.convert("L").crop((x1, y1, x2, y2))
    new_crop = new_image.convert("L").crop((x1, y1, x2, y2))
    diff = ImageChops.difference(prev_crop, new_crop)
    stat = ImageStat.Stat(diff)
    mean_value = float(stat.mean[0]) if stat.mean else 0.0
    return mean_value / 255.0


def screen_changed(prev_image: Image.Image, new_image: Image.Image, threshold: float = 0.01) -> bool:
    prev_small = prev_image.convert("L").resize((96, 54))
    new_small = new_image.convert("L").resize((96, 54))
    diff = ImageChops.difference(prev_small, new_small)
    stat = ImageStat.Stat(diff)
    mean_value = float(stat.mean[0]) if stat.mean else 0.0
    ratio = mean_value / 255.0
    return ratio > threshold


def verify_success(
    prev_image: Image.Image,
    new_image: Image.Image,
    action: UIAction,
    prev_elements: list[dict[str, Any]] | None = None,
    new_elements: list[dict[str, Any]] | None = None,
) -> bool:
    base_threshold = 0.01
    typing_threshold = 0.003

    changed = screen_changed(
        prev_image,
        new_image,
        threshold=typing_threshold if action.action == ActionEnum.TYPE else base_threshold,
    )

    ui_changed = False
    if prev_elements is not None and new_elements is not None:
        prev_sig = tuple(
            (
                str(item.get("type", "")).lower(),
                int(item.get("center", [0, 0])[0] // 24),
                int(item.get("center", [0, 0])[1] // 24),
            )
            for item in prev_elements[:12]
        )
        new_sig = tuple(
            (
                str(item.get("type", "")).lower(),
                int(item.get("center", [0, 0])[0] // 24),
                int(item.get("center", [0, 0])[1] // 24),
            )
            for item in new_elements[:12]
        )
        ui_changed = prev_sig != new_sig
    else:
        ui_changed = _ui_signature_from_image(prev_image) != _ui_signature_from_image(new_image)

    local_focus_changed = False
    if action.action == ActionEnum.TYPE and action.target_coordinates is not None:
        cx, cy = int(action.target_coordinates[0]), int(action.target_coordinates[1])
        local_focus_changed = _local_change_ratio(prev_image, new_image, (cx, cy), radius=28) > 0.01

    success = changed or ui_changed or local_focus_changed
    print(
        "[VERIFY] "
        f"action={action.action.value} changed={changed} "
        f"ui_changed={ui_changed} focus_changed={local_focus_changed} success={success}"
    )
    return success


def decide_action(
    state: LoopState,
    goal: str,
    last_action: Optional[UIAction] = None,
    parsed_intent: ParsedIntent | None = None,
) -> DecisionResult:
    start = time.perf_counter()
    intent = parsed_intent or parse_goal(goal)
    screen_size = state.payload.original_size
    elements = list(state.payload.ui_elements or [])

    if intent.action == ActionEnum.WAIT:
        wait_action = safe_wait_action(
            reason="Intent parser returned wait.",
            intent_summary="No actionable intent found.",
            next_step_summary="Wait for clearer goal input.",
        )
        return DecisionResult(
            action=wait_action,
            selected_element=None,
            intent=intent,
            used_fallback=False,
            decision_ms=(time.perf_counter() - start) * 1000.0,
            reason="intent_wait",
            ranked_candidates=[],
            candidate_plans=[],
            planned_actions=[wait_action],
        )

    ranked = rank_elements(
        elements=elements,
        goal=goal,
        screen_size=screen_size,
        top_k=8,
        target_hint=intent.target,
    )

    used_fallback = False
    reason = "ranked_candidates"

    if not ranked and intent.action in {ActionEnum.CLICK, ActionEnum.TYPE}:
        ranked = _fallback_ranked_candidates(intent, elements, screen_size)
        used_fallback = bool(ranked)
        reason = "heuristic_fallback" if used_fallback else "no_candidates"

    candidate_plans = _build_candidate_plans(
        intent=intent,
        ranked_candidates=ranked,
        goal=goal,
        screen_size=screen_size,
        last_action=last_action,
        top_k=3,
    )

    selected_plan = next((plan for plan in candidate_plans if plan.action.action != ActionEnum.WAIT), None)
    if selected_plan is None:
        wait_action = safe_wait_action(
            reason="No executable candidate produced a non-wait action.",
            intent_summary="Could not determine a robust next step.",
            next_step_summary="Observe again and retry with updated state.",
        )
        action = wait_action
        selected_element = None
        planned_actions = [wait_action]
        reason = "all_candidates_wait"
    else:
        action = selected_plan.action
        selected_element = selected_plan.element
        planned_actions = selected_plan.planned_actions

    decision_ms = (time.perf_counter() - start) * 1000.0

    print(
        "[DECISION] "
        f"goal='{goal}' intent={intent.action.value} candidates={len(ranked)} "
        f"used_fallback={used_fallback}"
    )
    for idx, ranked_candidate in enumerate(ranked[:3], start=1):
        center = _get_center(ranked_candidate.element, fallback=(screen_size[0] // 2, screen_size[1] // 2))
        print(
            "[DECISION] "
            f"#{idx} type={ranked_candidate.element.get('type')} center={center} "
            f"score={ranked_candidate.score:.3f} breakdown={ranked_candidate.score_breakdown}"
        )

    print("[PLAN]")
    for idx, planned_action in enumerate(planned_actions, start=1):
        print(
            f"[PLAN] Step {idx}: {planned_action.action.value} "
            f"target={planned_action.target_label} coords={planned_action.target_coordinates} "
            f"text={planned_action.text_to_type!r} conf={planned_action.confidence_score:.2f}"
        )

    return DecisionResult(
        action=action,
        selected_element=selected_element,
        intent=intent,
        used_fallback=used_fallback,
        decision_ms=decision_ms,
        reason=reason,
        ranked_candidates=ranked,
        candidate_plans=candidate_plans,
        planned_actions=planned_actions,
    )


def run_agent(
    goal: str,
    max_steps: int = 3,
    pause_seconds: float = 0.5,
    approval_callback: Callable[[int, UIAction, dict[str, Any] | None], bool] | None = None,
    after_step_callback: Callable[[StepResult], None] | None = None,
) -> list[StepResult]:
    results: list[StepResult] = []
    parsed_intent = parse_goal(goal)
    last_action: Optional[UIAction] = None

    for step in range(max(1, max_steps)):
        step_start = time.perf_counter()
        state = observe_state(max_width=1280)
        decision = decide_action(
            state=state,
            goal=goal,
            last_action=last_action,
            parsed_intent=parsed_intent,
        )

        if decision.action.action == ActionEnum.WAIT:
            step_ms = (time.perf_counter() - step_start) * 1000.0
            result = StepResult(step_index=step, decision=decision, executed=False, step_ms=step_ms)
            results.append(result)
            if after_step_callback is not None:
                after_step_callback(result)
            break

        executed_success = False
        candidate_plans = decision.candidate_plans[:3]

        for idx, plan in enumerate(candidate_plans, start=1):
            if plan.action.action == ActionEnum.WAIT:
                continue

            approved = True
            if approval_callback is not None:
                approved = bool(approval_callback(step, plan.action, plan.element))
            if not approved:
                break

            planned_actions = [item for item in plan.planned_actions if item.action != ActionEnum.WAIT]
            if not planned_actions:
                planned_actions = [plan.action]

            print(f"[ACTION] executing candidate #{idx}/{len(candidate_plans)} score={plan.score:.3f}")
            candidate_success = True
            current_image = state.payload.image
            current_elements = state.payload.ui_elements

            for candidate_action in planned_actions:
                executed = execute_action(
                    candidate_action,
                    selected_element=plan.element,
                    original_size=state.payload.original_size,
                    resized_size=state.payload.resized_size,
                )
                if not executed:
                    print(f"[RETRY] candidate #{idx} execution failed")
                    candidate_success = False
                    break

                new_image = capture_primary_screenshot()
                if not verify_success(current_image, new_image, candidate_action, prev_elements=current_elements):
                    print(f"[RETRY] candidate #{idx} failed verification; trying next candidate")
                    candidate_success = False
                    break

                current_image = new_image
                last_action = candidate_action

            if candidate_success:
                executed_success = True
                break

        if not executed_success:
            fallback_plan = build_retry_fallback_plan(state=state, goal=goal, last_action=last_action)
            if fallback_plan is not None and fallback_plan.action.action != ActionEnum.WAIT:
                print("[RETRY] fallback triggered after candidate failures")
                fallback_actions = [item for item in fallback_plan.planned_actions if item.action != ActionEnum.WAIT]
                if not fallback_actions:
                    fallback_actions = [fallback_plan.action]
                executed_success = True
                current_image = state.payload.image
                current_elements = state.payload.ui_elements
                for fallback_action in fallback_actions:
                    executed = execute_action(
                        fallback_action,
                        selected_element=fallback_plan.element,
                        original_size=state.payload.original_size,
                        resized_size=state.payload.resized_size,
                    )
                    if not executed:
                        executed_success = False
                        break
                    new_image = capture_primary_screenshot()
                    if not verify_success(current_image, new_image, fallback_action, prev_elements=current_elements):
                        executed_success = False
                        break
                    current_image = new_image
                    last_action = fallback_action
                if not executed_success:
                    print("[RETRY] fallback failed verification")

        step_ms = (time.perf_counter() - step_start) * 1000.0
        result = StepResult(step_index=step, decision=decision, executed=executed_success, step_ms=step_ms)
        results.append(result)

        if after_step_callback is not None:
            after_step_callback(result)

        if not executed_success:
            break

        time.sleep(max(0.0, pause_seconds))

    return results
