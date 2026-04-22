from __future__ import annotations

import json
import os
from typing import Any, Optional

import instructor
from openai import OpenAI

from schema import ActionEnum, UIAction, safe_wait_action
from heuristics import detect_by_goal_heuristic, is_safe_heuristic_location
from policy import get_action_risk, is_intent_clear, is_action_allowed


SYSTEM_PROMPT = """
You are a cautious but capable desktop assistant.

You are given:
- A screenshot of the current screen
- Structured UI elements detected by YOLO (type, bbox, center, confidence, region, semantic labels, nearby text)
- A user goal (optional)
- The last action taken (optional)

Your job:
1. Use the provided UI elements as the source of truth for available targets
2. If a goal exists, determine the NEXT BEST STEP toward that goal
3. If no goal exists, suggest a useful but safe action
4. If exact elements cannot be found, infer the most likely UI target based on layout, position, and goal
5. If uncertain, return action='wait'

Rules:
- Only suggest ONE step
- Do NOT assume hidden intent
- Base decisions ONLY on visible evidence and provided UI elements
- You MUST choose a target based on type, expected location, and semantic meaning when available
- You are allowed to choose approximate UI targets when structure and goal strongly suggest intent
- If multiple plausible candidates exist, explain ambiguity and return action='wait'
- If confidence < 0.7, include uncertainty_reason
- If no clear action exists, return 'wait'
- Do not output coordinates. Coordinates are handled by the vision subsystem.

Output requirements:
- action: click | type | scroll | wait
- target_label: short label for the UI element category (or null for wait)
- target_description: human-readable details of the target
- text_to_type: required when action is type
- confidence_score: 0.0 to 1.0
- uncertainty_reason: required when confidence_score < 0.7

You must output valid JSON matching the schema.
""".strip()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    raw = text[start : end + 1]
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def heuristic_action_fallback(
    goal: str,
    ui_elements: list[dict[str, Any]],
    screen_size: tuple[int, int] | None = None,
) -> UIAction:
    """
    Emergency fallback when LLM fails or detection is weak.
    Forces reasonable action based on goal intent.
    """
    goal_lower = goal.lower() if goal else ""
    
    print("[FALLBACK] Emergency heuristic fallback activated")
    print(f"[FALLBACK] Goal: {goal}")
    print(f"[FALLBACK] UI Elements available: {len(ui_elements)}")
    
    # Search/type/enter intent: find or infer input
    if any(kw in goal_lower for kw in ["search", "type", "enter", "find", "input"]):
        print("[FALLBACK] Search/type intent detected")
        
        # Try to find a wide element in center
        for element in ui_elements:
            region = str(element.get("region", "")).lower()
            if region == "center":
                bbox = element.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                width = x2 - x1
                height = y2 - y1
                if height > 0 and width / height > 2.5:
                    print(f"[FALLBACK] Found wide CENTER element: {element.get('type')}")
                    element_copy = dict(element)
                    element_copy["source"] = "heuristic_emergency"
                    element_copy["confidence"] = 0.65
                    return UIAction(
                        intent_summary="Search/input detected from goal.",
                        next_step_summary=f"Click inferred input field and prepare to type.",
                        action=ActionEnum.CLICK,
                        target_label="input",
                        target_description=f"Heuristic input at {element_copy.get('center')}",
                        target_coordinates=None,
                        text_to_type=None,
                        confidence_score=0.65,
                        uncertainty_reason="Using heuristic fallback with relaxed confidence.",
                    )
        
        # Fallback: use center screen position
        if screen_size:
            cx = screen_size[0] // 2
            cy = int(screen_size[1] * 0.35)
            print(f"[FALLBACK] Using center screen heuristic: ({cx}, {cy})")
            return UIAction(
                intent_summary="Search/input intent from goal with no specific element found.",
                next_step_summary="Click center screen search position.",
                action=ActionEnum.CLICK,
                target_label="input",
                target_description=f"Heuristic center-screen input at ({cx}, {cy})",
                target_coordinates=None,
                text_to_type=None,
                confidence_score=0.65,
                uncertainty_reason="Using emergency center-screen heuristic with relaxed confidence.",
            )
    
    # Submit/send intent
    if any(kw in goal_lower for kw in ["submit", "send", "enter", "login", "sign"]):
        print("[FALLBACK] Submit/send intent detected")
        for element in ui_elements:
            etype = str(element.get("type", "")).lower()
            if "button" in etype or "submit" in etype:
                element_copy = dict(element)
                element_copy["source"] = "heuristic_emergency"
                element_copy["confidence"] = max(0.55, float(element.get("confidence", 0.55)))
                print(f"[FALLBACK] Found button: {etype}")
                return UIAction(
                    intent_summary="Submit action from goal.",
                    next_step_summary="Click submit button.",
                    action=ActionEnum.CLICK,
                    target_label="button",
                    target_description=f"Submit button at {element_copy.get('center')}",
                    target_coordinates=None,
                    text_to_type=None,
                    confidence_score=0.65,
                    uncertainty_reason="Using heuristic fallback with relaxed confidence.",
                )
    
    # Click intent: pick best available element
    if any(kw in goal_lower for kw in ["click", "select", "tap"]):
        print("[FALLBACK] Click intent with available elements")
        if ui_elements:
            best = max(ui_elements, key=lambda e: float(e.get("confidence", 0.0)))
            element_copy = dict(best)
            element_copy["source"] = "heuristic_emergency"
            confidence = max(0.55, float(best.get("confidence", 0.55)))
            print(f"[FALLBACK] Using highest-confidence element: {best.get('type')} @ {confidence:.2f}")
            return UIAction(
                intent_summary="Click action from goal.",
                next_step_summary=f"Click {best.get('type')} element.",
                action=ActionEnum.CLICK,
                target_label=str(best.get("type", "button")).lower(),
                target_description=f"Best available element at {best.get('center')}",
                target_coordinates=None,
                text_to_type=None,
                confidence_score=0.65,
                uncertainty_reason="Using emergency highest-confidence fallback with relaxed confidence.",
            )
    
    # Fallback: if ANY goal exists and screen is not empty, take action
    if goal and ui_elements:
        print("[FALLBACK] Goal exists with elements available, forcing click on best element")
        best = max(ui_elements, key=lambda e: float(e.get("confidence", 0.0)))
        element_copy = dict(best)
        element_copy["source"] = "heuristic_emergency"
        return UIAction(
            intent_summary=f"Goal-driven action: {goal}",
            next_step_summary=f"Click {best.get('type')}.",
            action=ActionEnum.CLICK,
            target_label=str(best.get("type", "button")).lower(),
            target_description=f"Best element: {best.get('type')} at {best.get('center')}",
            target_coordinates=None,
            text_to_type=None,
            confidence_score=0.65,
            uncertainty_reason="Emergency heuristic: goal-driven action with available elements and relaxed confidence.",
        )
    
    # Absolute last resort: wait
    print("[FALLBACK] No goal or elements; defaulting to wait")
    return safe_wait_action(
        reason="Heuristic fallback exhausted; no clear intent or UI elements available.",
        intent_summary="Unable to determine action from goal or screen.",
        next_step_summary="Wait for clearer intent or screen state.",
    )


def _prepare_ui_elements_for_prompt(
    ui_elements: list[dict[str, Any]],
    max_items: int = 60,
) -> list[dict[str, Any]]:
    # Keep the prompt compact and deterministic while preserving full structure.
    sorted_items = sorted(
        ui_elements,
        key=lambda item: float(item.get("confidence", 0.0)),
        reverse=True,
    )
    prepared: list[dict[str, Any]] = []
    for item in sorted_items[:max_items]:
        prepared.append(
            {
                "type": str(item.get("type", "")).strip(),
                "bbox": [int(v) for v in item.get("bbox", [0, 0, 0, 0])],
                "center": [int(v) for v in item.get("center", [0, 0])],
                "confidence": round(float(item.get("confidence", 0.0)), 4),
                "region": str(item.get("region", "")).strip() or None,
                "semantic_label": str(item.get("semantic_label", "")).strip() or None,
                "text": str(item.get("text", "")).strip() or None,
            }
        )
    return prepared


def _coerce_action_payload(
    payload: dict[str, Any],
    detected_types: set[str],
) -> UIAction:
    action_raw = str(payload.get("action", "wait")).strip().lower()
    allowed_actions = {"click", "type", "scroll", "wait"}
    if action_raw not in allowed_actions:
        action_raw = "wait"

    target_label: str | None = payload.get("target_label")
    if target_label is not None:
        target_label = str(target_label).strip() or None

    text_to_type: str | None = payload.get("text_to_type")
    if text_to_type is not None:
        text_to_type = str(text_to_type)

    confidence = payload.get("confidence_score", 0.55)
    try:
        confidence_score = float(confidence)
    except Exception:
        confidence_score = 0.55
    confidence_score = max(0.0, min(1.0, confidence_score))

    if action_raw != "wait":
        if not target_label or target_label.lower() not in detected_types:
            action_raw = "wait"
            target_label = None
            confidence_score = min(confidence_score, 0.59)

    if action_raw == "type" and not text_to_type:
        action_raw = "wait"
        target_label = None
        confidence_score = min(confidence_score, 0.59)

    uncertainty_reason = payload.get("uncertainty_reason")
    if uncertainty_reason is not None:
        uncertainty_reason = str(uncertainty_reason)

    if confidence_score < 0.7 and not uncertainty_reason:
        uncertainty_reason = "Low-confidence response after fallback parsing."

    return UIAction(
        intent_summary=str(payload.get("intent_summary", "One-step action proposal generated.")),
        next_step_summary=str(payload.get("next_step_summary", "Execute one safe step or wait.")),
        action=ActionEnum(action_raw),
        target_label=target_label,
        target_description=(
            str(payload.get("target_description"))
            if payload.get("target_description") is not None
            else None
        ),
        target_coordinates=None,
        text_to_type=text_to_type,
        confidence_score=confidence_score,
        uncertainty_reason=uncertainty_reason,
    )

class LLMReasoner:
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        raw_client = OpenAI(
            base_url=base_url,
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )
        self.raw_client = raw_client
        self.client = instructor.from_openai(raw_client, mode=instructor.Mode.JSON)

    def analyze_screen(
        self,
        base64_image: str,
        ui_elements: list[dict[str, Any]] | None = None,
        current_goal: Optional[str] = None,
        last_action: Optional[UIAction] = None,
        screen_size: tuple[int, int] | None = None,
    ) -> UIAction:
        """Ask the local model for one structured, conservative next step."""
        last_error: Optional[Exception] = None

        goal_text = current_goal.strip() if current_goal else "No current goal provided."
        print(f"\n[REASONING] Analyzing screen. Goal: {goal_text}")
        print(f"[REASONING] UI Elements: {len(ui_elements or [])}")
        
        if last_action is None:
            last_action_text = "No previous action."
        else:
            last_action_text = last_action.model_dump_json()

        safe_ui_elements = _prepare_ui_elements_for_prompt(ui_elements or [])
        ui_elements_json = json.dumps(safe_ui_elements, ensure_ascii=True)
        detected_types = {str(item.get("type", "")).strip().lower() for item in safe_ui_elements}

        for attempt in range(2):  # one retry after an invalid output
            try:
                print(f"[REASONING] LLM attempt {attempt + 1}/2")
                action = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    max_retries=0,
                    response_model=UIAction,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Analyze this screenshot and propose one safe next step.\n"
                                        f"Current goal: {goal_text}\n"
                                        f"Last action: {last_action_text}\n"
                                        "Detected UI elements (JSON):\n"
                                        f"{ui_elements_json}\n"
                                        "Important: choose a target_label that exists in the detected UI element types. "
                                        "If there is no suitable element, return action='wait'."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{base64_image}",
                                    },
                                },
                            ],
                        },
                    ],
                )
                result = self._apply_safety_overrides(action, safe_ui_elements, goal_text)
                print(f"[REASONING] LLM returned: {result.action.value}")
                return result
            except Exception as exc:
                last_error = exc
                print(f"[REASONING] LLM attempt {attempt + 1} failed: {type(exc).__name__}")
                if attempt == 0:
                    continue

        # Fallback 1: Raw JSON extraction
        print("[REASONING] Attempting JSON fallback...")
        try:
            fallback = self.raw_client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                        + "\nReturn only one JSON object. No markdown. No prose.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Return one JSON object for UIAction.\n"
                                    f"Current goal: {goal_text}\n"
                                    f"Last action: {last_action_text}\n"
                                    "Detected UI elements (JSON):\n"
                                    f"{ui_elements_json}\n"
                                    "You must choose from detected types or return wait."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    },
                ],
            )
            content = fallback.choices[0].message.content or ""
            payload = _extract_json_object(content)
            if payload is not None:
                print("[REASONING] JSON fallback succeeded")
                action = _coerce_action_payload(payload, detected_types)
                return self._apply_safety_overrides(action, safe_ui_elements, goal_text)
        except Exception as exc:
            print(f"[REASONING] JSON fallback failed: {type(exc).__name__}")
            last_error = exc

        # Fallback 2: Emergency heuristic fallback (final resort)
        print("[REASONING] Activating emergency heuristic fallback")
        try:
            heuristic_action = heuristic_action_fallback(
                goal=goal_text,
                ui_elements=safe_ui_elements,
                screen_size=screen_size,
            )
            print(f"[REASONING] Heuristic fallback returned: {heuristic_action.action.value}")
            return heuristic_action
        except Exception as exc:
            print(f"[REASONING] Heuristic fallback failed: {type(exc).__name__}: {exc}")

        error_name = type(last_error).__name__ if last_error else "UnknownError"
        print(f"[REASONING] All fallbacks exhausted; returning wait. Error: {error_name}")
        return safe_wait_action(
            reason=f"All reasoning attempts failed ({error_name}).",
            intent_summary="Could not produce a reliable action from the screenshot.",
            next_step_summary="Wait and request a fresh trigger for another analysis attempt.",
        )

    @staticmethod
    def _apply_safety_overrides(
        action: UIAction,
        ui_elements: list[dict[str, Any]],
        goal: str = "",
    ) -> UIAction:
        """Apply risk-based safety checks with goal-aware relaxation."""
        action = action.model_copy(update={"target_coordinates": None})

        has_clear_intent = is_intent_clear(goal)

        print(f"[DECISION] Action: {action.action.value} | Confidence: {action.confidence_score:.2f} | Intent clear: {has_clear_intent} | Goal: {goal}")
        
        # RISK-BASED POLICY: Use policy.py to determine if action is allowed
        source = "heuristic" if (action.target_label and "heuristic" in str(action.target_label)) else "yolo"
        action_allowed = is_action_allowed(
            action=action.action.value,
            confidence=action.confidence_score,
            source=source,
            goal=goal,
        )
        
        if not action_allowed and action.action != ActionEnum.WAIT:
            print(f"[DECISION] Action rejected by risk-based policy. Downgrading to wait.")
            return UIAction(
                intent_summary=action.intent_summary,
                next_step_summary="Pause and wait for clearer visual evidence before acting.",
                action=ActionEnum.WAIT,
                target_label=None,
                target_description=None,
                target_coordinates=None,
                text_to_type=None,
                confidence_score=action.confidence_score,
                uncertainty_reason=(
                    action.uncertainty_reason
                    or "Action risk exceeds confidence level."
                ),
            )

        # Non-wait actions must provide a target label for vision grounding.
        if action.action != ActionEnum.WAIT and not action.target_label:
            print(f"[DECISION] No target label provided. Downgrading to wait.")
            return UIAction(
                intent_summary=action.intent_summary,
                next_step_summary="Wait because no target label was provided for visual grounding.",
                action=ActionEnum.WAIT,
                target_label=None,
                target_description=None,
                target_coordinates=None,
                text_to_type=None,
                confidence_score=min(action.confidence_score, 0.59),
                uncertainty_reason=(
                    action.uncertainty_reason
                    or "No target_label was provided for a non-wait action."
                ),
            )

        if action.action != ActionEnum.WAIT and action.target_label:
            detected_types = {str(item.get("type", "")).strip().lower() for item in ui_elements}
            wanted = action.target_label.strip().lower()
            if wanted not in detected_types:
                if has_clear_intent:
                    # With clear intent, pick best available instead of wait
                    print(f"[SAFETY] No exact match for {wanted}, but intent is clear: {goal}")
                    print(f"[SAFETY] Available types: {detected_types}")
                    # Downgrade confidence but still allow action
                    action = action.model_copy(
                        update={
                            "confidence_score": max(0.35, action.confidence_score - 0.15),
                            "uncertainty_reason": f"Target '{wanted}' not found; using heuristic fallback due to clear intent.",
                        }
                    )
                    return action
                
                # Otherwise wait
                return UIAction(
                    intent_summary=action.intent_summary,
                    next_step_summary="Wait because no matching detected UI element exists.",
                    action=ActionEnum.WAIT,
                    target_label=None,
                    target_description=None,
                    target_coordinates=None,
                    text_to_type=None,
                    confidence_score=min(action.confidence_score, 0.59),
                    uncertainty_reason=(
                        action.uncertainty_reason
                        or f"No detected UI element matches target_label='{action.target_label}'."
                    ),
                )

        return action


_DEFAULT_REASONER: Optional[LLMReasoner] = None


def _get_default_reasoner() -> LLMReasoner:
    global _DEFAULT_REASONER
    if _DEFAULT_REASONER is None:
        _DEFAULT_REASONER = LLMReasoner(
            model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )
    return _DEFAULT_REASONER


def analyze_screen(
    base64_image: str,
    ui_elements: list[dict[str, Any]] | None = None,
    current_goal: Optional[str] = None,
    last_action: Optional[UIAction] = None,
    screen_size: tuple[int, int] | None = None,
) -> UIAction:
    return _get_default_reasoner().analyze_screen(
        base64_image=base64_image,
        ui_elements=ui_elements,
        current_goal=current_goal,
        last_action=last_action,
        screen_size=screen_size,
    )
