from __future__ import annotations

import json
import os
from typing import Any, Optional

import instructor
from openai import OpenAI

from schema import ActionEnum, UIAction, safe_wait_action


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
    ) -> UIAction:
        """Ask the local model for one structured, conservative next step."""
        last_error: Optional[Exception] = None

        goal_text = current_goal.strip() if current_goal else "No current goal provided."
        if last_action is None:
            last_action_text = "No previous action."
        else:
            last_action_text = last_action.model_dump_json()

        safe_ui_elements = _prepare_ui_elements_for_prompt(ui_elements or [])
        ui_elements_json = json.dumps(safe_ui_elements, ensure_ascii=True)
        detected_types = {str(item.get("type", "")).strip().lower() for item in safe_ui_elements}

        for attempt in range(2):  # one retry after an invalid output
            try:
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
                return self._apply_safety_overrides(action, safe_ui_elements)
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    continue

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
                action = _coerce_action_payload(payload, detected_types)
                return self._apply_safety_overrides(action, safe_ui_elements)
        except Exception:
            pass

        error_name = type(last_error).__name__ if last_error else "UnknownError"
        return safe_wait_action(
            reason=f"Model output invalid after retry ({error_name}).",
            intent_summary="Could not produce a reliable action from the screenshot.",
            next_step_summary="Wait and request a fresh trigger for another analysis attempt.",
        )

    @staticmethod
    def _apply_safety_overrides(
        action: UIAction,
        ui_elements: list[dict[str, Any]],
    ) -> UIAction:
        # Coordinates from the LLM are ignored; grounding is delegated to YOLO.
        action = action.model_copy(update={"target_coordinates": None})

        # Low-confidence actions are downgraded to wait.
        if action.confidence_score < 0.6 and action.action != ActionEnum.WAIT:
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
                    or "Confidence below 0.6, so the system defaulted to wait."
                ),
            )

        # Non-wait actions must provide a target label for vision grounding.
        if action.action != ActionEnum.WAIT and not action.target_label:
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
) -> UIAction:
    return _get_default_reasoner().analyze_screen(
        base64_image=base64_image,
        ui_elements=ui_elements,
        current_goal=current_goal,
        last_action=last_action,
    )
