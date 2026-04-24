from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI

from schema import ActionEnum, UIAction, safe_wait_action


_INTENT_CACHE: dict[str, "GoalIntent"] = {}


@dataclass(frozen=True)
class ParsedIntent:
    action: ActionEnum
    text: str | None = None
    target: str | None = None
    source: str = "rule"


@dataclass(frozen=True)
class GoalIntent:
    action: ActionEnum
    text: str | None = None
    target: str | None = None
    open_app: bool = False
    source: str = "rule"


def _normalize_goal(goal: str) -> str:
    return " ".join(goal.lower().strip().split())


def clear_intent_cache(goal: str | None = None) -> None:
    if goal is None:
        _INTENT_CACHE.clear()
        return
    _INTENT_CACHE.pop(_normalize_goal(goal), None)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None

    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return None

    if isinstance(payload, dict):
        return payload
    return None


def _clean_text(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    return " ".join(cleaned.split())


def _extract_quoted_text(goal: str) -> str | None:
    match = re.search(r'"([^"]+)"|\'([^\']+)\'', goal)
    if not match:
        return None
    value = match.group(1) or match.group(2)
    return _clean_text(value)


def _strip_goal_prefix(goal: str, keywords: tuple[str, ...]) -> str:
    lowered = goal.lower().strip()
    for keyword in keywords:
        if lowered.startswith(keyword):
            return _clean_text(goal[len(keyword) :])
    return _clean_text(goal)


def _rule_parse_goal(goal: str) -> GoalIntent:
    raw = goal.strip()
    if not raw:
        return GoalIntent(action=ActionEnum.WAIT, source="rule")

    goal_lower = raw.lower()
    quoted_text = _extract_quoted_text(raw)

    if any(phrase in goal_lower for phrase in ["press enter", "hit enter", "submit and enter"]):
        return GoalIntent(action=ActionEnum.ENTER, target="enter", source="rule")

    open_match = re.search(r"(?:open|launch|start)\s+(.+)", raw, flags=re.IGNORECASE)
    if open_match:
        target = quoted_text or _strip_goal_prefix(open_match.group(1), ("the ", "app "))
        if target:
            return GoalIntent(action=ActionEnum.CLICK, target=target, open_app=True, source="rule")

    search_match = re.search(r"(?:search|find|look for)(?:\s+for)?\s+(.+)", raw, flags=re.IGNORECASE)
    if search_match:
        query = quoted_text or _clean_text(search_match.group(1))
        if query:
            return GoalIntent(
                action=ActionEnum.TYPE,
                text=query,
                target="search_input",
                source="rule",
            )
        return GoalIntent(action=ActionEnum.CLICK, target="search_input", source="rule")

    type_match = re.search(r"(?:type|enter|input|write)\s+(.+)", raw, flags=re.IGNORECASE)
    if type_match:
        typed = quoted_text or _clean_text(type_match.group(1))
        if typed:
            return GoalIntent(
                action=ActionEnum.TYPE,
                text=typed,
                target="input",
                source="rule",
            )

    click_match = re.search(r"(?:click|open|select|tap|press)\s+(.+)", raw, flags=re.IGNORECASE)
    if click_match:
        target = _clean_text(click_match.group(1))
        if target.lower() in {"enter", "return"}:
            return GoalIntent(action=ActionEnum.ENTER, target=target.lower(), source="rule")
        return GoalIntent(
            action=ActionEnum.CLICK,
            target=target if target else "button",
            source="rule",
        )

    if "scroll" in goal_lower:
        direction = "up" if "up" in goal_lower else "down"
        return GoalIntent(action=ActionEnum.SCROLL, target=direction, source="rule")

    if any(keyword in goal_lower for keyword in ["submit", "send", "login", "continue", "next"]):
        return GoalIntent(action=ActionEnum.CLICK, target="button", source="rule")

    return GoalIntent(action=ActionEnum.CLICK, target="button", source="rule")


def _rule_parse_intent(goal: str) -> ParsedIntent:
    raw = goal.strip()
    if not raw:
        return ParsedIntent(action=ActionEnum.WAIT, source="rule")

    goal_lower = raw.lower()
    quoted_text = _extract_quoted_text(raw)

    search_match = re.search(r"(?:search|find|look for)(?:\s+for)?\s+(.+)", raw, flags=re.IGNORECASE)
    if search_match:
        query = quoted_text or _clean_text(search_match.group(1))
        if query:
            return ParsedIntent(
                action=ActionEnum.TYPE,
                text=query,
                target="search_input",
                source="rule",
            )
        return ParsedIntent(action=ActionEnum.CLICK, target="search_input", source="rule")

    type_match = re.search(r"(?:type|enter|input|write)\s+(.+)", raw, flags=re.IGNORECASE)
    if type_match:
        typed = quoted_text or _clean_text(type_match.group(1))
        if typed:
            return ParsedIntent(
                action=ActionEnum.TYPE,
                text=typed,
                target="input",
                source="rule",
            )

    click_match = re.search(r"(?:click|open|select|tap|press)\s+(.+)", raw, flags=re.IGNORECASE)
    if click_match:
        target = _clean_text(click_match.group(1))
        return ParsedIntent(
            action=ActionEnum.CLICK,
            target=target if target else "button",
            source="rule",
        )

    if "scroll" in goal_lower:
        direction = "up" if "up" in goal_lower else "down"
        return ParsedIntent(action=ActionEnum.SCROLL, target=direction, source="rule")

    if any(keyword in goal_lower for keyword in ["submit", "send", "login", "continue", "next"]):
        return ParsedIntent(action=ActionEnum.CLICK, target="button", source="rule")

    return ParsedIntent(action=ActionEnum.CLICK, target="button", source="rule")


def _llm_enabled() -> bool:
    value = os.getenv("KAI_ENABLE_INTENT_LLM", "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _llm_parse_goal_once(goal: str) -> GoalIntent | None:
    if not _llm_enabled() or not goal.strip():
        return None

    try:
        client = OpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        )

        prompt = (
            "Parse this desktop automation goal into a single JSON object with keys "
            "action, text, target, open_app. "
            "Allowed actions: click, type, scroll, enter, wait. "
            "Return JSON only."
        )

        response = client.chat.completions.create(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b"),
            temperature=0.0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": goal},
            ],
        )

        content = (response.choices[0].message.content or "").strip()
        payload = _extract_json_object(content)
        if not payload:
            return None

        action_raw = str(payload.get("action", "wait")).strip().lower()
        mapping = {
            "click": ActionEnum.CLICK,
            "type": ActionEnum.TYPE,
            "scroll": ActionEnum.SCROLL,
            "enter": ActionEnum.ENTER,
            "wait": ActionEnum.WAIT,
        }
        action = mapping.get(action_raw, ActionEnum.WAIT)

        text = payload.get("text")
        target = payload.get("target")
        open_app = bool(payload.get("open_app", False))

        return GoalIntent(
            action=action,
            text=str(text).strip() if text else None,
            target=str(target).strip() if target else None,
            open_app=open_app,
            source="llm",
        )
    except Exception as exc:
        print(f"[INTENT] LLM parse skipped: {type(exc).__name__}")
        return None


def parse_goal(goal: str) -> GoalIntent:
    key = _normalize_goal(goal)
    cached = _INTENT_CACHE.get(key)
    if cached is not None:
        return GoalIntent(
            action=cached.action,
            text=cached.text,
            target=cached.target,
            open_app=cached.open_app,
            source="cache",
        )

    parsed = _rule_parse_goal(goal)

    if parsed.action == ActionEnum.CLICK and parsed.target == "button":
        llm_parsed = _llm_parse_goal_once(goal)
        if llm_parsed is not None:
            parsed = llm_parsed

    _INTENT_CACHE[key] = parsed
    return parsed


def parse_intent(goal: str) -> ParsedIntent:
    parsed_goal = parse_goal(goal)
    return ParsedIntent(
        action=parsed_goal.action,
        text=parsed_goal.text,
        target=parsed_goal.target,
        source=parsed_goal.source,
    )


def analyze_screen(
    base64_image: str,
    ui_elements: list[dict[str, Any]] | None = None,
    current_goal: Optional[str] = None,
    last_action: Optional[UIAction] = None,
    screen_size: tuple[int, int] | None = None,
) -> UIAction:
    """Compatibility wrapper that now emits a fast, intent-only action skeleton."""
    goal = (current_goal or "").strip()
    intent = parse_intent(goal)

    if intent.action == ActionEnum.WAIT:
        return safe_wait_action(
            reason="No actionable user goal provided.",
            intent_summary="No clear intent to execute.",
            next_step_summary="Set a concrete goal and run next step.",
        )

    target_label = intent.target or ("input" if intent.action == ActionEnum.TYPE else "button")
    confidence = 0.66

    return UIAction(
        intent_summary=f"Intent parsed from goal via {intent.source} parser.",
        next_step_summary="Use fast vision grounding to execute the next reversible step.",
        action=intent.action,
        target_label=target_label,
        target_description=f"Goal intent target: {target_label}",
        target_coordinates=None,
        text_to_type=intent.text if intent.action == ActionEnum.TYPE else None,
        confidence_score=confidence,
        uncertainty_reason="Fast intent parse; final target is resolved by perception loop.",
    )
