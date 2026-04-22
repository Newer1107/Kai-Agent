from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple

from pydantic import BaseModel, Field, model_validator


class ActionEnum(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"


class UIAction(BaseModel):
    """Structured, safety-focused action proposal returned by the LLM."""

    intent_summary: str = Field(..., min_length=1)
    next_step_summary: str = Field(..., min_length=1)
    action: ActionEnum
    target_label: Optional[str] = None
    target_description: Optional[str] = None
    target_coordinates: Optional[Tuple[int, int]] = None
    text_to_type: Optional[str] = None
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    uncertainty_reason: Optional[str] = None

    model_config = {
        "extra": "forbid",
        "str_strip_whitespace": True,
    }

    @model_validator(mode="after")
    def validate_safety_rules(self) -> "UIAction":
        if self.confidence_score < 0.7 and not self.uncertainty_reason:
            raise ValueError("uncertainty_reason is required when confidence_score < 0.7")

        if self.action == ActionEnum.TYPE and not self.text_to_type:
            raise ValueError("text_to_type is required when action='type'")

        return self


def safe_wait_action(
    reason: str,
    intent_summary: str = "No safe action identified.",
    next_step_summary: str = "Wait for clearer on-screen evidence before taking action.",
) -> UIAction:
    """Build a conservative wait action used for safe fallbacks."""
    return UIAction(
        intent_summary=intent_summary,
        next_step_summary=next_step_summary,
        action=ActionEnum.WAIT,
        target_label=None,
        target_description=None,
        target_coordinates=None,
        text_to_type=None,
        confidence_score=0.0,
        uncertainty_reason=reason,
    )
