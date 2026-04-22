from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ranking import RankedElement, rank_elements
from heuristics import (
    detect_by_goal_heuristic,
    detect_input_by_shape,
    infer_input_field,
    is_safe_heuristic_location,
)


@dataclass(frozen=True)
class ResolveResult:
    element: Optional[dict[str, Any]]
    ranked_candidates: list[RankedElement]
    reason: str
    ambiguous: bool
    used_heuristic: bool = False


def _goal_tokens(goal: str) -> list[str]:
    tokens = [token.strip().lower() for token in goal.split()]
    return [token for token in tokens if len(token) >= 3]


def _type_matches(element: dict[str, Any], target_label: str) -> bool:
    target = target_label.strip().lower()
    element_type = str(element.get("type", "")).strip().lower()
    semantic = str(element.get("semantic_label", "")).strip().lower()

    if element_type == target:
        return True
    if target in element_type or element_type in target:
        return True
    if semantic == target or target in semantic:
        return True
    return False


def _semantic_goal_filter(elements: list[dict[str, Any]], goal: str) -> list[dict[str, Any]]:
    tokens = _goal_tokens(goal)
    if not tokens:
        return elements

    filtered: list[dict[str, Any]] = []
    for element in elements:
        haystack = " ".join(
            [
                str(element.get("type", "")),
                str(element.get("semantic_label", "")),
                str(element.get("text", "")),
            ]
        ).lower()
        if any(token in haystack for token in tokens):
            filtered.append(element)

    return filtered or elements


def _get_min_confidence_for_target(target_label: str, goal: str) -> float:
    """Relax confidence threshold based on target type and goal."""
    target = target_label.strip().lower()
    goal_lower = goal.lower()
    
    if "input" in target or "search" in target:
        if any(kw in goal_lower for kw in ["search", "type", "enter", "find"]):
            return 0.3
        return 0.4
    
    if "button" in target:
        return 0.45
    
    return 0.5


def _debug_log_resolution(
    goal: str,
    target_label: str,
    candidates: list[dict[str, Any]] | list[RankedElement],
    chosen: dict[str, Any] | None,
    used_heuristic: bool,
) -> None:
    """Print debug info about resolution process."""
    print("=== RESOLUTION DEBUG ===")
    print(f"Goal: {goal}")
    print(f"Target: {target_label}")
    if isinstance(candidates, list) and candidates:
        if isinstance(candidates[0], dict):
            print(f"Candidates: {len(candidates)} elements")
        else:
            print(f"Candidates: {len(candidates)} ranked elements (top: {candidates[0].element.get('type')} @ {candidates[0].score:.2f})")
    else:
        print("Candidates: none")
    if chosen:
        print(f"Chosen: {chosen.get('type')} @ center {chosen.get('center')} (source: {chosen.get('source', 'yolo')})")
    else:
        print("Chosen: none (downgraded to wait)")
    print(f"Fallback used: {used_heuristic}")
    print("=== END DEBUG ===")


def resolve_target(
    ui_elements: list[dict[str, Any]],
    target_label: str,
    goal: str,
    action_type: str = "click",
) -> Optional[dict[str, Any]]:
    return resolve_target_detailed(
        ui_elements=ui_elements,
        target_label=target_label,
        goal=goal,
        action_type=action_type,
    ).element


def resolve_target_detailed(
    ui_elements: list[dict[str, Any]],
    target_label: str,
    goal: str,
    action_type: str = "click",
    min_confidence: float = 0.5,
    min_score: float = 0.42,
    ambiguity_margin: float = 0.06,
    screen_size: tuple[int, int] | None = None,
) -> ResolveResult:
    if not ui_elements:
        if screen_size:
            fallback = detect_by_goal_heuristic([], goal, screen_size[0], screen_size[1])
            if fallback and is_safe_heuristic_location(fallback, screen_size[0], screen_size[1]):
                _debug_log_resolution(goal, target_label, [], fallback, True)
                return ResolveResult(
                    element=fallback,
                    ranked_candidates=[],
                    reason="Resolved using goal-aware heuristic (no YOLO detections).",
                    ambiguous=False,
                    used_heuristic=True,
                )
        return ResolveResult(
            element=None,
            ranked_candidates=[],
            reason="No UI elements available for target resolution.",
            ambiguous=False,
            used_heuristic=False,
        )

    if not target_label:
        return ResolveResult(
            element=None,
            ranked_candidates=[],
            reason="No target label was provided.",
            ambiguous=False,
            used_heuristic=False,
        )

    adaptive_min_confidence = _get_min_confidence_for_target(target_label, goal)
    if min_confidence > 0.0:
        min_confidence = min(min_confidence, adaptive_min_confidence)

    typed_candidates = [item for item in ui_elements if _type_matches(item, target_label)]
    if not typed_candidates:
        if screen_size:
            if "input" in target_label.lower():
                shape_match = detect_input_by_shape(ui_elements, screen_size[0], screen_size[1])
                if shape_match and is_safe_heuristic_location(shape_match, screen_size[0], screen_size[1]):
                    _debug_log_resolution(goal, target_label, ui_elements, shape_match, True)
                    return ResolveResult(
                        element=shape_match,
                        ranked_candidates=[],
                        reason=f"Resolved using shape-based heuristic (no type match for '{target_label}').",
                        ambiguous=False,
                        used_heuristic=True,
                    )
            
            goal_match = detect_by_goal_heuristic(ui_elements, goal, screen_size[0], screen_size[1])
            if goal_match and is_safe_heuristic_location(goal_match, screen_size[0], screen_size[1]):
                _debug_log_resolution(goal, target_label, ui_elements, goal_match, True)
                return ResolveResult(
                    element=goal_match,
                    ranked_candidates=[],
                    reason=f"Resolved using goal-aware heuristic (no match for '{target_label}', but goal suggests action).",
                    ambiguous=False,
                    used_heuristic=True,
                )
        
        _debug_log_resolution(goal, target_label, ui_elements, None, False)
        return ResolveResult(
            element=None,
            ranked_candidates=[],
            reason=f"No candidate matches target_label='{target_label}' and heuristics failed.",
            ambiguous=False,
            used_heuristic=False,
        )

    sem_candidates = _semantic_goal_filter(typed_candidates, goal)
    ranked = rank_elements(sem_candidates, goal=goal, action_type=action_type)
    if not ranked:
        _debug_log_resolution(goal, target_label, [], None, False)
        return ResolveResult(
            element=None,
            ranked_candidates=[],
            reason="No ranked candidates after semantic filtering.",
            ambiguous=False,
            used_heuristic=False,
        )

    best = ranked[0]
    best_conf = float(best.element.get("confidence", 0.0))
    if best_conf < min_confidence:
        if screen_size:
            fallback = detect_by_goal_heuristic(ui_elements, goal, screen_size[0], screen_size[1])
            if fallback and is_safe_heuristic_location(fallback, screen_size[0], screen_size[1]):
                _debug_log_resolution(goal, target_label, ranked, fallback, True)
                return ResolveResult(
                    element=fallback,
                    ranked_candidates=ranked,
                    reason=f"Best candidate {best_conf:.2f} below threshold; using goal-aware heuristic.",
                    ambiguous=False,
                    used_heuristic=True,
                )
        
        _debug_log_resolution(goal, target_label, ranked, None, False)
        return ResolveResult(
            element=None,
            ranked_candidates=ranked,
            reason=f"Best candidate confidence {best_conf:.2f} below {min_confidence:.2f}.",
            ambiguous=False,
            used_heuristic=False,
        )

    if best.score < min_score:
        _debug_log_resolution(goal, target_label, ranked, None, False)
        return ResolveResult(
            element=None,
            ranked_candidates=ranked,
            reason=f"Best candidate score {best.score:.2f} below {min_score:.2f}.",
            ambiguous=False,
            used_heuristic=False,
        )

    if len(ranked) > 1:
        second = ranked[1]
        second_conf = float(second.element.get("confidence", 0.0))
        if second.score >= min_score and abs(best.score - second.score) < ambiguity_margin:
            _debug_log_resolution(goal, target_label, ranked, None, False)
            return ResolveResult(
                element=None,
                ranked_candidates=ranked,
                reason=(
                    "Ambiguous target: top candidates are too close "
                    f"(scores {best.score:.2f} vs {second.score:.2f})."
                ),
                ambiguous=True,
                used_heuristic=False,
            )
        if abs(best.score - second.score) < ambiguity_margin and abs(best_conf - second_conf) < 0.05:
            _debug_log_resolution(goal, target_label, ranked, None, False)
            return ResolveResult(
                element=None,
                ranked_candidates=ranked,
                reason="Ambiguous target: similar score and confidence for top candidates.",
                ambiguous=True,
                used_heuristic=False,
            )

    chosen = dict(best.element)
    chosen["resolution_score"] = round(best.score, 4)
    chosen["score_breakdown"] = best.score_breakdown
    chosen["ambiguous"] = False
    chosen["source"] = "yolo"
    _debug_log_resolution(goal, target_label, ranked, chosen, False)
    return ResolveResult(
        element=chosen,
        ranked_candidates=ranked,
        reason="Resolved target successfully.",
        ambiguous=False,
        used_heuristic=False,
    )
