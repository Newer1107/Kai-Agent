from __future__ import annotations

from typing import Any, Optional, Tuple

import pyautogui

from schema import ActionEnum, UIAction
from policy import get_action_risk, get_min_confidence_for_action, is_intent_clear

try:
    import win32api
    import win32con

    _HAS_PYWIN32 = True
except Exception:
    _HAS_PYWIN32 = False


pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05


def request_user_approval(action: UIAction) -> bool:
    """Display the proposed action and require explicit confirmation."""
    if _HAS_PYWIN32:
        try:
            win32api.MessageBeep(win32con.MB_ICONQUESTION)
        except Exception:
            pass

    print("\nProposed action:")
    print(f"  intent_summary: {action.intent_summary}")
    print(f"  action: {action.action.value}")
    print(f"  target_description: {action.target_description}")
    print(f"  target_coordinates: {action.target_coordinates}")
    print(f"  text_to_type: {action.text_to_type}")
    print(f"  confidence_score: {action.confidence_score:.2f}")
    print(f"  uncertainty_reason: {action.uncertainty_reason}")

    answer = input("Approve this action? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def validate_coordinates(x: int, y: int) -> bool:
    """Return True only when the coordinate is inside current screen bounds."""
    width, height = pyautogui.size()
    return 0 <= x < width and 0 <= y < height


def scale_coordinates(
    x: int,
    y: int,
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
) -> Tuple[int, int]:
    if resized_size[0] <= 0 or resized_size[1] <= 0:
        raise ValueError("resized_size must be positive for coordinate scaling")

    scale_x = original_size[0] / resized_size[0]
    scale_y = original_size[1] / resized_size[1]
    return int(x * scale_x), int(y * scale_y)


def to_screen_coordinates(
    model_x: int,
    model_y: int,
    original_size: Tuple[int, int],
    resized_size: Tuple[int, int],
    region_offset: Tuple[int, int] = (0, 0),
) -> Tuple[int, int]:
    """Map model coordinates from resized-image space into full screen coordinates."""
    scaled_x, scaled_y = scale_coordinates(model_x, model_y, original_size, resized_size)
    return scaled_x + region_offset[0], scaled_y + region_offset[1]


def highlight_target(x: int, y: int) -> None:
    """Briefly move the cursor to visualize the target, then restore position."""
    if not validate_coordinates(x, y):
        return

    current = pyautogui.position()
    try:
        pyautogui.moveTo(x, y, duration=0.12)
        pyautogui.moveTo(current.x, current.y, duration=0.12)
    except Exception:
        pass


def _scroll_amount_from_description(target_description: str | None) -> int:
    text = (target_description or "").lower()
    if "up" in text:
        return 350
    if "down" in text:
        return -350
    return -250


def resolve_action_coordinates(
    action: UIAction,
    selected_element: Optional[dict[str, Any]] = None,
    min_confidence: float = 0.5,
) -> Optional[Tuple[int, int]]:
    """Resolve execution coordinates using risk-based confidence policy."""
    if not selected_element:
        print("Execution skipped: no selected UI element was provided.")
        return None

    if bool(selected_element.get("ambiguous", False)):
        print("Execution skipped: resolved UI element is marked ambiguous.")
        return None

    confidence = float(selected_element.get("confidence", 0.0))
    source = str(selected_element.get("source", "yolo"))
    
    # Use RISK-BASED policy instead of flat 0.5 threshold
    actual_min_confidence = get_min_confidence_for_action(
        action=action.action.value,
        source=source,
        goal="",  # Goal not available at this layer, use default policy
    )
    
    print(f"[EXEC POLICY] Action: {action.action.value} | Risk: {get_action_risk(action.action.value)} | Confidence: {confidence:.2f} | Min required: {actual_min_confidence:.2f} | Source: {source}")
    
    if confidence < actual_min_confidence:
        print(
            f"Execution skipped: selected element confidence {confidence:.2f} below {actual_min_confidence:.2f} (source: {source})."
        )
        return None
    
    if source != "yolo":
        print(f"[debug] executing from {source} source with confidence {confidence:.2f}")

    center = selected_element.get("center")
    if not (isinstance(center, (list, tuple)) and len(center) == 2):
        print("Execution skipped: selected element does not include a valid center.")
        return None

    try:
        return int(center[0]), int(center[1])
    except Exception:
        return None


def execute_action(
    action: UIAction,
    selected_element: Optional[dict[str, Any]] = None,
    model_coordinates: Optional[Tuple[int, int]] = None,
    scaled_coordinates: Optional[Tuple[int, int]] = None,
    original_size: Optional[Tuple[int, int]] = None,
    resized_size: Optional[Tuple[int, int]] = None,
) -> bool:
    """Execute only validated, approved actions and fail safely on any error."""
    if action.action == ActionEnum.WAIT:
        print("No action executed (wait).")
        return False

    resolved = resolve_action_coordinates(action, selected_element=selected_element, min_confidence=0.5)
    if resolved is None:
        print("Execution skipped: no valid coordinates resolved from selected UI element.")
        return False

    x, y = resolved
    screen_width, screen_height = pyautogui.size()

    # Coordinate debug details for DPI/resize troubleshooting.
    print(f"[debug] screen_size: {(screen_width, screen_height)}")
    if original_size and resized_size:
        print(f"[debug] original_size: {original_size}, resized_size: {resized_size}")
    if model_coordinates is not None:
        print(f"[debug] model_coords: {model_coordinates}")
    print(f"[debug] scaled_coords: {scaled_coordinates or (x, y)}")
    if selected_element is not None:
        print(
            "[debug] selected_element: "
            f"type={selected_element.get('type')} confidence={selected_element.get('confidence')}"
        )

    if not validate_coordinates(x, y):
        print(f"[warning] coordinates are outside screen bounds: {(x, y)}")
        return False

    try:
        pyautogui.moveTo(x, y, duration=0.5)

        if action.action == ActionEnum.CLICK:
            pyautogui.click(x=x, y=y)

        elif action.action == ActionEnum.TYPE:
            if not action.text_to_type:
                print("Execution skipped: action='type' but text_to_type is empty.")
                return False
            pyautogui.click(x=x, y=y)
            pyautogui.typewrite(action.text_to_type, interval=0.01)

        elif action.action == ActionEnum.SCROLL:
            pyautogui.moveTo(x=x, y=y, duration=0.1)
            pyautogui.scroll(_scroll_amount_from_description(action.target_description))

        else:
            print("Execution skipped: unknown action type.")
            return False

        print("Action executed successfully.")
        return True

    except pyautogui.FailSafeException:
        print("Execution aborted by pyautogui fail-safe (cursor moved to top-left corner).")
        return False
    except Exception as exc:
        print(f"Execution failed safely: {exc}")
        return False
