from __future__ import annotations

import ctypes
import queue
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext
from typing import Any, Callable

import keyboard

from agent_state import (
    clear_state,
    disable_autopilot,
    enable_autopilot,
    get_current_step,
    get_goal,
    get_last_action,
    get_max_steps,
    is_autopilot_enabled,
    set_current_step,
    set_goal,
    set_last_action,
)
from agent_loop import (
    DecisionResult,
    LoopState,
    build_retry_fallback_plan,
    decide_action,
    observe_state,
    verify_success,
)
from execution import execute_action, validate_coordinates
from perception import capture_primary_screenshot
from reasoning import clear_intent_cache
from schema import ActionEnum, UIAction, safe_wait_action
from detector import DetectorError, draw_detections
from debug_overlay import draw_target_preview


def enable_dpi_awareness() -> None:
    """Ensure coordinate APIs use physical pixels on Windows."""
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


enable_dpi_awareness()


class LocalAssistant:
    def __init__(self) -> None:
        self._trigger_lock = threading.Lock()
        self._coordinate_debug: dict[str, Any] = {}

    def propose_next_action(self) -> UIAction:
        if not self._trigger_lock.acquire(blocking=False):
            return safe_wait_action(
                reason="Assistant is already processing another request.",
                intent_summary="A previous analysis is still running.",
                next_step_summary="Wait for the running analysis to finish before trying again.",
            )

        try:
            state = observe_state(max_width=1280)
            return self.propose_action_from_state(
                state=state,
                goal=(get_goal() or ""),
                last_action=get_last_action(),
            )
        except DetectorError as exc:
            return safe_wait_action(
                reason=f"Vision grounding unavailable: {exc}",
                intent_summary="Could not run YOLO grounding safely.",
                next_step_summary="Wait until the vision model is available.",
            )
        except Exception as exc:
            traceback.print_exc()
            return safe_wait_action(
                reason=f"Runtime error during analysis: {type(exc).__name__}",
                intent_summary="Screen analysis failed safely.",
                next_step_summary="Wait and trigger analysis again after checking logs.",
            )
        finally:
            self._trigger_lock.release()

    def propose_action_from_state(
        self,
        state: LoopState,
        goal: str,
        last_action: UIAction | None,
        decision: DecisionResult | None = None,
    ) -> UIAction:
        payload = state.payload
        if decision is None:
            decision = decide_action(
                state=state,
                goal=goal,
                last_action=last_action,
            )
        action = decision.action

        debug_path = draw_detections(
            image=payload.image,
            elements=payload.ui_elements,
            output_path=str(Path("debug") / "detections_latest.jpg"),
        )
        print(f"[debug] detection overlay saved to {debug_path}")

        self._coordinate_debug = {
            "model_coordinates": None,
            "scaled_coordinates": None,
            "original_size": payload.original_size,
            "resized_size": payload.resized_size,
            "detection_count": len(payload.ui_elements),
            "selected_element": decision.selected_element,
            "planned_actions": decision.planned_actions,
            "resolution_reason": decision.reason,
        }

        target = decision.selected_element
        if target is not None:
            center = target.get("center", action.target_coordinates or [0, 0])
            center_tuple = (int(center[0]), int(center[1]))
            self._coordinate_debug.update(
                {
                    "scaled_coordinates": center_tuple,
                    "target_label": target.get("type"),
                    "target_confidence": target.get("confidence"),
                }
            )

            if action.target_coordinates is None:
                action = action.model_copy(update={"target_coordinates": center_tuple})

            preview_path = draw_target_preview(
                image=payload.image,
                element=target,
                output_path=str(Path("debug") / "target_preview_latest.jpg"),
            )
            if preview_path:
                print(f"[debug] target preview saved to {preview_path}")

        print(
            "[PERF] "
            f"capture={state.capture_ms:.1f}ms "
            f"decision={decision.decision_ms:.1f}ms "
            f"fallback={decision.used_fallback}"
        )

        return self._validate_or_downgrade_coordinates(action)

    @staticmethod
    def _validate_or_downgrade_coordinates(action: UIAction) -> UIAction:
        if action.action in {ActionEnum.WAIT, ActionEnum.ENTER}:
            return action

        if action.target_coordinates is None:
            return action.model_copy(
                update={
                    "action": ActionEnum.WAIT,
                    "target_label": None,
                    "next_step_summary": "Wait because the suggested action has no coordinates.",
                    "target_description": None,
                    "target_coordinates": None,
                    "text_to_type": None,
                    "confidence_score": min(action.confidence_score, 0.59),
                    "uncertainty_reason": (
                        action.uncertainty_reason
                        or "No coordinates were provided for a non-wait action."
                    ),
                }
            )

        x, y = action.target_coordinates
        if validate_coordinates(x, y):
            return action

        print("Proposed coordinates are out of bounds. Downgrading action to wait.")
        return action.model_copy(
            update={
                "action": ActionEnum.WAIT,
                "target_label": None,
                "next_step_summary": "Wait because the suggested target location is invalid.",
                "target_description": None,
                "target_coordinates": None,
                "text_to_type": None,
                "confidence_score": min(action.confidence_score, 0.59),
                "uncertainty_reason": "Model proposed coordinates outside screen bounds.",
            }
        )

    def execute_approved_action(self, action: UIAction, approved: bool) -> bool:
        if action.action == ActionEnum.WAIT:
            print("No execution attempted because action is 'wait'. Returning to idle state.")
            return False

        if not approved:
            print("User rejected action. Returning to idle state.")
            return False

        executed = execute_action(
            action,
            selected_element=self._coordinate_debug.get("selected_element"),
            model_coordinates=self._coordinate_debug.get("model_coordinates"),
            scaled_coordinates=self._coordinate_debug.get("scaled_coordinates"),
            original_size=self._coordinate_debug.get("original_size"),
            resized_size=self._coordinate_debug.get("resized_size"),
        )
        if executed:
            set_last_action(action)
        return executed


class AssistantChatWindow:
    WINDOW_WIDTH = 460
    WINDOW_HEIGHT = 520
    WINDOW_MARGIN = 16

    def __init__(self, assistant: LocalAssistant) -> None:
        self.assistant = assistant
        self._step_lock = threading.Lock()
        self._ui_queue: queue.Queue[tuple[Callable[..., Any], tuple[object, ...]]] = queue.Queue()
        self._show_requested = threading.Event()
        self._reset_requested = threading.Event()
        self._exit_requested = threading.Event()
        self._stop_auto_requested = threading.Event()
        self._pending_action: UIAction | None = None

        self.root = tk.Tk()
        self.root.withdraw()

        self.window = tk.Toplevel(self.root)
        self.window.title("Kai Assistant Chat")
        self.window.resizable(False, False)
        self.window.attributes("-topmost", True)
        self.window.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.window.withdraw()

        self.goal_var = tk.StringVar(value="Goal: None")

        self._build_ui()
        self.root.after(50, self._process_ui_events)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.window, padx=10, pady=10)
        frame.pack(fill="both", expand=True)

        goal_label = tk.Label(frame, textvariable=self.goal_var, anchor="w", justify="left")
        goal_label.pack(fill="x", pady=(0, 8))

        self.chat_log = scrolledtext.ScrolledText(frame, height=20, wrap="word", state="disabled")
        self.chat_log.pack(fill="both", expand=True)

        input_row = tk.Frame(frame)
        input_row.pack(fill="x", pady=(8, 0))

        self.entry = tk.Entry(input_row)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda _event: self.on_send())

        send_button = tk.Button(input_row, text="Send", width=10, command=self.on_send)
        send_button.pack(side="left", padx=(8, 0))

        action_row = tk.Frame(frame)
        action_row.pack(fill="x", pady=(8, 0))

        step_button = tk.Button(action_row, text="Next Step", command=self.on_next_step)
        step_button.pack(side="left")

        auto_button = tk.Button(action_row, text="Auto", command=self.on_auto)
        auto_button.pack(side="left", padx=(8, 0))

        stop_button = tk.Button(action_row, text="Stop", command=self.on_stop_autopilot)
        stop_button.pack(side="left", padx=(8, 0))

        approve_button = tk.Button(action_row, text="Approve", command=self.on_approve)
        approve_button.pack(side="left", padx=(8, 0))

        reject_button = tk.Button(action_row, text="Reject", command=self.on_reject)
        reject_button.pack(side="left", padx=(8, 0))

        reset_button = tk.Button(action_row, text="Reset", command=self.on_reset)
        reset_button.pack(side="left", padx=(8, 0))

        hide_button = tk.Button(action_row, text="Hide", command=self.hide_window)
        hide_button.pack(side="right")

        self._append_chat("assistant", "Chat ready. Enter a goal or press Next Step.")
        self._append_chat(
            "assistant",
            "Commands: /step, /approve, /reject, /auto, /autopilot on, /autopilot off, /stop, /status, /reset, /goal <text>, /help",
        )

    def run(self) -> None:
        self.root.mainloop()

    def open_from_hotkey(self) -> None:
        self._show_requested.set()

    def reset_from_hotkey(self) -> None:
        self._reset_requested.set()

    def request_exit(self) -> None:
        self._exit_requested.set()

    def stop_autopilot_from_hotkey(self) -> None:
        disable_autopilot()
        self._stop_auto_requested.set()

    def _process_ui_events(self) -> None:
        if self._show_requested.is_set():
            self._show_requested.clear()
            self.show_window()

        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self.on_reset()

        if self._exit_requested.is_set():
            self._exit_requested.clear()
            self._exit()
            return

        if self._stop_auto_requested.is_set():
            self._stop_auto_requested.clear()
            self._append_chat("assistant", "[AUTO] Emergency stop received. Autopilot disabled.")

        while True:
            try:
                callback, args = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except Exception:
                traceback.print_exc()

        self.root.after(50, self._process_ui_events)

    def _enqueue_ui(self, callback: Callable[..., Any], *args: object) -> None:
        self._ui_queue.put((callback, args))

    def show_window(self) -> None:
        self._refresh_goal_label()
        self._reposition_bottom_right()
        self.window.deiconify()
        self.window.attributes("-topmost", True)
        self.window.lift()
        try:
            self.window.focus_force()
            self.entry.focus_set()
        except Exception:
            pass

    def hide_window(self) -> None:
        self.window.withdraw()

    def on_send(self) -> None:
        text = self.entry.get().strip()
        self.entry.delete(0, tk.END)

        if not text:
            self.on_next_step()
            return

        self._append_chat("you", text)

        lower = text.lower()
        if lower in {"/step", "/next"}:
            self.on_next_step()
            return

        if lower == "/approve":
            self.on_approve()
            return

        if lower == "/reject":
            self.on_reject()
            return

        if lower == "/help":
            self._append_chat(
                "assistant",
                "Commands: /step, /approve, /reject, /auto, /autopilot on, /autopilot off, /stop, /status, /reset, /goal <text>",
            )
            return

        if lower == "/auto":
            self.on_auto()
            return

        if lower.startswith("/autopilot"):
            parts = lower.split()
            if len(parts) == 1 or parts[1] == "on":
                self.on_auto()
            elif parts[1] == "off":
                self.on_stop_autopilot()
            else:
                self._append_chat("assistant", "Usage: /autopilot on | /autopilot off")
            return

        if lower == "/stop":
            self.on_stop_autopilot()
            return

        if lower == "/status":
            self._append_chat("assistant", self._autopilot_status_line())
            return

        if lower == "/reset":
            self.on_reset()
            return

        if lower.startswith("/goal "):
            goal = text[6:].strip()
            if goal:
                set_goal(goal)
                self._append_chat("assistant", f"Goal updated: {goal}")
                self._refresh_goal_label()
            else:
                self._append_chat("assistant", "Goal was empty. Nothing changed.")
            return

        set_goal(text)
        self._refresh_goal_label()
        self._append_chat("assistant", "Goal updated. Run /step when you want one safe next step.")

    def on_next_step(self) -> None:
        if is_autopilot_enabled():
            self._append_chat("assistant", "[AUTO] Disable autopilot first using /stop.")
            return

        if not self._step_lock.acquire(blocking=False):
            self._append_chat("assistant", "Already analyzing. Please wait.")
            return

        worker = threading.Thread(target=self._run_next_step_worker, daemon=True)
        worker.start()

    def on_approve(self) -> None:
        if is_autopilot_enabled():
            self._append_chat("assistant", "[AUTO] Manual approve is disabled while autopilot is running.")
            return

        if self._pending_action is None:
            self._append_chat("assistant", "No pending action to approve.")
            return

        action = self._pending_action
        self._pending_action = None
        self._append_chat("assistant", "Executing approved action...")
        worker = threading.Thread(target=self._run_execute_worker, args=(action,), daemon=True)
        worker.start()

    def on_reject(self) -> None:
        if self._pending_action is None:
            self._append_chat("assistant", "No pending action to reject.")
            return

        self._pending_action = None
        self._append_chat("assistant", "Pending action rejected. No execution performed.")

    def on_reset(self) -> None:
        disable_autopilot()
        clear_state()
        clear_intent_cache()
        self._pending_action = None
        self._refresh_goal_label()
        self._append_chat("assistant", "Goal and last action were cleared.")

    def on_auto(self) -> None:
        goal = (get_goal() or "").strip()
        if not goal:
            self._append_chat("assistant", "[AUTO] Cannot start autopilot: set a goal first.")
            return

        if is_autopilot_enabled():
            self._append_chat("assistant", "[AUTO] Autopilot already running.")
            return

        if not self._step_lock.acquire(blocking=False):
            self._append_chat("assistant", "[AUTO] Assistant is busy. Try again shortly.")
            return

        self._pending_action = None

        worker = threading.Thread(target=self._run_autopilot_worker, daemon=True)
        worker.start()

    def on_stop_autopilot(self) -> None:
        if is_autopilot_enabled():
            disable_autopilot()
            self._append_chat("assistant", "[AUTO] Stop requested. Finishing current action safely.")
            return
        self._append_chat("assistant", "[AUTO] Autopilot is not running.")

    def _run_next_step_worker(self) -> None:
        try:
            self._enqueue_ui(self._append_chat, "assistant", "Capturing screen...")
            self._enqueue_ui(self._append_chat, "assistant", "Analyzing...")
            action = self.assistant.propose_next_action()
            self._enqueue_ui(self._handle_proposed_action, action)

        except Exception:
            self._enqueue_ui(self._append_chat, "assistant", f"Unexpected error:\n{traceback.format_exc()}")
        finally:
            self._step_lock.release()

    @staticmethod
    def _is_high_risk_action(action: UIAction) -> bool:
        if action.action not in {ActionEnum.CLICK, ActionEnum.TYPE, ActionEnum.SCROLL, ActionEnum.ENTER}:
            return True

        combined = " ".join(
            [
                action.intent_summary or "",
                action.next_step_summary or "",
                action.target_label or "",
                action.target_description or "",
                action.text_to_type or "",
            ]
        ).lower()

        dangerous_keywords = [
            "delete",
            "remove",
            "uninstall",
            "format",
            "shutdown",
            "restart",
            "power off",
            "registry",
            "system32",
            "wipe",
            "erase",
        ]
        return any(keyword in combined for keyword in dangerous_keywords)

    @staticmethod
    def _autopilot_step_summary(step: int, action: UIAction) -> str:
        if action.action == ActionEnum.CLICK:
            return f"[AUTO] Step {step}: Clicking {action.target_label or 'target'}"
        if action.action == ActionEnum.TYPE:
            text = (action.text_to_type or "")[:48]
            return f"[AUTO] Step {step}: Typing \"{text}\""
        if action.action == ActionEnum.SCROLL:
            return f"[AUTO] Step {step}: Scrolling"
        if action.action == ActionEnum.ENTER:
            return f"[AUTO] Step {step}: Pressing enter"
        return f"[AUTO] Step {step}: Waiting"

    @staticmethod
    def _autopilot_candidate_summary(step: int, index: int, total: int, action: UIAction) -> str:
        return (
            f"[ACTION] Step {step}: executing candidate #{index}/{total} "
            f"({action.action.value}, conf={action.confidence_score:.2f})"
        )

    @staticmethod
    def _autopilot_status_line() -> str:
        mode = "ON" if is_autopilot_enabled() else "OFF"
        return (
            f"[AUTO] Status: {mode} | "
            f"Step {get_current_step()}/{get_max_steps()}"
        )

    def _run_autopilot_worker(self) -> None:
        goal = (get_goal() or "").strip()
        max_steps = get_max_steps()
        consecutive_failures = 0

        enable_autopilot(step_limit=max_steps)
        self._enqueue_ui(self._append_chat, "assistant", f"[AUTO] Enabled. Goal: {goal}")

        try:
            for step in range(1, max_steps + 1):
                if not is_autopilot_enabled():
                    self._enqueue_ui(self._append_chat, "assistant", "[AUTO] Interrupted by user.")
                    break

                set_current_step(step)
                prev_state = observe_state(max_width=1280)
                decision = decide_action(
                    state=prev_state,
                    goal=goal,
                    last_action=get_last_action(),
                )
                action = self.assistant.propose_action_from_state(
                    state=prev_state,
                    goal=goal,
                    last_action=get_last_action(),
                    decision=decision,
                )

                # Hybrid mode: always log proposed action, even when skipping approval.
                self._enqueue_ui(self._append_chat, "assistant", self._autopilot_step_summary(step, action))
                self._enqueue_ui(self._append_chat, "assistant", self._format_action_summary(action))

                if action.action == ActionEnum.WAIT:
                    self._enqueue_ui(self._append_chat, "assistant", f"[AUTO] Step {step}: Completed.")
                    break

                if action.confidence_score < 0.4:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[AUTO] Step {step}: Stopped (confidence {action.confidence_score:.2f} < 0.40).",
                    )
                    break

                if self._is_high_risk_action(action):
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[AUTO] Step {step}: Blocked high-risk action. Switching to manual mode.",
                    )
                    break

                candidate_plans = [plan for plan in decision.candidate_plans[:3] if plan.action.action != ActionEnum.WAIT]
                if not candidate_plans:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[RETRY] Step {step}: no executable ranked candidates; trying fallback.",
                    )

                step_success = False

                for idx, plan in enumerate(candidate_plans, start=1):
                    if not is_autopilot_enabled():
                        self._enqueue_ui(self._append_chat, "assistant", "[AUTO] Interrupted before execute.")
                        break

                    planned_actions = [planned for planned in plan.planned_actions if planned.action != ActionEnum.WAIT]
                    if not planned_actions:
                        planned_actions = [plan.action]

                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        self._autopilot_candidate_summary(step, idx, len(candidate_plans), planned_actions[0]),
                    )

                    candidate_prev_image = prev_state.payload.image
                    candidate_prev_elements = prev_state.payload.ui_elements
                    candidate_success = True

                    for sub_idx, candidate_action in enumerate(planned_actions, start=1):
                        if candidate_action.confidence_score < 0.4:
                            self._enqueue_ui(
                                self._append_chat,
                                "assistant",
                                f"[ACTION] candidate #{idx}.{sub_idx} aborted (confidence {candidate_action.confidence_score:.2f} < 0.40)",
                            )
                            candidate_success = False
                            consecutive_failures += 1
                            break

                        if self._is_high_risk_action(candidate_action):
                            self._enqueue_ui(
                                self._append_chat,
                                "assistant",
                                f"[ACTION] candidate #{idx}.{sub_idx} blocked as high-risk.",
                            )
                            candidate_success = False
                            consecutive_failures += 1
                            break

                        executed = execute_action(
                            candidate_action,
                            selected_element=plan.element,
                            original_size=prev_state.payload.original_size,
                            resized_size=prev_state.payload.resized_size,
                        )
                        if not executed:
                            consecutive_failures += 1
                            candidate_success = False
                            self._enqueue_ui(
                                self._append_chat,
                                "assistant",
                                f"[RETRY] candidate #{idx}.{sub_idx} execution failed ({consecutive_failures}/2)",
                            )
                            if consecutive_failures >= 2:
                                break
                            continue

                        next_image = capture_primary_screenshot()
                        success = verify_success(
                            prev_image=candidate_prev_image,
                            new_image=next_image,
                            action=candidate_action,
                            prev_elements=candidate_prev_elements,
                        )
                        if not success:
                            consecutive_failures += 1
                            candidate_success = False
                            self._enqueue_ui(
                                self._append_chat,
                                "assistant",
                                f"[VERIFY] candidate #{idx}.{sub_idx} failed; trying next candidate.",
                            )
                            if consecutive_failures >= 2:
                                break
                            break

                        consecutive_failures = 0
                        set_last_action(candidate_action)
                        candidate_prev_image = next_image
                        self._enqueue_ui(
                            self._append_chat,
                            "assistant",
                            f"[VERIFY] Step {step}: success on candidate #{idx}.{sub_idx}",
                        )

                    if candidate_success:
                        step_success = True
                        break

                    if consecutive_failures >= 2:
                        break

                if step_success:
                    continue

                if consecutive_failures >= 2:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        "[AUTO] Stopped after 2 consecutive failures.",
                    )
                    break

                fallback_plan = build_retry_fallback_plan(
                    state=prev_state,
                    goal=goal,
                    last_action=get_last_action(),
                )
                if fallback_plan is None or fallback_plan.action.action == ActionEnum.WAIT:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[RETRY] Step {step}: fallback unavailable. Stopping.",
                    )
                    break

                if fallback_plan.action.confidence_score < 0.4:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[RETRY] Step {step}: fallback confidence too low ({fallback_plan.action.confidence_score:.2f}).",
                    )
                    break

                if self._is_high_risk_action(fallback_plan.action):
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[RETRY] Step {step}: fallback blocked as high-risk.",
                    )
                    break

                self._enqueue_ui(self._append_chat, "assistant", "[RETRY] fallback triggered")
                fallback_actions = [planned for planned in fallback_plan.planned_actions if planned.action != ActionEnum.WAIT]
                if not fallback_actions:
                    fallback_actions = [fallback_plan.action]

                fallback_prev_image = prev_state.payload.image
                fallback_prev_elements = prev_state.payload.ui_elements
                fallback_success = True

                for sub_idx, fallback_action in enumerate(fallback_actions, start=1):
                    if fallback_action.confidence_score < 0.4:
                        self._enqueue_ui(
                            self._append_chat,
                            "assistant",
                            f"[RETRY] fallback candidate #{sub_idx} aborted (confidence {fallback_action.confidence_score:.2f} < 0.40)",
                        )
                        consecutive_failures += 1
                        fallback_success = False
                        break

                    if self._is_high_risk_action(fallback_action):
                        self._enqueue_ui(
                            self._append_chat,
                            "assistant",
                            f"[RETRY] fallback candidate #{sub_idx} blocked as high-risk.",
                        )
                        consecutive_failures += 1
                        fallback_success = False
                        break

                    executed = execute_action(
                        fallback_action,
                        selected_element=fallback_plan.element,
                        original_size=prev_state.payload.original_size,
                        resized_size=prev_state.payload.resized_size,
                    )

                    if not executed:
                        consecutive_failures += 1
                        fallback_success = False
                        self._enqueue_ui(
                            self._append_chat,
                            "assistant",
                            f"[RETRY] fallback execution failed ({consecutive_failures}/2)",
                        )
                        if consecutive_failures >= 2:
                            break
                        continue

                    fallback_next_image = capture_primary_screenshot()
                    verified = verify_success(
                        prev_image=fallback_prev_image,
                        new_image=fallback_next_image,
                        action=fallback_action,
                        prev_elements=fallback_prev_elements,
                    )
                    if not verified:
                        consecutive_failures += 1
                        fallback_success = False
                        self._enqueue_ui(
                            self._append_chat,
                            "assistant",
                            f"[VERIFY] fallback candidate #{sub_idx} failed.",
                        )
                        if consecutive_failures >= 2:
                            break
                        break

                    consecutive_failures = 0
                    set_last_action(fallback_action)
                    fallback_prev_image = fallback_next_image

                if fallback_success:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[VERIFY] Step {step}: fallback succeeded",
                    )
                    continue

                if consecutive_failures >= 2:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        "[AUTO] Stopped after 2 consecutive failures.",
                    )
                    break

                self._enqueue_ui(
                    self._append_chat,
                    "assistant",
                    f"[AUTO] Step {step}: No screen change detected after retries. Stopping.",
                )
                break
            else:
                self._enqueue_ui(
                    self._append_chat,
                    "assistant",
                    f"[AUTO] Step limit reached ({max_steps}). Autopilot stopped.",
                )
        except Exception:
            self._enqueue_ui(self._append_chat, "assistant", f"[AUTO] Runtime error:\n{traceback.format_exc()}")
        finally:
            disable_autopilot()
            self._enqueue_ui(self._append_chat, "assistant", "[AUTO] Disabled.")
            self._step_lock.release()

    def _run_execute_worker(self, action: UIAction) -> None:
        try:
            executed = self.assistant.execute_approved_action(action, approved=True)
            if executed:
                self._enqueue_ui(self._append_chat, "assistant", "Action executed successfully.")
                follow_up = self.assistant.propose_next_action()
                if follow_up.action != ActionEnum.WAIT:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        "Observed updated screen and prepared next step.",
                    )
                    self._enqueue_ui(self._handle_proposed_action, follow_up)
                else:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        "Goal loop paused: no immediate safe follow-up step.",
                    )
            else:
                self._enqueue_ui(
                    self._append_chat,
                    "assistant",
                    "Execution was attempted but did not complete safely.",
                )
        except Exception:
            self._enqueue_ui(self._append_chat, "assistant", f"Unexpected error:\n{traceback.format_exc()}")

    def _handle_proposed_action(self, action: UIAction) -> None:
        self._append_chat("assistant", self._format_action_summary(action))

        if action.action == ActionEnum.WAIT:
            self._pending_action = None
            self._append_chat("assistant", "No action executed because the safe decision is wait.")
            return

        self._pending_action = action
        self._append_chat("assistant", "Use Approve button or /approve to execute. Use Reject or /reject to cancel.")

    def _format_action_summary(self, action: UIAction) -> str:
        return (
            "Proposed action:\n"
            f"- intent_summary: {action.intent_summary}\n"
            f"- next_step_summary: {action.next_step_summary}\n"
            f"- action: {action.action.value}\n"
            f"- target_label: {action.target_label}\n"
            f"- target_description: {action.target_description}\n"
            f"- target_coordinates: {action.target_coordinates}\n"
            f"- confidence_score: {action.confidence_score:.2f}\n"
            f"- uncertainty_reason: {action.uncertainty_reason}"
        )

    def _append_chat(self, role: str, message: str) -> None:
        prefix = "You" if role == "you" else "Assistant"
        self.chat_log.configure(state="normal")
        self.chat_log.insert(tk.END, f"{prefix}: {message}\n\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see(tk.END)

    def _refresh_goal_label(self) -> None:
        goal = get_goal()
        self.goal_var.set(f"Goal: {goal if goal else 'None'}")

    def _reposition_bottom_right(self) -> None:
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = max(0, screen_width - self.WINDOW_WIDTH - self.WINDOW_MARGIN)
        y = max(0, screen_height - self.WINDOW_HEIGHT - self.WINDOW_MARGIN)
        self.window.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}+{x}+{y}")

    def _exit(self) -> None:
        self.root.quit()


def main() -> None:
    assistant = LocalAssistant()
    chat_window = AssistantChatWindow(assistant)

    keyboard.add_hotkey("ctrl+alt+space", chat_window.open_from_hotkey)
    keyboard.add_hotkey("ctrl+alt+r", chat_window.reset_from_hotkey)
    keyboard.add_hotkey("ctrl+alt+q", chat_window.request_exit)
    keyboard.add_hotkey("ctrl+alt+s", chat_window.stop_autopilot_from_hotkey)
    print("Local assistant is running in idle mode.")
    print("Hotkey: Ctrl+Alt+Space (open/focus chat window)")
    print("Reset state hotkey: Ctrl+Alt+R")
    print("Emergency stop hotkey: Ctrl+Alt+S")
    print("Exit hotkey: Ctrl+Alt+Q")

    try:
        chat_window.run()
    finally:
        keyboard.unhook_all_hotkeys()
        print("Assistant stopped.")


if __name__ == "__main__":
    main()
