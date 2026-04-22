from __future__ import annotations

import ctypes
import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext
from typing import Any, Callable

import keyboard

from agent_state import clear_state, get_goal, get_last_action, set_goal, set_last_action
from execution import execute_action, validate_coordinates
from perception import capture_screen_for_inference
from reasoning import analyze_screen
from resolver import resolve_target_detailed
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
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_AMBIGUITY_MARGIN = 0.05

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

        goal = get_goal()
        try:
            payload = capture_screen_for_inference(max_width=1280)
            action = analyze_screen(
                base64_image=payload.image_base64,
                ui_elements=payload.ui_elements,
                current_goal=goal,
                last_action=get_last_action(),
                screen_size=(payload.original_size[0], payload.original_size[1]),
            )
            action = self._enforce_runtime_safety(action)

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
                "selected_element": None,
                "resolution_reason": None,
            }

            action = self._ground_action_with_vision(
                action,
                payload.ui_elements,
                payload.image,
                payload.original_size,
            )

            return self._validate_or_downgrade_coordinates(action)
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

    def _ground_action_with_vision(
        self,
        action: UIAction,
        detections: list[dict[str, Any]],
        image,
        original_size: tuple[int, int],
    ) -> UIAction:
        if action.action == ActionEnum.WAIT:
            return action

        if not action.target_label:
            return action.model_copy(
                update={
                    "action": ActionEnum.WAIT,
                    "target_coordinates": None,
                    "confidence_score": min(action.confidence_score, 0.59),
                    "uncertainty_reason": (
                        action.uncertainty_reason
                        or "No target label available for YOLO grounding."
                    ),
                    "next_step_summary": "Wait because the target label is missing.",
                }
            )

        resolved = resolve_target_detailed(
            ui_elements=detections,
            target_label=action.target_label,
            goal=get_goal() or "",
            action_type=action.action.value,
            min_confidence=self.DETECTION_MIN_CONFIDENCE,
            ambiguity_margin=self.DETECTION_AMBIGUITY_MARGIN,
            screen_size=(original_size[0], original_size[1]),
        )
        self._coordinate_debug["resolution_reason"] = resolved.reason

        if resolved.element is None:
            return action.model_copy(
                update={
                    "action": ActionEnum.WAIT,
                    "target_coordinates": None,
                    "confidence_score": min(action.confidence_score, 0.59),
                    "uncertainty_reason": resolved.reason
                    or "No matching detection found for target label.",
                    "next_step_summary": "Wait because no reliable visual target was found.",
                }
            )

        target = resolved.element
        center = target["center"]
        center_tuple = (int(center[0]), int(center[1]))
        self._coordinate_debug.update(
            {
                "scaled_coordinates": center_tuple,
                "target_label": target["type"],
                "target_confidence": target["confidence"],
                "selected_element": target,
            }
        )

        preview_path = draw_target_preview(
            image=image,
            element=target,
            output_path=str(Path("debug") / "target_preview_latest.jpg"),
        )
        if preview_path:
            print(f"[debug] target preview saved to {preview_path}")

        return action.model_copy(
            update={
                "target_coordinates": center_tuple,
                "target_label": target["type"],
                "target_description": action.target_description
                or (
                    f"{target['type']}"
                    f" text={target.get('text')}"
                    f" region={target.get('region')}"
                    f" (YOLO {target['confidence']:.2f}, score {target.get('resolution_score', 0.0):.2f})"
                ),
            }
        )

    @staticmethod
    def _enforce_runtime_safety(action: UIAction) -> UIAction:
        if action.confidence_score < 0.6 and action.action != ActionEnum.WAIT:
            return action.model_copy(
                update={
                    "action": ActionEnum.WAIT,
                    "target_label": None,
                    "next_step_summary": "Pause and wait because confidence is below safe threshold.",
                    "target_description": None,
                    "target_coordinates": None,
                    "text_to_type": None,
                    "uncertainty_reason": (
                        action.uncertainty_reason
                        or "Confidence below 0.6, so execution was downgraded to wait."
                    ),
                }
            )
        return action

    @staticmethod
    def _validate_or_downgrade_coordinates(action: UIAction) -> UIAction:
        if action.action == ActionEnum.WAIT:
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
            "Commands: /step, /approve, /reject, /reset, /goal <text>, /help",
        )

    def run(self) -> None:
        self.root.mainloop()

    def open_from_hotkey(self) -> None:
        self._show_requested.set()

    def reset_from_hotkey(self) -> None:
        self._reset_requested.set()

    def request_exit(self) -> None:
        self._exit_requested.set()

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
        if lower == "/step":
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
                "Commands: /step, /approve, /reject, /reset, /goal <text>",
            )
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
        if not self._step_lock.acquire(blocking=False):
            self._append_chat("assistant", "Already analyzing. Please wait.")
            return

        worker = threading.Thread(target=self._run_next_step_worker, daemon=True)
        worker.start()

    def on_approve(self) -> None:
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
        clear_state()
        self._pending_action = None
        self._refresh_goal_label()
        self._append_chat("assistant", "Goal and last action were cleared.")

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

    def _run_execute_worker(self, action: UIAction) -> None:
        try:
            executed = self.assistant.execute_approved_action(action, approved=True)
            if executed:
                self._enqueue_ui(self._append_chat, "assistant", "Action executed successfully.")
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
    print("Local assistant is running in idle mode.")
    print("Hotkey: Ctrl+Alt+Space (open/focus chat window)")
    print("Reset state hotkey: Ctrl+Alt+R")
    print("Exit hotkey: Ctrl+Alt+Q")

    try:
        chat_window.run()
    finally:
        keyboard.unhook_all_hotkeys()
        print("Assistant stopped.")


if __name__ == "__main__":
    main()
