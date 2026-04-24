from __future__ import annotations

import ctypes
import queue
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from typing import Any, Callable

try:
    import customtkinter as ctk
except Exception:  # pragma: no cover
    ctk = None

import keyboard

from agent_state import (
    clear_state,
    disable_autopilot,
    enable_autopilot,
    get_current_step,
    get_dynamic_max_steps,
    get_goal,
    get_last_action,
    get_remaining_budget,
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
from perception import capture_primary_screenshot, get_telemetry_summary
from reasoning import clear_intent_cache
from schema import ActionEnum, UIAction, safe_wait_action
from detector import DetectorError, draw_detections
from debug_overlay import draw_target_preview
from memory.session_store import get_session_memory
from eval.benchmark import run_quick_benchmark_suite
from universal_router import UniversalRouter
from action_router import ActionRouter


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


class _Tooltip:
    def __init__(self, widget: Any, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: tk.Toplevel | None = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, _: object) -> None:
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.configure(bg="#0D1117")
        label = tk.Label(
            self.tip_window,
            text=self.text,
            bg="#0D1117",
            fg="#E6EDF3",
            padx=8,
            pady=4,
            relief="solid",
            bd=1,
            font=("Segoe UI", 9),
        )
        label.pack()
        self.tip_window.wm_geometry(f"+{x}+{y}")

    def _hide(self, _: object) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class AssistantChatWindow:
    WINDOW_WIDTH = 400
    WINDOW_HEIGHT = 540
    WINDOW_X_MARGIN = 20
    WINDOW_Y_MARGIN = 20
    HEADER_HEIGHT = 140

    COLORS = {
        "bg": "#1C1C1E",
        "chat_bg": "#1C1C1E",
        "input_bg": "#1C1C1E",
        "input_field": "#2A2A2E",
        "header_left": "#5B4FE8",
        "header_right": "#4B8BF5",
        "header_shadow": "#151517",
        "title_border": "#2C2C2E",
        "text": "#E8E8E8",
        "text_user": "#CCCCCC",
        "muted": "#555555",
        "muted_soft": "#444444",
        "accent": "#7B6FF0",
        "accent_strong": "#5B4FE8",
        "thumb": "#2C2C2E",
        "status_idle": "#333333",
    }

    STATUS_COLORS = {
        "IDLE": "#333333",
        "THINKING": "#7B6FF0",
        "EXECUTING": "#7B6FF0",
        "AUTOPILOT": "#7C3AED",
    }

    def _enqueue_ui(self, func: Callable[..., Any], *args: object) -> None:
        """Safely queues UI updates from background worker threads."""
        self._ui_queue.put((func, args))

    def _process_ui_events(self) -> None:
        """Polls for hotkey events and queued UI updates safely on the main thread."""
        if self._exit_requested.is_set():
            self._exit()
            return

        if self._toggle_requested.is_set():
            self._toggle_requested.clear()
            if self.window.state() == "withdrawn":
                self.window.deiconify()
                self._reposition_bottom_right()
                self.window.lift()
                self.window.focus_force()
                self.entry.focus_set()
            else:
                self.hide_window()

        if self._show_requested.is_set():
            self._show_requested.clear()
            self.window.deiconify()
            self._reposition_bottom_right()
            self.window.lift()
            self.window.focus_force()
            self.entry.focus_set()

        if self._reset_requested.is_set():
            self._reset_requested.clear()
            self.on_reset()

        if self._stop_auto_requested.is_set():
            self._stop_auto_requested.clear()
            self.on_stop_autopilot()

        # Process any pending UI updates from the worker threads
        while True:
            try:
                func, args = self._ui_queue.get_nowait()
                func(*args)
            except queue.Empty:
                break

        # Re-schedule the polling loop
        self.root.after(50, self._process_ui_events)

    # Hotkey triggers
    def open_from_hotkey(self) -> None:
        self._toggle_requested.set()

    def reset_from_hotkey(self) -> None:
        self._reset_requested.set()

    def request_exit(self) -> None:
        self._exit_requested.set()

    def stop_autopilot_from_hotkey(self) -> None:
        self._stop_auto_requested.set()

    def run(self) -> None:
        """Starts the main Tkinter event loop."""
        self.root.mainloop()

    def __init__(self, assistant: LocalAssistant) -> None:
        if ctk is None:
            raise RuntimeError(
                "CustomTkinter is required for the redesigned UI. Install with: pip install customtkinter"
            )

        ctk.set_appearance_mode("dark")
        self.assistant = assistant
        self._step_lock = threading.Lock()
        self._ui_queue: queue.Queue[tuple[Callable[..., Any], tuple[object, ...]]] = queue.Queue()
        self._toggle_requested = threading.Event()
        self._show_requested = threading.Event()
        self._reset_requested = threading.Event()
        self._exit_requested = threading.Event()
        self._stop_auto_requested = threading.Event()
        self._pending_action: UIAction | None = None
        self._session_memory = get_session_memory()
        self._router = UniversalRouter(status_callback=self._router_status)
        self._desktop_router = ActionRouter()
        self._status_state = "IDLE"
        self._status_dot_on = True
        self._typing_active = False
        self._title_drag_offset = (0, 0)
        self._status_animation_tick = 0
        self._commands_visible = False
        self._entry_has_placeholder = True
        self._goal_editing = False
        self._chat_rows: list[dict[str, Any]] = []
        self._auto_peek_job: str | None = None

        self.root = ctk.CTk()
        self.root.withdraw()

        self.window = ctk.CTkToplevel(self.root)
        self.window.title("Kai Agent")
        self.window.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.window.resizable(False, False)
        self.window.attributes("-topmost", True)
        self.window.overrideredirect(True)
        self.window.protocol("WM_DELETE_WINDOW", self.hide_window)
        self.window.withdraw()

        self.goal_var = tk.StringVar(value="Goal: None")

        self._build_ui()
        self._refresh_goal_label()
        self._reposition_bottom_right()
        self._animate_status_dot()
        self._poll_stats()
        self.root.after(50, self._process_ui_events)

    def _build_ui(self) -> None:
        self.window.configure(fg_color=self.COLORS["bg"])

        self.outer_frame = ctk.CTkFrame(
            self.window,
            fg_color=self.COLORS["bg"],
            corner_radius=10,
            border_width=1,
            border_color=self.COLORS["title_border"],
        )
        self.outer_frame.pack(fill="both", expand=True)

        self.header_canvas = tk.Canvas(
            self.outer_frame,
            width=self.WINDOW_WIDTH,
            height=self.HEADER_HEIGHT,
            bg=self.COLORS["header_left"],
            highlightthickness=0,
            bd=0,
        )
        self.header_canvas.pack(fill="x")
        self._draw_header_gradient()

        self.header_canvas.create_text(
            95,
            54,
            text="Kai,",
            fill="#FFFFFF",
            font=("Segoe UI", 22, "bold"),
            anchor="w",
            tags=("drag",),
        )
        self.header_canvas.create_text(
            95,
            82,
            text="Your AI Agent",
            fill="#D7D7F5",
            font=("Segoe UI", 12),
            anchor="w",
            tags=("drag",),
        )

        self.logo_canvas = tk.Canvas(
            self.header_canvas,
            width=56,
            height=56,
            bg=self.COLORS["header_left"],
            highlightthickness=0,
        )
        self.logo_canvas.place(x=26, y=42)
        self.logo_rings = [
            self.logo_canvas.create_oval(6, 6, 50, 50, outline="#FFFFFF", width=2),
            self.logo_canvas.create_oval(13, 13, 43, 43, outline="#FFFFFF", width=2),
            self.logo_canvas.create_oval(20, 20, 36, 36, outline="#FFFFFF", width=2),
        ]
        self._animate_logo_pulse(0)

        self.min_text = self.header_canvas.create_text(
            self.WINDOW_WIDTH - 34,
            16,
            text="-",
            fill="#FFFFFF",
            font=("Segoe UI", 13),
            tags=("min_btn",),
        )
        self.close_text = self.header_canvas.create_text(
            self.WINDOW_WIDTH - 14,
            16,
            text="x",
            fill="#FFFFFF",
            font=("Segoe UI", 13),
            tags=("close_btn",),
        )
        self.header_canvas.tag_bind("min_btn", "<Button-1>", lambda _: self._minimize_window())
        self.header_canvas.tag_bind("close_btn", "<Button-1>", lambda _: self.request_exit())
        self.header_canvas.tag_bind("min_btn", "<Enter>", lambda _: self.header_canvas.itemconfigure(self.min_text, fill="#EDEDFD"))
        self.header_canvas.tag_bind("min_btn", "<Leave>", lambda _: self.header_canvas.itemconfigure(self.min_text, fill="#FFFFFF"))
        self.header_canvas.tag_bind("close_btn", "<Enter>", lambda _: self.header_canvas.itemconfigure(self.close_text, fill="#EDEDFD"))
        self.header_canvas.tag_bind("close_btn", "<Leave>", lambda _: self.header_canvas.itemconfigure(self.close_text, fill="#FFFFFF"))

        self.shadow_strip = ctk.CTkFrame(
            self.outer_frame,
            fg_color=self.COLORS["header_shadow"],
            height=2,
            corner_radius=0,
        )
        self.shadow_strip.pack(fill="x")

        for evt in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            self.header_canvas.tag_bind("drag", evt, getattr(self, {"<ButtonPress-1>": "_start_drag", "<B1-Motion>": "_on_drag", "<ButtonRelease-1>": "_on_drag_release"}[evt]))
        self.header_canvas.bind("<ButtonPress-1>", self._start_drag)
        self.header_canvas.bind("<B1-Motion>", self._on_drag)
        self.header_canvas.bind("<ButtonRelease-1>", self._on_drag_release)

        self.chat_scroll = ctk.CTkScrollableFrame(
            self.outer_frame,
            fg_color=self.COLORS["chat_bg"],
            corner_radius=0,
            border_width=0,
            scrollbar_button_color=self.COLORS["thumb"],
            scrollbar_button_hover_color="#3A3A3C",
        )
        self.chat_scroll.pack(fill="both", expand=True)
        self.chat_scroll.grid_columnconfigure(0, weight=1)

        self.status_frame = ctk.CTkFrame(
            self.outer_frame,
            fg_color=self.COLORS["input_bg"],
            corner_radius=0,
            height=20,
        )
        self.status_frame.pack(fill="x")
        self.status_frame.pack_propagate(False)

        left_status = ctk.CTkFrame(self.status_frame, fg_color="transparent")
        left_status.pack(side="left", padx=10)
        self.status_dot_canvas = tk.Canvas(
            left_status,
            width=8,
            height=8,
            bg=self.COLORS["input_bg"],
            highlightthickness=0,
        )
        self.status_dot_canvas.pack(side="left", pady=6)
        self.status_dot_item = self.status_dot_canvas.create_oval(2, 2, 6, 6, fill="#333333", outline="")
        self.status_label = ctk.CTkLabel(
            left_status,
            text="idle",
            text_color="#444444",
            font=("Segoe UI", 10),
        )
        self.status_label.pack(side="left", padx=(4, 0), pady=(0, 1))

        self.goal_text = ctk.CTkLabel(
            self.status_frame,
            text="goal: none",
            text_color="#444444",
            font=("Segoe UI", 10),
            cursor="hand2",
        )
        self.goal_text.pack(side="right", padx=8)
        self.goal_text.bind("<Button-1>", self._start_goal_edit)

        input_border = ctk.CTkFrame(self.outer_frame, fg_color=self.COLORS["title_border"], corner_radius=0, height=1)
        input_border.pack(fill="x")

        self.input_frame = ctk.CTkFrame(
            self.outer_frame,
            fg_color=self.COLORS["input_bg"],
            corner_radius=0,
            height=56,
        )
        self.input_frame.pack(fill="x")
        self.input_frame.pack_propagate(False)

        self.input_pill = ctk.CTkFrame(
            self.input_frame,
            fg_color=self.COLORS["input_field"],
            corner_radius=20,
            border_width=0,
            border_color=self.COLORS["accent_strong"],
            width=self.WINDOW_WIDTH - 32,
            height=38,
        )
        self.input_pill.place(x=16, y=9)

        self.entry = tk.Entry(
            self.input_pill,
            bg=self.COLORS["input_field"],
            fg=self.COLORS["muted"],
            insertbackground=self.COLORS["accent"],
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 12),
        )
        self.entry.place(x=14, y=9, width=self.WINDOW_WIDTH - 90, height=20)
        self.entry.insert(0, "Write a goal or /command...")
        self.entry.bind("<Return>", lambda _: self.on_send())
        self.entry.bind("<FocusIn>", self._on_entry_focus_in)
        self.entry.bind("<FocusOut>", self._on_entry_focus_out)

        self.send_circle = ctk.CTkFrame(
            self.input_pill,
            fg_color=self.COLORS["accent_strong"],
            corner_radius=13,
            width=26,
            height=26,
        )
        self.send_circle.place(x=self.WINDOW_WIDTH - 70, y=6)
        self.send_arrow = ctk.CTkLabel(
            self.send_circle,
            text="↑",
            text_color="#FFFFFF",
            font=("Segoe UI", 12, "bold"),
            cursor="hand2",
        )
        self.send_arrow.place(relx=0.5, rely=0.5, anchor="center")
        self.send_arrow.bind("<Button-1>", lambda _: self.on_send())
        self.send_circle.bind("<Button-1>", lambda _: self.on_send())

        self.command_chip_layer = ctk.CTkFrame(self.outer_frame, fg_color="transparent", height=16)
        self.command_chip_buttons: list[Any] = []
        self.command_chip_separators: list[Any] = []
        commands = ["step", "approve", "reject", "auto", "stop", "reset"]
        for idx, command in enumerate(commands):
            chip = ctk.CTkLabel(
                self.command_chip_layer,
                text=command,
                text_color="#444444",
                font=("Segoe UI", 10),
                cursor="hand2",
            )
            chip.pack(side="left")
            chip.bind("<Enter>", lambda _, c=chip: c.configure(text_color="#9B8FF0"))
            chip.bind("<Leave>", lambda _, c=chip: c.configure(text_color="#555555"))
            chip.bind("<Button-1>", lambda _, cmd=command: self._run_chip_command(cmd))
            self.command_chip_buttons.append(chip)
            if idx < len(commands) - 1:
                sep = ctk.CTkLabel(
                    self.command_chip_layer,
                    text=" · ",
                    text_color="#444444",
                    font=("Segoe UI", 10),
                )
                sep.pack(side="left")
                self.command_chip_separators.append(sep)

        self.window.bind("<Map>", self._restore_frameless_after_minimize)

        self._append_chat("assistant", "Chat ready. Enter a goal or /step.")
        self._append_chat("assistant", "Commands: /step /approve /reject /auto /stop /status /goal /reset")

    def _draw_header_gradient(self) -> None:
        self.header_canvas.delete("grad")
        width = self.WINDOW_WIDTH
        steps = 110
        start = (0x5B, 0x4F, 0xE8)
        end = (0x4B, 0x8B, 0xF5)
        for i in range(steps):
            t = i / max(1, steps - 1)
            r = int(start[0] + (end[0] - start[0]) * t)
            g = int(start[1] + (end[1] - start[1]) * t)
            b = int(start[2] + (end[2] - start[2]) * t)
            color = f"#{r:02x}{g:02x}{b:02x}"
            x0 = int(i * width / steps)
            x1 = int((i + 1) * width / steps)
            self.header_canvas.create_rectangle(x0, 0, x1, self.HEADER_HEIGHT, fill=color, outline=color, tags=("grad",))
        self.header_canvas.tag_lower("grad")

    def _animate_logo_pulse(self, tick: int) -> None:
        phase = (tick % 40) / 39.0
        if phase <= 0.5:
            amp = 0.4 + phase * 1.2
        else:
            amp = 1.0 - (phase - 0.5) * 1.2

        def blend(base: int) -> str:
            v = int(base + (255 - base) * amp * 0.35)
            return f"#{v:02x}{v:02x}{v:02x}"

        colors = [blend(180), blend(160), blend(140)]
        for ring, color in zip(self.logo_rings, colors):
            self.logo_canvas.itemconfigure(ring, outline=color)
        self.window.after(50, lambda: self._animate_logo_pulse(tick + 1))

    def _reposition_bottom_right(self) -> None:
        self.window.update_idletasks()
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = max(0, screen_width - self.WINDOW_WIDTH - self.WINDOW_X_MARGIN)
        y = max(0, screen_height - self.WINDOW_HEIGHT - self.WINDOW_Y_MARGIN)
        self.window.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}+{x}+{y}")

    def _start_goal_edit(self, _: object) -> None:
        if not hasattr(self, "goal_text"):
            return
        if self._goal_editing:
            return
        self._goal_editing = True
        current_goal = (get_goal() or "").strip()
        self.goal_edit = tk.Entry(
            self.status_frame,
            bg=self.COLORS["input_bg"],
            fg=self.COLORS["muted"],
            insertbackground=self.COLORS["accent"],
            relief="flat",
            bd=0,
            highlightthickness=0,
            font=("Segoe UI", 10),
        )
        self.goal_edit.insert(0, current_goal)
        self.goal_text.pack_forget()
        self.goal_edit.pack(side="right", padx=8)
        self.goal_edit.focus_set()
        self.goal_edit.bind("<Return>", self._commit_goal_edit)
        self.goal_edit.bind("<FocusOut>", self._commit_goal_edit)

    def _commit_goal_edit(self, _: object) -> None:
        if not self._goal_editing:
            return
        new_goal = self.goal_edit.get().strip()
        set_goal(new_goal)
        self.goal_edit.destroy()
        self._goal_editing = False
        if hasattr(self, "goal_text"):
            self.goal_text.pack(side="right", padx=8)
        self._refresh_goal_label()

    def _on_entry_focus_in(self, _: object) -> None:
        if self._entry_has_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.configure(fg=self.COLORS["text"])
            self._entry_has_placeholder = False
        self.input_pill.configure(border_width=1)
        self._show_command_chips()

    def _on_entry_focus_out(self, _: object) -> None:
        self.window.after(120, self._hide_chips_if_needed)
        self.input_pill.configure(border_width=0)
        if not self.entry.get().strip():
            self.entry.delete(0, tk.END)
            self.entry.insert(0, "Write a goal or /command...")
            self.entry.configure(fg=self.COLORS["muted"])
            self._entry_has_placeholder = True

    def _show_command_chips(self) -> None:
        if self._commands_visible:
            return
        self._commands_visible = True
        self.command_chip_layer.place(relx=0.5, anchor="s", y=self.WINDOW_HEIGHT - 64)
        for chip in self.command_chip_buttons:
            chip.configure(text_color="#444444")
        for sep in self.command_chip_separators:
            sep.configure(text_color="#444444")

        def animate(step: int = 0) -> None:
            if not self._commands_visible:
                return
            y = self.WINDOW_HEIGHT - 64 - min(step * 3, 12)
            self.command_chip_layer.place_configure(y=y)
            tint = ["#444444", "#505050", "#5A5A5A", "#666666", "#777777"]
            idx = min(step, len(tint) - 1)
            for chip in self.command_chip_buttons:
                chip.configure(text_color=tint[idx])
            for sep in self.command_chip_separators:
                sep.configure(text_color=tint[max(0, idx - 1)])
            if step < 4:
                self.window.after(25, lambda: animate(step + 1))

        animate()

    def _hide_chips_if_needed(self) -> None:
        focus_widget = self.window.focus_get()
        if focus_widget == self.entry or focus_widget in self.command_chip_buttons:
            return
        self._hide_command_chips()

    def _hide_command_chips(self) -> None:
        if not self._commands_visible:
            return

        def animate(step: int = 0) -> None:
            y = self.WINDOW_HEIGHT - 76 + min(step * 3, 12)
            self.command_chip_layer.place_configure(y=y)
            tint = ["#777777", "#666666", "#5A5A5A", "#505050", "#444444"]
            idx = min(step, len(tint) - 1)
            for chip in self.command_chip_buttons:
                chip.configure(text_color=tint[idx])
            if step < 4:
                self.window.after(20, lambda: animate(step + 1))
                return
            self.command_chip_layer.place_forget()
            self._commands_visible = False

        animate()

    def _run_chip_command(self, command: str) -> None:
        self.entry.delete(0, tk.END)
        self.entry.configure(fg=self.COLORS["text"])
        self._entry_has_placeholder = False
        self.entry.insert(0, f"/{command}")
        self.on_send()

    def _minimize_window(self) -> None:
        self.window.overrideredirect(False)
        self.window.iconify()

    def _restore_frameless_after_minimize(self, _: object) -> None:
        if self.window.state() == "normal":
            self.window.overrideredirect(True)
            self._reposition_bottom_right()

    def _start_drag(self, event: Any) -> None:
        self._title_drag_offset = (event.x_root, event.y_root)

    def _on_drag(self, event: Any) -> None:
        start_x, start_y = self._title_drag_offset
        dx = event.x_root - start_x
        dy = event.y_root - start_y
        self.window.geometry(f"+{self.window.winfo_x() + dx}+{self.window.winfo_y() + dy}")
        self._title_drag_offset = (event.x_root, event.y_root)

    def _on_drag_release(self, _: object) -> None:
        # Keep the window where the user drops it.
        return

    def _set_status(self, state: str) -> None:
        self._status_state = state
        self._update_status_text()

    def _set_typing(self, active: bool) -> None:
        self._typing_active = active

    def _update_status_text(self) -> None:
        current = get_current_step()
        budget = get_dynamic_max_steps()
        if self._status_state == "AUTOPILOT":
            line = f"autopilot · step {current} / budget {budget}"
        elif self._status_state == "EXECUTING":
            line = f"executing step {max(1, current)}"
        elif self._status_state == "THINKING":
            line = "thinking..."
        else:
            line = "idle"
        self.status_label.configure(text=line)

    def _animate_status_dot(self) -> None:
        active_color = self.STATUS_COLORS.get(self._status_state, self.COLORS["status_idle"])
        if self._status_state == "IDLE":
            dot_color = self.COLORS["status_idle"]
        else:
            self._status_animation_tick += 1
            dot_color = active_color if self._status_animation_tick % 2 == 0 else "#4E4A79"
        self.status_dot_canvas.itemconfig(self.status_dot_item, fill=dot_color)
        self.window.after(500, self._animate_status_dot)

    def _poll_stats(self) -> None:
        self._update_status_text()
        self._refresh_goal_label()
        self.window.after(1000, self._poll_stats)

    def _truncate_goal(self, text: str, limit: int = 18) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 1].rstrip() + "..."

    @staticmethod
    def _now_timestamp() -> str:
        return time.strftime("%H:%M:%S")

    def _classify_message(self, role: str, message: str) -> tuple[str, str | None, str]:
        lower = message.lower()
        if role == "you":
            return "user", None, message

        if lower.startswith("[status]"):
            return "system", None, message

        if any(token in lower for token in ["approved", "approve"]):
            return "action", "approve", message
        if any(token in lower for token in ["rejected", "reject"]):
            return "action", "reject", message
        if any(token in lower for token in ["executing", "execution"]):
            return "action", "execute", message

        if message.startswith("[") and "]" in message:
            return "system", None, message

        return "assistant", None, message

    def _append_chat(self, role: str, message: str) -> None:
        msg_type, _action_type, text = self._classify_message(role, message)
        self._set_typing(False)

        was_hidden = self.window.state() == "withdrawn"

        row = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        row.grid(sticky="ew", padx=16, pady=5)
        row.grid_columnconfigure(0, weight=1)

        if msg_type == "user":
            bubble = ctk.CTkFrame(
                row,
                fg_color="#2A2A2E",
                corner_radius=14,
                border_width=0,
            )
            bubble.grid(row=0, column=0, sticky="e")
            ctk.CTkLabel(
                bubble,
                text="you",
                text_color=self.COLORS["muted"],
                font=("Consolas", 10),
            ).pack(anchor="e", padx=10, pady=(6, 0))
            label = ctk.CTkLabel(
                bubble,
                text=text,
                justify="right",
                text_color="#888888",
                wraplength=250,
                font=("Segoe UI", 12),
            )
            label.pack(anchor="e", padx=10, pady=(0, 7))
            fade_palette = ["#777777", "#AAAAAA", self.COLORS["text_user"]]
        elif msg_type == "assistant":
            block = ctk.CTkFrame(row, fg_color="transparent")
            block.grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                block,
                text="kai",
                text_color=self.COLORS["accent"],
                font=("Consolas", 10),
            ).pack(anchor="w")
            label = ctk.CTkLabel(
                block,
                text=text,
                justify="left",
                text_color="#7B7B7B",
                wraplength=320,
                font=("Segoe UI", 12),
            )
            label.pack(anchor="w", pady=(0, 2))
            fade_palette = ["#7A7A7A", "#B7B7B7", self.COLORS["text"]]
        else:
            label = ctk.CTkLabel(
                row,
                text=text,
                justify="center",
                text_color="#3A3A3A",
                wraplength=340,
                font=("Segoe UI", 10, "italic"),
            )
            label.grid(row=0, column=0)
            fade_palette = ["#383838", "#414141", "#444444"]

        self._chat_rows.append({"row": row, "label": label})
        self._fade_label_in(label, fade_palette, 0)
        self._smooth_scroll_to_bottom()

        if was_hidden and role != "you":
            self.window.deiconify()
            self._reposition_bottom_right()
            self.window.lift()
            if self._auto_peek_job is not None:
                self.window.after_cancel(self._auto_peek_job)
            self._auto_peek_job = self.window.after(2000, self.hide_window)

    def append_message(self, role: str, message: str) -> None:
        """Public compatibility API for external chat append callers."""
        self._append_chat(role, message)

    def _fade_label_in(self, label: Any, palette: list[str], idx: int) -> None:
        if idx >= len(palette):
            return
        label.configure(text_color=palette[idx])
        self.window.after(40, lambda: self._fade_label_in(label, palette, idx + 1))

    def _smooth_scroll_to_bottom(self, step: int = 0) -> None:
        canvas = getattr(self.chat_scroll, "_parent_canvas", None)
        if canvas is None:
            return
        target = 1.0
        try:
            current = float(canvas.yview()[1])
        except Exception:
            return
        if current >= target - 0.01 or step >= 8:
            canvas.yview_moveto(target)
            return
        canvas.yview_moveto(min(target, current + 0.15))
        self.window.after(14, lambda: self._smooth_scroll_to_bottom(step + 1))

    def hide_window(self) -> None:
        self._hide_command_chips()
        self.window.withdraw()

    def on_send(self) -> None:
        text = self.entry.get().strip()
        self.entry.delete(0, "end")

        if self._entry_has_placeholder:
            text = ""

        if not text:
            self.on_next_step()
            if self._entry_has_placeholder:
                return
            self.entry.insert(0, "Write a goal or /command...")
            self.entry.configure(fg=self.COLORS["muted_soft"])
            self._entry_has_placeholder = True
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
                "Commands: /step, /approve, /reject, /auto, /autopilot on, /autopilot off, /stop, /status, /goal <text>, /reset, /telemetry, /memory, /benchmark",
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

        if lower == "/telemetry":
            self._append_chat("assistant", self._format_telemetry_table())
            return

        if lower == "/memory":
            self._append_chat("assistant", self._session_memory.get_summary())
            return

        if lower == "/benchmark":
            self._append_chat("assistant", "[BENCHMARK] Running quick benchmark suite (5 easy tasks)...")
            worker = threading.Thread(target=self._run_benchmark_worker, daemon=True)
            worker.start()
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

        self._set_status("THINKING")
        self._set_typing(True)
        worker = threading.Thread(target=self._run_router_worker, args=(text,), daemon=True)
        worker.start()

        self.entry.insert(0, "Write a goal or /command...")
        self.entry.configure(fg=self.COLORS["muted_soft"])
        self._entry_has_placeholder = True

    def on_next_step(self) -> None:
        if is_autopilot_enabled():
            self._append_chat("assistant", "[AUTO] Disable autopilot first using /stop.")
            return

        if not self._step_lock.acquire(blocking=False):
            self._append_chat("assistant", "Already analyzing. Please wait.")
            return

        self._set_status("THINKING")
        self._set_typing(True)
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
        self._clear_action_preview()
        self._append_chat("assistant", "Executing approved action...")
        worker = threading.Thread(target=self._run_execute_worker, args=(action,), daemon=True)
        worker.start()

    def on_reject(self) -> None:
        if self._pending_action is None:
            self._append_chat("assistant", "No pending action to reject.")
            return

        self._pending_action = None
        self._clear_action_preview()
        self._append_chat("assistant", "Pending action rejected. No execution performed.")

    def on_reset(self) -> None:
        disable_autopilot()
        clear_state()
        clear_intent_cache()
        self._pending_action = None
        self._clear_action_preview()
        self._refresh_goal_label()
        self._animate_reset_clear(lambda: self._append_chat("assistant", "Goal and last action were cleared."))
        self._set_status("IDLE")

    def on_auto(self) -> None:
        goal = (get_goal() or "").strip()
        if not goal:
            self._append_chat("assistant", "[AUTO] Cannot start autopilot: set a goal first.")
            return

        similar_tasks = self._session_memory.get_similar_tasks(goal, top_k=3)
        if similar_tasks:
            best = similar_tasks[0]
            step_hint = len(best.actions)
            status_text = "succeeded" if best.success else "failed"
            self._append_chat(
                "assistant",
                f"[MEMORY] Similar past task: \"{best.goal}\" -> {status_text} in {step_hint} steps",
            )

        if is_autopilot_enabled():
            self._append_chat("assistant", "[AUTO] Autopilot already running.")
            return

        if not self._step_lock.acquire(blocking=False):
            self._append_chat("assistant", "[AUTO] Assistant is busy. Try again shortly.")
            return

        self._pending_action = None
        self._set_status("AUTOPILOT")
        self._set_typing(True)

        worker = threading.Thread(target=self._run_autopilot_worker, daemon=True)
        worker.start()

    def on_stop_autopilot(self) -> None:
        if is_autopilot_enabled():
            disable_autopilot()
            self._append_chat("assistant", "[AUTO] Stop requested. Finishing current action safely.")
            self._set_status("IDLE")
            self._set_typing(False)
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
            self._enqueue_ui(self._set_typing, False)
            self._enqueue_ui(self._set_status, "IDLE")
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
        goal = get_goal() or ""
        current = get_current_step()
        budget = get_dynamic_max_steps()
        remaining = get_remaining_budget()
        return (
            f"[STATUS] Autopilot: {mode} | Step: {current}/{budget} | "
            f"Budget: dynamic (remaining {remaining}) | Goal: \"{goal}\""
        )

    @staticmethod
    def _format_telemetry_table() -> str:
        summary = get_telemetry_summary()
        labels = summary.get("labels", [])
        if not labels:
            return "[TELEMETRY] No inference telemetry available yet."

        lines = ["Label | Avg Conf | Detections | Status"]
        for row in labels:
            label = str(row.get("label", "unknown"))
            avg = float(row.get("avg_conf", 0.0))
            detections = int(row.get("detections", 0))
            status = str(row.get("status", "OK"))
            if status.upper() == "WEAK":
                status = "WEAK WARNING"
            lines.append(f"{label} | {avg:.2f} | {detections} | {status}")
        return "\n".join(lines)

    def _run_benchmark_worker(self) -> None:
        try:
            results = run_quick_benchmark_suite()
            passed = sum(1 for item in results if bool(item.get("passed")))
            total = len(results)
            self._enqueue_ui(
                self._append_chat,
                "assistant",
                f"[BENCHMARK] Completed quick suite: {passed}/{total} passed.",
            )
        except Exception:
            self._enqueue_ui(
                self._append_chat,
                "assistant",
                f"[BENCHMARK] Failed:\n{traceback.format_exc()}",
            )

    def _router_status(self, message: str) -> None:
        self._enqueue_ui(self._append_chat, "assistant", f"[ROUTER] {message}")

    def _run_router_worker(self, command: str) -> None:
        try:
            if is_autopilot_enabled():
                self._enqueue_ui(
                    self._append_chat,
                    "assistant",
                    "[ROUTER] Command ignored while autopilot is active. Use /stop first.",
                )
                return

            desktop_result = self._desktop_router.route(command)
            if desktop_result.handled:
                if desktop_result.success:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[DESKTOP] {desktop_result.method} succeeded. {desktop_result.output}",
                    )
                else:
                    self._enqueue_ui(
                        self._append_chat,
                        "assistant",
                        f"[DESKTOP] {desktop_result.method} failed: {desktop_result.error}",
                    )
                return

            self._enqueue_ui(
                self._append_chat,
                "assistant",
                f"[DESKTOP] {desktop_result.output}",
            )

            result = self._router.execute(command)
            if result.success:
                self._enqueue_ui(
                    self._append_chat,
                    "assistant",
                    f"[ROUTER] {result.method} succeeded in {result.duration_ms:.1f}ms. {result.output}",
                )
                if result.method == "vision_fallback":
                    set_goal(command)
                    self._enqueue_ui(self._refresh_goal_label)
            else:
                self._enqueue_ui(
                    self._append_chat,
                    "assistant",
                    f"[ROUTER] {result.method} failed in {result.duration_ms:.1f}ms: {result.error}",
                )
        except Exception:
            self._enqueue_ui(
                self._append_chat,
                "assistant",
                f"[ROUTER] Worker error:\n{traceback.format_exc()}",
            )
        finally:
            self._enqueue_ui(self._set_typing, False)
            self._enqueue_ui(self._set_status, "IDLE")

    @staticmethod
    def _log_autopilot_error(trace_text: str) -> None:
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        date_tag = time.strftime("%Y-%m-%d")
        out_path = logs_dir / f"errors_{date_tag}.log"
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(trace_text)
            if not trace_text.endswith("\n"):
                handle.write("\n")

    def _run_autopilot_worker(self) -> None:
        goal = (get_goal() or "").strip()
        max_steps = get_dynamic_max_steps()
        consecutive_failures = 0
        executed_actions: list[str] = []
        app_context_name = "unknown"
        run_success = False

        enable_autopilot(step_limit=max_steps)
        self._enqueue_ui(self._set_status, "AUTOPILOT")
        self._enqueue_ui(self._set_typing, True)
        self._enqueue_ui(self._append_chat, "assistant", f"[AUTO] Enabled. Goal: {goal}")

        try:
            for step in range(1, max_steps + 1):
                if not is_autopilot_enabled():
                    self._enqueue_ui(self._append_chat, "assistant", "[AUTO] Interrupted by user.")
                    break

                set_current_step(step)
                self._enqueue_ui(self._set_status, "EXECUTING")
                prev_state = observe_state(max_width=1280)
                app_context_name = getattr(prev_state.payload.app_context, "app_name", "unknown")
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
                    run_success = True
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
                            goal=goal,
                            prev_elements=candidate_prev_elements,
                        )
                        if not success.success:
                            consecutive_failures += 1
                            candidate_success = False
                            self._enqueue_ui(
                                self._append_chat,
                                "assistant",
                                f"[VERIFY] candidate #{idx}.{sub_idx} failed ({success.method}); trying next candidate.",
                            )
                            if consecutive_failures >= 2:
                                break
                            break

                        consecutive_failures = 0
                        set_last_action(candidate_action)
                        executed_actions.append(candidate_action.action.value)
                        candidate_prev_image = next_image
                        self._enqueue_ui(
                            self._append_chat,
                            "assistant",
                            f"[VERIFY] Step {step}: success on candidate #{idx}.{sub_idx} ({success.method})",
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
                        goal=goal,
                        prev_elements=fallback_prev_elements,
                    )
                    if not verified.success:
                        consecutive_failures += 1
                        fallback_success = False
                        self._enqueue_ui(
                            self._append_chat,
                            "assistant",
                            f"[VERIFY] fallback candidate #{sub_idx} failed ({verified.method}).",
                        )
                        if consecutive_failures >= 2:
                            break
                        break

                    consecutive_failures = 0
                    set_last_action(fallback_action)
                    executed_actions.append(fallback_action.action.value)
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
                run_success = len(executed_actions) > 0 and consecutive_failures == 0
        except Exception:
            trace_text = traceback.format_exc()
            self._log_autopilot_error(trace_text)
            disable_autopilot()
            self._enqueue_ui(self._append_chat, "assistant", f"[AUTO] Runtime error:\n{trace_text}")
        finally:
            if run_success:
                self._session_memory.add_task(
                    goal=goal,
                    actions=executed_actions,
                    success=True,
                    app_context=app_context_name,
                )
            disable_autopilot()
            self._enqueue_ui(self._append_chat, "assistant", "[AUTO] Disabled.")
            self._enqueue_ui(self._set_typing, False)
            self._enqueue_ui(self._set_status, "IDLE")
            self._step_lock.release()

    def _run_execute_worker(self, action: UIAction) -> None:
        try:
            self._enqueue_ui(self._set_status, "EXECUTING")
            self._enqueue_ui(self._set_typing, True)
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
        finally:
            self._enqueue_ui(self._set_typing, False)
            self._enqueue_ui(self._set_status, "IDLE")

    def _handle_proposed_action(self, action: UIAction) -> None:
        self._append_chat("assistant", self._format_action_summary(action))

        if action.action == ActionEnum.WAIT:
            self._pending_action = None
            self._clear_action_preview()
            self._append_chat("assistant", "No action executed because the safe decision is wait.")
            return

        self._pending_action = action
        self._render_action_preview(action)
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

    def _refresh_goal_label(self) -> None:
        goal = get_goal()
        self.goal_var.set(f"Goal: {goal if goal else 'None'}")
        shown = self._truncate_goal(goal if goal else "none")
        if hasattr(self, "goal_text"):
            self.goal_text.configure(text=f"goal: {shown}")

    def _render_action_preview(self, action: UIAction) -> None:
        self._pending_preview = (
            f"{action.action.value} -> {action.target_label or 'target'} @ {action.target_coordinates}"
        )

    def _clear_action_preview(self) -> None:
        self._pending_preview = None

    def _animate_reset_clear(self, callback: Callable[[], None] | None = None) -> None:
        rows_snapshot = list(self._chat_rows)
        if not rows_snapshot:
            if callback is not None:
                callback()
            return

        labels = [item["label"] for item in rows_snapshot]
        fades = ["#666666", "#4F4F4F", "#3A3A3A"]

        def fade_step(index: int) -> None:
            if index >= len(fades):
                for item in rows_snapshot:
                    item["row"].destroy()
                self._chat_rows = [item for item in self._chat_rows if item not in rows_snapshot]
                if callback is not None:
                    callback()
                return
            for label in labels:
                label.configure(text_color=fades[index])
            self.window.after(40, lambda: fade_step(index + 1))

        fade_step(0)

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
