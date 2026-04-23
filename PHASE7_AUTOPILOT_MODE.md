# Phase 7: Safe Autopilot Mode

## Goal

Add a fully automated execution mode that can complete multi-step tasks without per-step approval, while remaining bounded, interruptible, and risk-aware.

## Implemented Components

## 1. Autopilot State

File: `agent_state.py`

Added state fields:

- `autopilot_enabled: bool = False`
- `max_steps: int = 5`
- `current_step: int = 0`

Added APIs:

- `enable_autopilot(step_limit: int | None = None)`
- `disable_autopilot()`
- `is_autopilot_enabled()`
- `get_max_steps()` / `set_max_steps(...)`
- `get_current_step()` / `set_current_step(...)`

State access is lock-protected for thread safety.

## 2. Chat Commands

File: `main.py`

Added commands:

- `/auto` -> starts autopilot loop
- `/stop` -> immediately disables autopilot
- `/status` -> prints ON/OFF and current step counter

Also added UI buttons:

- `Auto`
- `Stop`

## 3. Autopilot Loop

File: `main.py` (`AssistantChatWindow._run_autopilot_worker`)

Loop flow:

1. enable autopilot
2. for each step up to max_steps:
   - observe screen
   - propose action
   - log proposed action
   - execute directly (no approval)
   - sleep 0.5
   - observe again and verify progress
3. disable autopilot

## 4. Approval Bypass in Autopilot

In autopilot worker, execution uses:

- `self.assistant.execute_approved_action(action, approved=True)`

No `/approve` is required in autopilot mode.

## 5. Safety Stop Conditions

Autopilot stops when:

- 2 consecutive action failures
- no meaningful screen change detected after action
- confidence below 0.30
- high-risk action content detected
- step limit reached
- user stop command/hotkey triggers disable flag

## 6. Step Boundaries

Default max step limit:

- `max_steps = 5`

No infinite loops are allowed.

## 7. Visual Feedback

Chat outputs include:

- `[AUTO] Step N: ...`
- action summaries per step
- stop reasons and completion status

## 8. Emergency Kill Switch

File: `main.py`

Added hotkey:

- `Ctrl + Alt + S` -> `stop_autopilot_from_hotkey`

This disables autopilot state immediately and posts emergency stop feedback in chat.

## 9. Hybrid Mode

Autopilot behavior:

- skips approval
- logs each proposed action in chat
- can return to manual mode at any time via `/stop`

## 10. Risk-Aware Execution

Autopilot allows low-risk action classes (`click`, `type`, `scroll`) but blocks actions matching dangerous intent signals (e.g., delete/format/shutdown/system32/registry keywords).

## Notes

- Manual mode behavior is preserved.
- Autopilot and manual step workers are serialized by `_step_lock` to prevent overlap.
- `clear_state()` now resets autopilot runtime fields.
