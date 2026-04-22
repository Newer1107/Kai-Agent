# Local AI Windows Assistant (Guided Safe Mode)

This project is a **local, goal-aware, safety-first desktop assistant prototype** for Windows.

It does **not** run autonomously. Instead, it works as a guided decision system with a docked chat window:

1. Opens/focuses a chat dock on hotkey
2. Accepts an optional user goal from the chat input
3. Captures and analyzes the current screen only when requested
4. Runs YOLOv8 structured detection to produce typed UI elements with centers
5. Optionally runs OCR for nearby text grounding
6. Enriches detections with semantics + layout regions
7. Uses the LLM to decide **what** to do from enriched structured UI elements
8. Uses deterministic target resolution (ranking + ambiguity checks) to decide **where** on screen to act
9. Requires user approval before any execution
10. Returns to idle after that single step

Default local model:

- `gemma4:e2b`

## What This System Is

- A cautious decision engine for one-step desktop actions
- Goal-aware during the current runtime session
- Strictly human-approved before each action

## What This System Is Not

- Not an autonomous loop
- Not self-chaining multi-step execution
- Not hidden-intent inference

## Core Safety Guarantees

The current implementation enforces:

1. Strict structured output (`UIAction`) with schema validation
2. **RISK-BASED EXECUTION POLICY**: Confidence thresholds vary by action type (type=0.3, click=0.4-0.5, scroll=0.7)
3. **REMOVED GLOBAL BLOCK**: No more blanket "confidence < 0.6 → wait" rule
4. **HEURISTIC SOURCES ALWAYS ALLOWED**: Elements from heuristic fallbacks execute with min confidence 0.25
5. **EMERGENCY FALLBACK**: If LLM fails entirely, use heuristic_action_fallback() with 0.65 confidence
6. If action is non-`wait` and coordinates are missing, action is forced to `wait`
7. If coordinates are out of bounds, action is forced to `wait`
8. If action is `wait`, nothing is executed
9. User approval is required before every executable action
10. Trigger lock prevents overlapping runs
11. Main loop catches exceptions so one failure does not crash the assistant
12. Actions are only executed after explicit chat approval (`Approve` or `/approve`)
13. **HARD RULE**: System NEVER returns `wait` if goal is clear AND screen contains UI elements (uses heuristic fallback instead)
14. Execution coordinates are grounded from YOLO detection centers or heuristic fallback centers in full-screen space
15. Process DPI awareness is enabled at startup to avoid logical-vs-physical pixel mismatch
16. **USABILITY OVER PERFECTION**: Typing in a wrong field is acceptable; doing nothing is not

## Project Structure

- `agent_state.py`
  - In-memory guided state
  - Stores `current_goal` and `last_action`
  - Provides `set_goal`, `get_goal`, `clear_goal`
  - Also includes `set_last_action`, `get_last_action`, and `clear_state`

- `schema.py`
  - Defines `ActionEnum` and `UIAction`
  - Includes goal-aware field `next_step_summary`
  - Enforces strict validation and low-confidence uncertainty requirements
  - Provides `safe_wait_action(...)` fallback constructor

- `perception.py`
  - Captures primary screen
  - Creates resized/base64 image for LLM analysis
  - Runs YOLO detection as part of capture flow
  - Optionally runs OCR (toggle via `KAI_ENABLE_OCR`)
  - Adds region + semantic enrichment to UI elements
  - Returns structured payload with `image_base64`, `ui_elements`, and `text_regions`

- `detector.py`
  - Loads model from `runs/detect/train-14/weights/best.pt` (fallback `yolov8n.pt`)
  - Caches model globally so it is loaded once per process
  - Runs YOLOv8 detection and filters confidence below `0.5`
  - Returns structured detections: `type`, `bbox`, `center`, `confidence`
  - Provides debug drawing overlay utility for detections

- `reasoning.py`
  - Calls local Ollama via OpenAI-compatible endpoint + `instructor`
  - Accepts image, `ui_elements`, optional `current_goal`, and optional `last_action`
  - Prompts for exactly one cautious next step and `target_label` using type + region + semantics
  - Allows approximate target inference when goal strongly suggests intent
  - **EMERGENCY FALLBACK**: If LLM fails (InstructorRetryException or network error), activates `heuristic_action_fallback()` to force action
  - Safety overrides relaxed when goal is clear: allows low-confidence actions if intent is explicit
  - Does not rely on LLM coordinates
  - Applies goal-aware safety overrides and JSON fallback parser for robust outputs
  - Debug logging with `[REASONING]`, `[FALLBACK]`, `[SAFETY]` prefixes

- `heuristics.py`
  - Provides fallback detection when YOLO fails or detections are low-confidence
  - Functions:
    - `infer_input_field(screen_width, screen_height)`: Returns center-biased input field estimate
    - `detect_input_by_shape(elements, screen_width, screen_height)`: Finds horizontally-stretched elements (aspect > 3.5) in CENTER region
    - `detect_by_goal_heuristic(elements, goal, screen_width, screen_height)`: Uses goal keywords to infer best fallback element
    - `is_safe_heuristic_location(element, screen_width, screen_height)`: Validates heuristic element center is in safe UI zone (not edges, not taskbar)
  - All heuristics tag elements with `source: "heuristic_*"` for tracking and relaxed confidence checks

- `policy.py`
  - **NEW**: Risk-based execution policy replacing global confidence blocks
  - Functions:
    - `get_action_risk(action)`: Classifies action as low/medium/high risk (type=low, click=medium, scroll=high)
    - `is_intent_clear(goal)`: Detects explicit user intent from goal keywords
    - `get_min_confidence_for_action(action, source, goal)`: Returns required confidence threshold based on risk profile
    - `is_action_allowed(action, confidence, source, goal)`: Determines if action should execute
  - Risk-based thresholds: type >= 0.3 (low risk), click >= 0.4-0.5 (medium), scroll >= 0.7 (high)
  - Heuristic sources (confidence >= 0.25) ALWAYS allowed (no global block)
  - Clear intent lowers thresholds for low/medium risk actions
  - Replaces "useless `wait`" with "useful guesses"

- `layout.py`
  - Assigns each UI element to `TOP_BAR`, `CENTER`, `LEFT_PANEL`, `RIGHT_PANEL`, or `FOOTER`

- `semantics.py`
  - Extracts OCR text regions (optional)
  - Attaches nearest text + semantic labels to UI elements

- `ranking.py`
  - Scores candidates by confidence, size, type relevance, position heuristics, and cluster context
  - **NEW**: CENTER region bias: +0.15 for center region, +0.15 for wide (w/h > 2.5) elements (total +0.30 boost)
  - Prioritizes horizontally-stretched center elements (typical search/input pattern)

- `resolver.py`
  - Resolves a safe target through:
    1. Strict type matching
    2. Semantic goal filtering + ranking
    3. Heuristic fallbacks (shape-based, goal-aware)
  - Goal-aware confidence relaxation: input fields allow >= 0.3 when search intent detected, others >= 0.5
  - Rejects ambiguous or low-confidence candidates (unless heuristic provides reasonable alternative)
  - Tags chosen element with `source: "yolo" | "heuristic_*"`
  - Prints debug trace for resolution decisions

- `debug_overlay.py`
  - Draws chosen target preview for explicit verification before approval/execution

- `execution.py`
  - Displays detailed proposal and asks for explicit confirmation
  - Provides coordinate utilities and guarded pyautogui execution
  - Resolves target coordinates from selected UI element `center`
  - Validates confidence and in-bounds coordinates before click/type/scroll execution
  - Enables `pyautogui.FAILSAFE`
  - Prints debug details and previews cursor location before action

- `main.py`
  - Enables Windows process DPI awareness at startup
  - Registers hotkeys and opens/focuses a chat dock window
  - Keeps the chat window anchored at bottom-right and always on top
  - Accepts goal and control commands via chat input
  - Passes goal + last action context to reasoning
  - Resolves LLM `target_label` using deterministic resolver logic
  - Forces `wait` on no match, low confidence, or ambiguous matches
  - Saves target preview overlay to show exactly what will be clicked
  - Uses in-chat approve/reject controls before each executable action
  - Updates `last_action` only after approved execution
  - Supports state reset and app-exit hotkeys

## Requirements

- Windows
- Python 3.11+
- Ollama running locally
- Model available locally: `gemma4:e2b`

Dependencies are listed in `requirements.txt`.

## Quick Start

### 1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
pip install -r .\requirements.txt
```

### 3) Ensure model is present

```powershell
ollama pull gemma4:e2b
```

### 4) Run assistant

```powershell
python .\main.py
```

Hotkeys:

- Open/focus chat dock: `Ctrl+Alt+Space`
- Reset goal + last action: `Ctrl+Alt+R`
- Exit assistant: `Ctrl+Alt+Q`

Chat behavior:

- The dock opens at bottom-right and stays topmost
- It is focused when opened from hotkey
- Enter text to set/update goal
- Use commands to control the flow:
  - `/next` or `/step`: capture + analyze one step
  - `/approve`: execute the currently pending proposed action
  - `/reject`: reject the currently pending proposed action
  - `/goal <text>`: set goal without analysis
  - `/reset`: clear goal + last action
  - `/help`: show available commands

## Runtime Behavior

On `Ctrl+Alt+Space`:

1. Chat dock opens/focuses at bottom-right
2. You provide a goal or issue a command

On `/next` or Next Step button:

1. Capture screen
2. Run YOLO detection and produce structured `ui_elements`
3. Optionally run OCR and attach semantic labels
4. Assign each element to a screen region
5. Analyze with current goal + last action context using screenshot + enriched `ui_elements`
6. Resolve `target_label` via semantic filtering + ranking + ambiguity checks
7. Save detection and chosen-target preview overlays
6. Show proposed step in chat:
  - `intent_summary`
  - `next_step_summary`
  - `action`
  - `target_label`
  - `confidence_score`
  - `uncertainty_reason`
  - target details
8. If `action == wait`, do nothing and return idle
9. If actionable, mark it as pending and wait for `/approve` or Approve button
10. If approved, execute one action and store as `last_action`
11. Return idle

No autonomous continuation occurs after a step.

## Coordinate System Notes

- LLM analysis runs on a resized screenshot for speed.
- YOLO detection runs on original-resolution screenshots for precise grounding.
- The LLM does not provide execution coordinates.
- Coordinates come from selected UI element centers in full-screen pixel space.
- During execution, debug logs print:
  - chosen target coordinates
  - original/resized sizes
  - current screen size

## Action Schema

`UIAction` fields:

- `intent_summary: str`
- `next_step_summary: str`
- `action: "click" | "type" | "scroll" | "wait"`
- `target_label: str | None`
- `target_description: str | None`
- `target_coordinates: tuple[int, int] | None`
- `text_to_type: str | None`
- `confidence_score: float` (`0.0` to `1.0`)
- `uncertainty_reason: str | None`

Validation rules:

- `uncertainty_reason` is required when `confidence_score < 0.7`
- `text_to_type` is required when `action == "type"`
- `target_coordinates` is populated by YOLO grounding, not by the LLM

## Environment Variables

Optional configuration:

- `OLLAMA_MODEL` (default: `gemma4:e2b`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
- `OLLAMA_API_KEY` (default fallback in code: `ollama`)
- `KAI_ENABLE_OCR` (`0` by default; set to `1` to enable OCR text extraction)

PowerShell example:

```powershell
$env:OLLAMA_MODEL = "gemma4:e2b"
$env:OLLAMA_BASE_URL = "http://localhost:11434/v1"
python .\main.py
```

## Troubleshooting

### `py .\main.py` exits with code 1

Common causes:

- Missing dependency in active interpreter
- Global hotkey permission issue
- Ollama not running or wrong base URL
- Missing YOLO/OpenCV dependencies
- OCR dependencies missing when OCR is enabled

Checks:

```powershell
python -c "import keyboard, pyautogui, instructor, pydantic, openai; print('imports ok')"
python -c "import ultralytics, cv2; print('vision deps ok')"
python -c "import pytesseract; print('ocr deps ok')"
curl http://localhost:11434/api/tags
```

### YOLO model weights

- Preferred custom model path: `runs/detect/train-14/weights/best.pt`.
- If that file is missing, the system falls back to `yolov8n.pt`.
- If both are unavailable, the assistant safely returns `wait` instead of blind clicks.

### `NameError: name 'self' is not defined` during approved action

If this appears from `execute_approved_action`, it means the method is being treated as static while reading instance debug state. Use the latest `main.py` where `execute_approved_action` is an instance method.

### Editor shows unresolved imports

- Activate the same venv used for install
- Select correct VS Code Python interpreter
- Reinstall dependencies if needed:

```powershell
pip install -r .\requirements.txt
```

### Hotkey not triggering

- Ensure process is still running
- Some systems need elevated terminal permissions
- Try running terminal as Administrator

### Chat window does not appear

- Press `Ctrl+Alt+Space` again to force focus
- Check if another app blocks focus-stealing behavior
- Verify Python Tk support is available:

```powershell
python -c "import tkinter; print('tk ok')"
```

### Chat window becomes unresponsive

- This build uses a thread-safe UI queue to avoid direct Tk calls from worker/hotkey threads
- If it still freezes, restart the app and test once with no other global hotkey tools active
- Keep interaction in the dock using `/step`, then `/approve` or `/reject`

### Assistant returns `wait` often

Expected in ambiguous states. The assistant prefers safe inaction when confidence or grounding is weak.

## Security and Privacy Notes

- Screenshots may contain sensitive information
- Keep inference local through Ollama
- Avoid logging raw screenshots unless redaction is implemented

## Extend Next

1. OCR pass before reasoning for better text grounding
2. Bounding-box verification prior to click execution
3. Action audit log with timestamps and goal context
4. Per-application safety profiles
5. Dry-run mode that never executes, only proposes

---

This repository is intentionally conservative and meant to be a reliable base for guided local desktop assistance, not autonomous control.