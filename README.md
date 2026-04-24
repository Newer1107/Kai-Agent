# Kai Agent (Fast Iterative Desktop Loop)

Kai is a local Windows desktop agent optimized for speed and iteration.

Core loop:

1. Observe screen
2. Decide next reversible action
3. Execute action
4. Observe again
5. Refine until goal is complete

This replaces the older one-shot pipeline that waited too often.

## Design Priorities

1. Speed over heavy analysis
2. Iteration over paralysis
3. Reversible actions over perfect certainty
4. User control over execution mode (manual or autopilot)

## Runtime Architecture

- Hybrid perception (`perception.py`)
  - Every observe step combines YOLO detections, geometry heuristics, and optional OCR
  - Element payloads are enriched with region, text, and affordance metadata
  - YOLO runs on resized input with a short-lived cache to keep step latency low

- Fast goal parsing (`reasoning.py`)
  - `parse_goal()` handles click, type, scroll, enter, and open-app intent before any LLM call
  - Intent parsing is cached per goal string
  - Optional one-shot LLM parsing still exists behind `KAI_ENABLE_INTENT_LLM=1`

- Planner and loop controller (`agent_loop.py`)
  - `score_element()` blends confidence, center bias, shape bias, text similarity, size, and affordance score
  - `decide_action()` builds multi-step plans such as click → type → enter
  - `verify_success()` runs layered verification (fast visual checks, semantic completion, optional LLM verifier)
  - `build_retry_fallback_plan()` still exists, but now reuses the same plan builder as the main path

- Execution (`execution.py`)
  - Fast cursor movement and typing intervals remain in place
  - `enter` is a first-class action for form submission and confirmation steps
  - Confidence checks remain risk-based instead of hard-coded to one label source
  - Fail-safe and out-of-bounds protection disable autopilot safely when needed

- Universal command router (`universal_router.py`)
  - Layer 1 deterministic fast paths for common app/web/file/system/clipboard/typing actions
  - Layer 2 optional Ollama code-gen path with safety validation and bounded execution
  - Layer 3 vision fallback delegates to the main agent loop with dynamic step budgeting

## Key Guarantees

1. No global `confidence < X => wait` block
2. Heuristic fallback is mandatory when detection is weak
3. Search/type goals can use center input heuristics
4. Click goals can use best-available element fallback
5. Manual mode requires explicit user approval before each execution
6. Autopilot mode can execute multi-step loops without per-step approval
7. Autopilot is bounded by a dynamic step budget (estimated per goal)
8. Autopilot stops on repeated failures, no screen-change, or very low confidence
9. Emergency kill switch is always available (`Ctrl+Alt+S`)
10. The system re-observes after execution to propose corrective follow-up actions
11. Candidate retries are attempted before fallback (top-3 strategy)
12. Verification is mandatory after each autopilot action

## Performance Targets

- Screen capture: < 100 ms target
- YOLO inference: < 300 ms target
- Decision logic: < 50 ms target
- End-to-end step: ~1 second class target

You can inspect per-step timing in logs:

- `[PERF] capture=... decision=...`
- `[LOOP] ... capture=... decision=...`

## Project Structure

- `agent_loop.py`
  - Main iterative controller and step runner
- `reasoning.py`
  - Cached intent parsing (rule-first, optional one-shot LLM)
- `perception.py`
  - Fast screen capture + YOLO detection payload
- `detector.py`
  - YOLO model loading and inference
- `execution.py`
  - Action execution and coordinate resolution
- `main.py`
  - Chat UI, hotkeys, approval flow, router dispatch, and iterative follow-up proposal
- `universal_router.py`
  - Three-layer universal execution router for non-slash natural language commands

## Requirements

- Windows
- Python 3.11+
- Ollama (optional for intent parsing if `KAI_ENABLE_INTENT_LLM=1`)
- YOLO model weights (`runs/detect/train-14/weights/best.pt` preferred, fallback `yolov8n.pt`)

Install:

```powershell
pip install -r .\requirements.txt
```

Run:

```powershell
py .\main.py
```

## Hotkeys

- Open/focus chat: `Ctrl+Alt+Space`
- Reset goal/state: `Ctrl+Alt+R`
- Emergency stop autopilot: `Ctrl+Alt+S`
- Exit: `Ctrl+Alt+Q`

## Chat Commands

- `/step` or `Next Step`: Observe and propose the next plan
- `/approve`: Execute pending action
- `/reject`: Reject pending action
- `/auto`: Enable autopilot and run continuous steps (bounded)
- `/autopilot on`: Alias for enabling autopilot
- `/autopilot off`: Alias for stopping autopilot
- `/stop`: Stop autopilot immediately
- `/status`: Show autopilot state, dynamic budget, and remaining steps
- `/goal <text>`: Set goal
- `/reset`: Clear goal and action state
- `/telemetry`: Show rolling per-label detection confidence summary
- `/memory`: Show session-memory summary and app-context success distribution
- `/benchmark`: Run quick 5-task benchmark suite in the background

Non-slash chat input is routed through `UniversalRouter` in this order:

1. Deterministic fast path
2. Optional code generation path (if Ollama is reachable)
3. Vision-agent fallback

After a successful approval, Kai automatically observes the new screen and proposes the next plan.

## Autopilot Mode

Autopilot executes multi-step goals in a loop without requiring `/approve` on every step.

Loop behavior:

1. Observe current screen
2. Rank candidate targets using hybrid perception and affordance scoring
3. Build a short action plan such as click → type → enter
4. Execute each planned action with verification after every step
5. Trigger fallback (`[RETRY]`) if candidates fail
6. Continue loop until completion or safety stop

Autopilot stop conditions:

- step limit reached (`max_steps`)
- 2 consecutive action failures
- no meaningful screen change after action
- confidence below `0.40`
- high-risk action detection (safety block)
- `/stop` command or `Ctrl+Alt+S` hotkey

## Robust Targeting Details

Target ranking does not trust model label text directly. Each element is scored from:

- model confidence
- center proximity bias
- shape bias (input-like vs button-like geometry)
- size relevance
- goal alignment

Score blend (implemented in code):

- `0.4 * confidence`
- `0.2 * center_bias`
- `0.2 * shape_bias`
- `0.1 * size_relevance`
- `0.1 * goal_match`

Debug telemetry:

- `[DECISION]` ranked candidates and score breakdown
- `[ACTION]` candidate execution attempts
- `[VERIFY]` success/failure validation
- `[RETRY]` fallback triggers and retry outcomes

## Environment Variables

- `OLLAMA_MODEL` (default: `qwen2.5-coder:1.5b`)
- `OLLAMA_BASE_URL` (default: `http://localhost:11434/v1`)
- `OLLAMA_API_KEY` (default: `ollama`)
- `KAI_ENABLE_INTENT_LLM` (default: `0`)
- `KAI_ENABLE_OCR` (default: `0`; enables OCR enrichment and semantic checks)
- `KAI_ENABLE_VERIFY_LLM` (default: `0`; enables Layer 3 LLM verification with 2s timeout)
- `KAI_TELEMETRY_LOG` (default: `0`; when enabled, writes `logs/telemetry_YYYY-MM-DD.jsonl`)
- `KAI_PERSIST_MEMORY` (default: `0`; when enabled, persists successful tasks to `memory/session_YYYY-MM-DD.json`)

## Logging

- `logs/` is created on demand when writing runtime artifacts.
- Autopilot runtime exceptions are appended to `logs/errors_YYYY-MM-DD.log`.
- Telemetry JSONL files are created only when `KAI_TELEMETRY_LOG=1`.

## Notes

- This system is intentionally action-oriented and may take low-risk guesses.
- It is designed to recover on the next iteration rather than stall on uncertainty.
- Sensitive screens should be handled carefully; screenshots may contain private information.
