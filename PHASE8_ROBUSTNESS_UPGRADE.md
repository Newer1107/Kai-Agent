# Phase 8: Robustness Upgrade (Scoring + Verification + Retry)

## Objective

Upgrade Kai from single-shot target selection to robust self-correcting execution:

- geometry-aware ranking
- multi-candidate retries
- mandatory post-action verification
- fallback behavior after failures

## Implemented Components

## 1. Advanced Target Scoring

File: `agent_loop.py`

Added:

- `score_element(element, goal, screen_size, target_hint=None)`
- `rank_elements(elements, goal, screen_size, top_k, target_hint)`

Scoring blend:

- confidence (0.4)
- center bias (0.2)
- shape bias (0.2)
- size relevance (0.1)
- goal match (0.1)

Label strings are treated only as weak signals, not as truth.

## 2. Geometry Heuristics

File: `agent_loop.py`

Added helpers:

- `is_probable_input(...)`
- `is_probable_button(...)`

Used during scoring and confidence adaptation.

## 3. Multi-Candidate Execution

Files: `agent_loop.py`, `main.py`

- `DecisionResult` now contains ranked candidates and candidate plans.
- Autopilot executes top-3 candidate actions before giving up.
- Candidate execution logs include `[ACTION] candidate #i/N`.

## 4. Action Verification

File: `agent_loop.py`

Added:

- `screen_changed(prev_image, new_image, threshold)`
- `verify_success(prev_image, new_image, action, prev_elements, new_elements)`

Verification checks:

- global pixel-diff change
- UI structure signature change
- typing local focus-region change near target coordinates

## 5. Retry and Fallback

Files: `agent_loop.py`, `main.py`

- if all top candidates fail, trigger fallback plan:
  - type/search intent -> center-input click heuristic
  - click intent -> highest-confidence element
- if fallback fails -> stop safely

## 6. Adaptive Confidence

File: `agent_loop.py`

Action confidence is adjusted using:

- shape strength
- goal clarity
- goal alignment
- fallback source

No global hard confidence block is used.

## 7. Telemetry

Runtime logs now include:

- `[DECISION]` candidate ranking and score breakdown
- `[ACTION]` candidate attempt execution
- `[VERIFY]` success/failure outcome
- `[RETRY]` fallback and retry events

## 8. Performance Notes

- YOLO runs once at step observation (`observe_state`) in autopilot path.
- Verification uses screenshot-only checks (no OCR, no LLM in loop).
- Intent parsing remains cached and optionally LLM-assisted once per goal.

## Safety Conditions Preserved

Autopilot still stops when:

- 2 consecutive failures
- confidence < 0.30
- high-risk action keywords detected
- max step limit reached
- user interrupt (`/stop` or `Ctrl+Alt+S`)
