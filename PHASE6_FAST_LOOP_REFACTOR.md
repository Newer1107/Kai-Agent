# Phase 6: Fast Iterative Agent Loop

## Objective

Refactor Kai from a one-shot reasoning pipeline to an iterative desktop control loop:

observe -> decide -> act -> observe -> refine

## What Changed

### 1. New Loop Controller

File: `agent_loop.py`

- Added `observe_state()` for fast screen capture and YOLO detection payloads
- Added `decide_action()` with lightweight target resolution and heuristic fallback
- Added `run_agent(goal, max_steps=3)` for iterative execution

### 2. Lightweight Intent Parsing

File: `reasoning.py`

- Removed instructor-based structured output and retry loops
- Added `ParsedIntent` and `parse_intent(goal)`
- Intent is cached by normalized goal
- Optional one-shot LLM parse is supported only when `KAI_ENABLE_INTENT_LLM=1`

### 3. Fast Perception Path

File: `perception.py`

- Default path is YOLO-only detections
- Semantic/layout enrichment moved to optional `enriched=True`
- Heavy OCR/semantic pipeline no longer used in default loop

### 4. Main Pipeline Rewire

File: `main.py`

- Removed dependency on old `analyze_screen()` + resolver path in runtime flow
- `LocalAssistant.propose_next_action()` now calls:
  - `observe_state()`
  - `decide_action()`
- After approved execution, assistant automatically proposes follow-up step from new observation

### 5. Faster Execution

File: `execution.py`

- Reduced pyautogui pause and movement durations
- Supports fallback to direct `target_coordinates` if selected element is missing
- Keeps risk-based confidence checks

### 6. Dependency Cleanup

File: `requirements.txt`

- Removed `instructor` dependency

## Mandatory Fallback Behavior

- Type/search goals:
  - Prefer input-like detections
  - Fallback to center input heuristic if needed
- Click goals:
  - Prefer resolved match
  - Fallback to highest-confidence element
- No elements:
  - return wait

## Expected Behavior Example

Goal: `search hello kitty`

1. Step 1: click inferred input
2. Step 2: type `hello kitty`
3. Step 3: wait (goal likely completed)

## Performance Logging

- `[PERF] capture=... decision=...`
- `[LOOP] ... capture=... decision=... fallback=...`

Use these logs to verify step latency and identify bottlenecks.
