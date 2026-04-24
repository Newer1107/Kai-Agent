from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_loop import run_agent, verify_success
from perception import capture_primary_screenshot
from schema import ActionEnum, UIAction, safe_wait_action


BENCHMARK_TASKS: list[dict[str, Any]] = [
    {
        "id": "browser_google_search",
        "goal": "open chrome and search kai agent",
        "app_context": "browser",
        "expected_outcome_keywords": ["results", "loaded", "found"],
        "max_steps": 8,
        "difficulty": "easy",
    },
    {
        "id": "browser_search_hello_kitty",
        "goal": "search hello kitty",
        "app_context": "browser",
        "expected_outcome_keywords": ["results", "loaded"],
        "max_steps": 6,
        "difficulty": "easy",
    },
    {
        "id": "browser_open_chatgpt",
        "goal": "open chatgpt and type hello",
        "app_context": "browser",
        "expected_outcome_keywords": ["active", "typed", "entered"],
        "max_steps": 9,
        "difficulty": "medium",
    },
    {
        "id": "browser_submit_query",
        "goal": "click search input and type desktop automation then press enter",
        "app_context": "browser",
        "expected_outcome_keywords": ["results", "confirmed"],
        "max_steps": 10,
        "difficulty": "medium",
    },
    {
        "id": "explorer_open_documents",
        "goal": "open file explorer and click documents",
        "app_context": "file explorer",
        "expected_outcome_keywords": ["active", "selected"],
        "max_steps": 7,
        "difficulty": "easy",
    },
    {
        "id": "explorer_select_downloads",
        "goal": "open file explorer then click downloads",
        "app_context": "file explorer",
        "expected_outcome_keywords": ["selected", "changed"],
        "max_steps": 8,
        "difficulty": "easy",
    },
    {
        "id": "explorer_sort_view",
        "goal": "open file explorer and click view then click details",
        "app_context": "file explorer",
        "expected_outcome_keywords": ["changed", "activated"],
        "max_steps": 10,
        "difficulty": "medium",
    },
    {
        "id": "notepad_type_line",
        "goal": "open notepad and type hello world",
        "app_context": "notepad",
        "expected_outcome_keywords": ["typed", "entered"],
        "max_steps": 6,
        "difficulty": "easy",
    },
    {
        "id": "notepad_type_paragraph",
        "goal": "open notepad and type kai agent test line",
        "app_context": "notepad",
        "expected_outcome_keywords": ["typed", "filled"],
        "max_steps": 7,
        "difficulty": "easy",
    },
    {
        "id": "notepad_multi_step",
        "goal": "open notepad then type automation demo and press enter",
        "app_context": "notepad",
        "expected_outcome_keywords": ["entered", "confirmed"],
        "max_steps": 8,
        "difficulty": "medium",
    },
    {
        "id": "calculator_open",
        "goal": "open calculator",
        "app_context": "calculator",
        "expected_outcome_keywords": ["launched", "active"],
        "max_steps": 5,
        "difficulty": "easy",
    },
    {
        "id": "calculator_click_digit",
        "goal": "open calculator and click 7",
        "app_context": "calculator",
        "expected_outcome_keywords": ["changed", "selected"],
        "max_steps": 7,
        "difficulty": "medium",
    },
    {
        "id": "calculator_simple_expression",
        "goal": "open calculator and click 2 then click plus then click 3",
        "app_context": "calculator",
        "expected_outcome_keywords": ["changed", "activated"],
        "max_steps": 10,
        "difficulty": "hard",
    },
    {
        "id": "settings_open",
        "goal": "open settings",
        "app_context": "system settings",
        "expected_outcome_keywords": ["launched", "visible"],
        "max_steps": 6,
        "difficulty": "easy",
    },
    {
        "id": "settings_search_bluetooth",
        "goal": "open settings and search bluetooth",
        "app_context": "system settings",
        "expected_outcome_keywords": ["results", "found"],
        "max_steps": 9,
        "difficulty": "medium",
    },
    {
        "id": "settings_click_display",
        "goal": "open settings then click display",
        "app_context": "system settings",
        "expected_outcome_keywords": ["selected", "active"],
        "max_steps": 8,
        "difficulty": "medium",
    },
    {
        "id": "vscode_open_file",
        "goal": "open vscode and click explorer",
        "app_context": "vscode",
        "expected_outcome_keywords": ["selected", "active"],
        "max_steps": 8,
        "difficulty": "medium",
    },
    {
        "id": "vscode_type_search",
        "goal": "open vscode and type kai",
        "app_context": "vscode",
        "expected_outcome_keywords": ["typed", "entered"],
        "max_steps": 7,
        "difficulty": "medium",
    },
    {
        "id": "browser_login_form",
        "goal": "open browser and click login then type username",
        "app_context": "browser",
        "expected_outcome_keywords": ["filled", "active"],
        "max_steps": 10,
        "difficulty": "hard",
    },
    {
        "id": "explorer_new_folder_flow",
        "goal": "open file explorer and click new then click folder",
        "app_context": "file explorer",
        "expected_outcome_keywords": ["created", "changed"],
        "max_steps": 11,
        "difficulty": "hard",
    },
]


def _consecutive_failure_stop(step_results: list[Any]) -> bool:
    failures = 0
    for result in step_results:
        if bool(result.executed):
            failures = 0
            continue
        failures += 1
        if failures >= 2:
            return True
    return False


def score_task(
    verification_success: bool,
    steps_used: int,
    max_steps: int,
    consecutive_failure_stop: bool,
) -> bool:
    return bool(verification_success) and int(steps_used) <= int(max_steps) and not bool(consecutive_failure_stop)


class BenchmarkRunner:
    def __init__(self, tasks: list[dict[str, Any]]) -> None:
        self.tasks = tasks

    def run(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        for task in self.tasks:
            started_at = time.perf_counter()
            before_image = capture_primary_screenshot()
            step_results = run_agent(goal=task["goal"], max_steps=int(task["max_steps"]), pause_seconds=0.2)
            after_image = capture_primary_screenshot()
            elapsed = time.perf_counter() - started_at

            steps_used = len(step_results)
            fallback_triggered = any(bool(step.decision.used_fallback) for step in step_results)
            stopped_consecutive_failures = _consecutive_failure_stop(step_results)

            final_action: UIAction = safe_wait_action(reason="No action executed during benchmark run.")
            for step in reversed(step_results):
                if step.decision.action.action != ActionEnum.WAIT:
                    final_action = step.decision.action
                    break

            verification = verify_success(
                prev_image=before_image,
                new_image=after_image,
                action=final_action,
                goal=str(task.get("goal", "")),
            )

            passed = score_task(
                verification_success=verification.success,
                steps_used=steps_used,
                max_steps=int(task["max_steps"]),
                consecutive_failure_stop=stopped_consecutive_failures,
            )

            result = {
                "task_id": task["id"],
                "goal": task["goal"],
                "app_context": task["app_context"],
                "difficulty": task["difficulty"],
                "passed": passed,
                "steps_used": steps_used,
                "max_steps": int(task["max_steps"]),
                "time_elapsed": round(elapsed, 3),
                "fallback_triggered": fallback_triggered,
                "verification_result": asdict(verification),
                "stopped_consecutive_failures": stopped_consecutive_failures,
            }
            results.append(result)

        self._save_results(results)
        self._print_summary(results)
        return results

    @staticmethod
    def _save_results(results: list[dict[str, Any]]) -> None:
        out_dir = Path("eval")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"results_{timestamp}.json"
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"[BENCHMARK] Saved results to {out_path}")

    @staticmethod
    def _print_summary(results: list[dict[str, Any]]) -> None:
        print("task_id | goal | result | steps | time | fallback")
        for item in results:
            outcome = "PASS" if item["passed"] else "FAIL"
            print(
                f"{item['task_id']} | {item['goal']} | {outcome} | "
                f"{item['steps_used']} | {item['time_elapsed']:.2f}s | {item['fallback_triggered']}"
            )


def _select_tasks(tasks: str | None, task_id: str | None) -> list[dict[str, Any]]:
    if task_id:
        for task in BENCHMARK_TASKS:
            if task["id"] == task_id:
                return [task]
        raise ValueError(f"Unknown task id: {task_id}")

    if tasks is None or tasks == "all":
        return list(BENCHMARK_TASKS)

    wanted = tasks.strip().lower()
    filtered = [task for task in BENCHMARK_TASKS if task["difficulty"].lower() == wanted]
    if not filtered:
        raise ValueError(f"No benchmark tasks found for group: {tasks}")
    return filtered


def run_quick_benchmark_suite() -> list[dict[str, Any]]:
    easy = [task for task in BENCHMARK_TASKS if task["difficulty"] == "easy"]
    runner = BenchmarkRunner(tasks=easy[:5])
    return runner.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Kai benchmark harness")
    parser.add_argument("--tasks", default="all", help="all | easy | medium | hard")
    parser.add_argument("--task", default=None, help="Run a single task by task id")
    args = parser.parse_args()

    selected_tasks = _select_tasks(tasks=args.tasks, task_id=args.task)
    runner = BenchmarkRunner(tasks=selected_tasks)
    runner.run()


if __name__ == "__main__":
    main()
