from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, List


@dataclass(frozen=True)
class TaskRecord:
    goal: str
    actions: list[str]
    success: bool
    app_context: str
    timestamp: float


class SessionMemory:
    def __init__(self, capacity: int = 20) -> None:
        self._records: Deque[TaskRecord] = deque(maxlen=max(1, capacity))
        self._lock = threading.Lock()

    @staticmethod
    def _persist_enabled() -> bool:
        return os.getenv("KAI_PERSIST_MEMORY", "0").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        normalized = " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))
        return {token for token in normalized.split() if len(token) >= 2}

    def _persist_success_record(self, record: TaskRecord) -> None:
        if not self._persist_enabled() or not record.success:
            return

        memory_dir = Path("memory")
        memory_dir.mkdir(parents=True, exist_ok=True)
        date_tag = datetime.now().strftime("%Y-%m-%d")
        out_path = memory_dir / f"session_{date_tag}.json"

        existing: list[dict] = []
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            except Exception:
                existing = []

        existing.append(asdict(record))
        out_path.write_text(json.dumps(existing, indent=2, ensure_ascii=True), encoding="utf-8")

    def add_task(self, goal: str, actions: list[str], success: bool, app_context: str) -> None:
        record = TaskRecord(
            goal=(goal or "").strip(),
            actions=[str(action) for action in (actions or [])],
            success=bool(success),
            app_context=(app_context or "unknown").strip().lower() or "unknown",
            timestamp=time.time(),
        )
        with self._lock:
            self._records.append(record)
        self._persist_success_record(record)

    def get_similar_tasks(self, goal: str, top_k: int = 3) -> List[TaskRecord]:
        target_tokens = self._tokenize(goal)
        if not target_tokens:
            return []

        with self._lock:
            records = list(self._records)

        scored: list[tuple[float, TaskRecord]] = []
        for record in records:
            source_tokens = self._tokenize(record.goal)
            if not source_tokens:
                continue
            union = target_tokens | source_tokens
            if not union:
                continue
            overlap = target_tokens & source_tokens
            score = len(overlap) / len(union)
            if score >= 0.3:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[: max(1, top_k)]]

    def get_success_rate(self, app_context: str | None = None) -> float:
        with self._lock:
            records = list(self._records)

        if app_context:
            wanted = app_context.strip().lower()
            records = [record for record in records if record.app_context == wanted]

        if not records:
            return 0.0

        success_count = sum(1 for record in records if record.success)
        return success_count / len(records)

    def get_summary(self) -> str:
        with self._lock:
            records = list(self._records)

        total = len(records)
        if total == 0:
            return "[MEMORY] No session tasks recorded yet."

        success_rate = self.get_success_rate()
        by_app: dict[str, int] = {}
        for record in records:
            by_app[record.app_context] = by_app.get(record.app_context, 0) + 1

        app_parts = [f"{app}:{count}" for app, count in sorted(by_app.items())]
        return (
            f"[MEMORY] Tasks={total} | SuccessRate={success_rate:.2f} | "
            f"Apps={', '.join(app_parts)}"
        )


_SESSION_MEMORY = SessionMemory(capacity=20)


def get_session_memory() -> SessionMemory:
    return _SESSION_MEMORY
