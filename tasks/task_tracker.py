"""
Task Tracker — lightweight project task monitoring.

Usage:
    # As a module
    from tasks import TaskTracker
    tracker = TaskTracker()
    tracker.dashboard()

    # From command line
    python -m tasks.task_tracker              # show dashboard
    python -m tasks.task_tracker start <id>   # mark task running
    python -m tasks.task_tracker done <id>    # mark task completed
    python -m tasks.task_tracker fail <id>    # mark task failed
    python -m tasks.task_tracker reset <id>   # reset to pending
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List


TASKS_FILE = Path(__file__).parent / "project_tasks.yaml"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


# Category colors for dashboard display
CATEGORY_LABELS = {
    "pipeline_a": "Pipeline A (Explain-All)",
    "pipeline_b": "Pipeline B (Budgeted)",
    "evaluation": "Evaluation & Analysis",
    "paper": "Paper & Write-up",
}

STATUS_SYMBOLS = {
    TaskStatus.PENDING: "○",
    TaskStatus.RUNNING: "▶",
    TaskStatus.COMPLETED: "●",
    TaskStatus.FAILED: "✗",
    TaskStatus.BLOCKED: "◌",
}


@dataclass
class Task:
    """A single trackable task."""
    id: str
    name: str
    status: str = "pending"
    category: str = "pipeline_a"
    dataset: Optional[str] = None
    depends_on: List[str] = field(default_factory=list)
    created: Optional[str] = None
    started: Optional[str] = None
    completed: Optional[str] = None
    metrics_file: Optional[str] = None
    metrics_summary: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None
    error: Optional[str] = None

    @property
    def status_enum(self) -> TaskStatus:
        return TaskStatus(self.status)

    def to_dict(self) -> Dict:
        d = {}
        d["id"] = self.id
        d["name"] = self.name
        d["status"] = self.status
        d["category"] = self.category
        if self.dataset:
            d["dataset"] = self.dataset
        if self.depends_on:
            d["depends_on"] = self.depends_on
        d["created"] = self.created
        if self.started:
            d["started"] = self.started
        if self.completed:
            d["completed"] = self.completed
        if self.metrics_file:
            d["metrics_file"] = self.metrics_file
        if self.metrics_summary:
            d["metrics_summary"] = self.metrics_summary
        if self.notes:
            d["notes"] = self.notes
        if self.error:
            d["error"] = self.error
        return d


class TaskTracker:
    """
    Load, update, and display project tasks from a YAML file.

    Tasks live in tasks/project_tasks.yaml — human-readable
    and version-controlled so the team always sees the state.
    """

    def __init__(self, tasks_file: Optional[str] = None):
        self.tasks_file = Path(tasks_file) if tasks_file else TASKS_FILE
        self.tasks: Dict[str, Task] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────

    def _load(self) -> None:
        """Load tasks from YAML."""
        if not self.tasks_file.exists():
            self.tasks = {}
            return
        with open(self.tasks_file, "r") as f:
            data = yaml.safe_load(f) or {}
        for t in data.get("tasks", []):
            task = Task(**t)
            self.tasks[task.id] = task

    def _save(self) -> None:
        """Write tasks back to YAML."""
        data = {
            "# Task Monitor": "Auto-updated by TaskTracker — also human-editable",
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "tasks": [t.to_dict() for t in self.tasks.values()],
        }
        with open(self.tasks_file, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # ── mutations ────────────────────────────────────────────

    def start(self, task_id: str) -> Task:
        """Mark a task as running."""
        task = self._get(task_id)
        task.status = TaskStatus.RUNNING.value
        task.started = datetime.now().strftime("%Y-%m-%d %H:%M")
        task.error = None
        self._save()
        print(f"▶ Started: {task.name}")
        return task

    def complete(
        self,
        task_id: str,
        metrics_file: Optional[str] = None,
        metrics_summary: Optional[Dict] = None,
    ) -> Task:
        """Mark a task as completed, optionally recording metrics."""
        task = self._get(task_id)
        task.status = TaskStatus.COMPLETED.value
        task.completed = datetime.now().strftime("%Y-%m-%d %H:%M")
        if metrics_file:
            task.metrics_file = metrics_file
        if metrics_summary:
            task.metrics_summary = metrics_summary
        task.error = None
        self._save()
        print(f"● Completed: {task.name}")
        return task

    def fail(self, task_id: str, error: Optional[str] = None) -> Task:
        """Mark a task as failed."""
        task = self._get(task_id)
        task.status = TaskStatus.FAILED.value
        task.completed = datetime.now().strftime("%Y-%m-%d %H:%M")
        task.error = error
        self._save()
        print(f"✗ Failed: {task.name}" + (f" — {error}" if error else ""))
        return task

    def reset(self, task_id: str) -> Task:
        """Reset a task to pending."""
        task = self._get(task_id)
        task.status = TaskStatus.PENDING.value
        task.started = None
        task.completed = None
        task.metrics_file = None
        task.metrics_summary = None
        task.error = None
        self._save()
        print(f"○ Reset: {task.name}")
        return task

    def add_task(
        self,
        task_id: str,
        name: str,
        category: str = "pipeline_a",
        dataset: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Task:
        """Add a new task."""
        if task_id in self.tasks:
            raise ValueError(f"Task '{task_id}' already exists")
        task = Task(
            id=task_id,
            name=name,
            category=category,
            dataset=dataset,
            depends_on=depends_on or [],
            created=datetime.now().strftime("%Y-%m-%d"),
            notes=notes,
        )
        self.tasks[task_id] = task
        self._save()
        return task

    # ── queries ──────────────────────────────────────────────

    def get(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def _get(self, task_id: str) -> Task:
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task: '{task_id}'")
        return task

    def by_status(self, status: TaskStatus) -> List[Task]:
        return [t for t in self.tasks.values() if t.status == status.value]

    def by_category(self, category: str) -> List[Task]:
        return [t for t in self.tasks.values() if t.category == category]

    def is_blocked(self, task_id: str) -> bool:
        """True if any dependency is not completed."""
        task = self._get(task_id)
        for dep_id in task.depends_on:
            dep = self.tasks.get(dep_id)
            if dep is None or dep.status != TaskStatus.COMPLETED.value:
                return True
        return False

    # ── dashboard ────────────────────────────────────────────

    def dashboard(self) -> None:
        """Print a nicely formatted task dashboard."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        total = len(self.tasks)
        done = len(self.by_status(TaskStatus.COMPLETED))
        running = len(self.by_status(TaskStatus.RUNNING))
        failed = len(self.by_status(TaskStatus.FAILED))
        pending = total - done - running - failed

        print()
        print("=" * 68)
        print("  PROJECT TASK MONITOR")
        print(f"  {now}    {done}/{total} completed", end="")
        if running:
            print(f"    {running} running", end="")
        if failed:
            print(f"    {failed} failed", end="")
        print()
        print("=" * 68)

        # Progress bar
        if total > 0:
            bar_len = 50
            filled = int(bar_len * done / total)
            bar = "█" * filled + "░" * (bar_len - filled)
            pct = done / total * 100
            print(f"  [{bar}] {pct:.0f}%")
        print()

        # Group by category
        categories_order = ["pipeline_a", "pipeline_b", "evaluation", "paper"]
        seen_cats = set()
        for t in self.tasks.values():
            seen_cats.add(t.category)
        # Handle categories not in the predefined order
        for cat in seen_cats:
            if cat not in categories_order:
                categories_order.append(cat)

        for cat in categories_order:
            cat_tasks = self.by_category(cat)
            if not cat_tasks:
                continue
            cat_label = CATEGORY_LABELS.get(cat, cat)
            cat_done = sum(1 for t in cat_tasks if t.status == TaskStatus.COMPLETED.value)
            print(f"  ── {cat_label} ({cat_done}/{len(cat_tasks)}) ──")
            for t in cat_tasks:
                sym = STATUS_SYMBOLS.get(TaskStatus(t.status), "?")
                line = f"    {sym} {t.name}"
                if t.dataset:
                    line += f"  [{t.dataset}]"
                if t.status == TaskStatus.COMPLETED.value and t.metrics_summary:
                    # Show key metric inline
                    if "pass_rate" in t.metrics_summary:
                        line += f"  pass={t.metrics_summary['pass_rate']}"
                    if "total_explained" in t.metrics_summary:
                        line += f"  n={t.metrics_summary['total_explained']}"
                if t.status == TaskStatus.RUNNING.value and t.started:
                    line += f"  (started {t.started})"
                if t.status == TaskStatus.FAILED.value and t.error:
                    line += f"  ✗ {t.error[:40]}"
                if self.is_blocked(t.id) and t.status == TaskStatus.PENDING.value:
                    line += "  ⊘ blocked"
                print(line)
            print()

        print("=" * 68)

    # ── convenience for pipeline integration ─────────────────

    def pipeline_start(self, task_id: str) -> "TaskTracker":
        """Call at the beginning of a pipeline run."""
        self.start(task_id)
        return self

    def pipeline_complete(
        self,
        task_id: str,
        metrics: Dict[str, Any],
        metrics_file: str,
    ) -> "TaskTracker":
        """Call at the end of a successful pipeline run."""
        summary = {
            "total_explained": metrics.get("counts", {}).get("successful_explanations"),
            "pass_rate": f"{metrics.get('verification', {}).get('pass_rate', 0):.1%}",
            "avg_tokens": int(metrics.get("tokens", {}).get("avg_per_explanation", 0)),
            "total_time_s": int(metrics.get("total_time_seconds", 0)),
        }
        self.complete(task_id, metrics_file=metrics_file, metrics_summary=summary)
        return self

    def pipeline_fail(self, task_id: str, error: str) -> "TaskTracker":
        """Call when a pipeline run fails."""
        self.fail(task_id, error=error)
        return self


# ── CLI entry-point ──────────────────────────────────────────

def main():
    import sys
    tracker = TaskTracker()

    args = sys.argv[1:]
    if not args:
        tracker.dashboard()
        return

    cmd = args[0]
    if cmd == "dashboard":
        tracker.dashboard()
    elif cmd == "start" and len(args) >= 2:
        tracker.start(args[1])
        tracker.dashboard()
    elif cmd == "done" and len(args) >= 2:
        tracker.complete(args[1])
        tracker.dashboard()
    elif cmd == "fail" and len(args) >= 2:
        error = args[2] if len(args) >= 3 else None
        tracker.fail(args[1], error=error)
        tracker.dashboard()
    elif cmd == "reset" and len(args) >= 2:
        tracker.reset(args[1])
        tracker.dashboard()
    elif cmd == "list":
        for t in tracker.tasks.values():
            s = STATUS_SYMBOLS.get(TaskStatus(t.status), "?")
            print(f"  {s} {t.id:30s} {t.status:10s} {t.name}")
    else:
        print("Usage: python -m tasks.task_tracker [dashboard|start|done|fail|reset|list] [task_id]")


if __name__ == "__main__":
    main()
