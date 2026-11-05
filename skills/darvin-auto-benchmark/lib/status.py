"""FIX 4 -- Precise task-status parsing (do not confuse command/task/job states).

The original pipeline did a blind recursive walk of the whole `--json` report for
any `name=="status"` and took the first. That is wrong for `solution status`
because a single report contains three unrelated "status" notions:

  1. top-level report["status"]         -> COMMAND result (success/warning/failed)
  2. section "Task"  field "status"     -> the MAPPED TASK STATE (what we want)
  3. table  "Jobs"   column "Status"    -> per-JOB state
  (+ section "Solution" field "status"  -> model-only mode, no training task)

A command can be status="success" while the task it reports is status="failed".
This parser targets the exact (section, field), so the terminal-state decision is
driven only by the "Task" section's "status" field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import _common, report


@dataclass
class TaskStatus:
    command_ok: bool = False
    command_state: str = ""
    error_code: Optional[str] = None
    # task identity / state (from section "Task")
    has_task: bool = False
    task_state: Optional[str] = None          # mapped TaskState
    raw_status: Optional[str] = None
    task_id: Optional[str] = None
    model_app_id: Optional[str] = None
    subnet_id: Optional[str] = None
    # model-only mode (section "Solution", no training task)
    model_only: bool = False
    solution_status: Optional[str] = None
    score: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.task_state in _common.TERMINAL_TASK_STATES

    @property
    def outcome(self) -> str:
        """Human-meaningful terminal classification (or a non-terminal label)."""
        if not self.has_task:
            return "model_only" if self.model_only else "no_task"
        if self.task_state == "success":
            return "evaluation_complete"
        if self.task_state in _common.TERMINAL_TASK_STATES:
            return f"ended_not_success:{self.task_state}"
        return f"non_terminal:{self.task_state}"


def _clean(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip()
    return None if _common.is_empty(v) else v


def parse_status_report(report_json: Dict[str, Any]) -> TaskStatus:
    """Parse a `darvin solution status <task_id> --json` (or `task status`) report.

    The terminal-state decision is driven ONLY by the "Task" section's "status"
    field, never by the top-level command state.
    """
    info = TaskStatus()
    info.command_state = report.command_state(report_json)
    info.command_ok = report.command_ok(report_json)
    info.error_code = report.error_code(report_json)
    info.warnings = [str(w) for w in (report_json.get("warnings") or [])]

    task_fields = report.section_fields(report_json, _common.SECTION_TASK)
    sol_fields = report.section_fields(report_json, _common.SECTION_SOLUTION)

    if task_fields:
        info.task_state = _clean(task_fields.get("status"))
        info.raw_status = _clean(task_fields.get("raw_status"))
        info.task_id = _clean(task_fields.get("task_id"))
        info.model_app_id = _clean(task_fields.get("model_app_id"))
        info.subnet_id = _clean(task_fields.get("subnet_id"))
        info.has_task = info.task_id is not None

    if sol_fields:
        info.model_only = True
        info.solution_status = _clean(sol_fields.get("status"))
        info.model_app_id = info.model_app_id or _clean(sol_fields.get("model_app_id"))
        info.score = _clean(sol_fields.get("score")) or _clean(sol_fields.get("score_raw"))

    # In task mode the score can also appear in the Jobs table.
    if info.score is None:
        info.score = _score_from_jobs(report_json)

    return info


def _score_from_jobs(report_json: Dict[str, Any]) -> Optional[str]:
    rows = report.table_rows_as_dicts(report_json, "Jobs")
    for col in ("Score (raw)", "Score"):
        for row in rows:
            if col in row and not _common.is_empty(row[col]):
                return str(row[col])
    return None


def is_terminal(state: Optional[str]) -> bool:
    return state is not None and state in _common.TERMINAL_TASK_STATES


def should_download_logs(info: TaskStatus) -> bool:
    """Terminal non-success tasks warrant a log pull as evidence."""
    return info.terminal and info.task_state != "success"
