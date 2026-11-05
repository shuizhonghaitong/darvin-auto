"""Helpers to read darvin-cli `--json` command reports by STRUCTURE, not by
blindly walking keys.

A report is a serialized `CommandReport` (darvin_cli/models/report.py):

    {
      "command": str, "status": <ReportStatus>, "error_code": str|None,
      "steps":  [{"name","status","detail"}],
      "sections":[{"title","fields":[{"name","value"}]}],
      "tables":  [{"title","columns":[...],"rows":[[...]]}],
      "artifacts":[{"name","value","path"}],
      "warnings":[...], "errors":[...], "data": {...}
    }

The CLI distinguishes three different kinds of "status":
  * the TOP-LEVEL report "status"  -> the *command* result (success/warning/failed)
  * a section field named "status" -> a *domain* value (e.g. mapped task state)
  * a table column named "Status"  -> per-row values (e.g. per-job status)

Correct parsing therefore targets a specific (section, field) or (table, column).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import _common


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def sections(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return report.get("sections") or []


def tables(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return report.get("tables") or []


def artifacts(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return report.get("artifacts") or []


def section_by_title(report: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
    for sec in sections(report):
        if sec.get("title") == title:
            return sec
    return None


def table_by_title(report: Dict[str, Any], title: str) -> Optional[Dict[str, Any]]:
    for tab in tables(report):
        if tab.get("title") == title:
            return tab
    return None


def section_fields(report: Dict[str, Any], title: str) -> Dict[str, str]:
    """Return {field_name: field_value} for a named section ({} if absent)."""
    sec = section_by_title(report, title)
    if not sec:
        return {}
    out: Dict[str, str] = {}
    for f in sec.get("fields") or []:
        name = f.get("name")
        if name is None:
            continue
        out[str(name)] = "" if f.get("value") is None else str(f.get("value"))
    return out


def table_rows_as_dicts(report: Dict[str, Any], title: str) -> List[Dict[str, str]]:
    """Return the rows of a named table as a list of {column: value} dicts."""
    tab = table_by_title(report, title)
    if not tab:
        return []
    cols = [str(c) for c in (tab.get("columns") or [])]
    out: List[Dict[str, str]] = []
    for row in tab.get("rows") or []:
        if not isinstance(row, list):
            continue
        rec: Dict[str, str] = {}
        for i, col in enumerate(cols):
            rec[col] = "" if i >= len(row) or row[i] is None else str(row[i])
        out.append(rec)
    return out


def artifact_value(report: Dict[str, Any], name: str) -> Optional[str]:
    for a in artifacts(report):
        if a.get("name") == name:
            if a.get("value") not in (None, ""):
                return str(a.get("value"))
            if a.get("path") not in (None, ""):
                return str(a.get("path"))
    return None


def command_state(report: Dict[str, Any]) -> str:
    """The COMMAND result state (ReportStatus), independent of any domain state."""
    return str(report.get("status") or "").strip().lower()


def command_ok(report: Dict[str, Any]) -> bool:
    return command_state(report) in _common.COMMAND_OK_STATES


def error_code(report: Dict[str, Any]) -> Optional[str]:
    code = report.get("error_code")
    return str(code) if code else None
