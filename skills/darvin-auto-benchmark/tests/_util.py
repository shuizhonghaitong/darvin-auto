"""Shared helpers for the offline test suite. No network, no docker, no darvin."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")


def load(name: str) -> Dict[str, Any]:
    with open(os.path.join(FIXDIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def solution_rows(name: str) -> List[Dict[str, str]]:
    from lib import report, _common

    return report.table_rows_as_dicts(load(name), _common.TABLE_MY_SOLUTIONS)
