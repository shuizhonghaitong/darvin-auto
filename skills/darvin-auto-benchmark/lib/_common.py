"""Shared enums and constants for the general Darvin auto-benchmark pipeline.

These values are pinned to the installed darvin-cli 0.5.2 source (offline ground
truth). If the CLI changes, update this module and its tests together:

- TaskState terminal set            -> darvin_cli/models/common.py (TaskState)
- raw->mapped status table          -> darvin_cli/services/status_mapping_service.py
- solution status report sections   -> darvin_cli/workflows/solution_status.py
- solution list columns             -> darvin_cli/workflows/solution_list.py
- subnet list columns               -> darvin_cli/workflows/subnet_discovery.py
- submit receipt fields / PartialSuccessError -> darvin_cli/workflows/solution_submit.py,
                                                 darvin_cli/services/solution_service.py
- error codes                       -> darvin_cli/utils/errors.py
"""

from __future__ import annotations

from enum import Enum
from typing import Final

# --- canonical task states (darvin_cli TaskState) ---------------------------
# Terminal = the evaluation has stopped progressing; no further polling helps.
TERMINAL_TASK_STATES: Final[frozenset] = frozenset(
    {"success", "failed", "killed", "cancelled", "mixed"}
)
NON_TERMINAL_TASK_STATES: Final[frozenset] = frozenset(
    {"waiting_for_jobs", "running", "paused", "not_killable", "unknown"}
)

# --- CLI command report states (darvin_cli ReportStatus) ---------------------
# NOTE: the TOP-LEVEL report "status" is the *command* state (did the CLI command
# succeed), NOT the task state. A command can report status="success" while the
# underlying task is status="failed". Never conflate the two.
COMMAND_OK_STATES: Final[frozenset] = frozenset({"success", "warning"})

# --- report section / table titles (must match CLI exactly) ------------------
SECTION_TASK: Final = "Task"
SECTION_SOLUTION: Final = "Solution"
SECTION_SUBMISSION: Final = "Submission"
SECTION_SUBNET: Final = "Subnet"
TABLE_SUBNETS: Final = "Subnets"
TABLE_MY_SOLUTIONS: Final = "My solutions"

# --- "My solutions" columns (darvin_cli solution_list.py) -------------------
SOLUTIONS_COL_MODEL_ID: Final = "Model ID"
SOLUTIONS_COL_SUBMISSION_ID: Final = "Submission ID"
SOLUTIONS_COL_NAME: Final = "Name"
SOLUTIONS_COL_SUBNET: Final = "Subnet"
SOLUTIONS_COL_STATUS: Final = "Status"
SOLUTIONS_COL_EVALUATION: Final = "Evaluation"
SOLUTIONS_COL_SCORE: Final = "Score"
SOLUTIONS_COL_TASK_ID: Final = "Task ID"
SOLUTIONS_COL_LOCAL_RECEIPT: Final = "Local receipt"

# --- subnet list columns (darvin_cli subnet_discovery.py) -------------------
SUBNET_COL_BACKEND_ID: Final = "Backend ID"
SUBNET_COL_SUBNET_ID: Final = "Subnet ID"
SUBNET_COL_NAME: Final = "Name"
SUBNET_COL_CONTRACT: Final = "Contract"
SUBNET_COL_STAGE: Final = "Stage"
SUBNET_COL_STATUS: Final = "Status"
SUBNET_COL_SUBMISSION_TYPE: Final = "Submission type"

# --- sentinel used by the CLI to mean "no value" ----------------------------
EMPTY_TOKEN: Final = "-"


def is_empty(value: object) -> bool:
    """True for the CLI's 'no value' sentinel or blank strings."""
    if value is None:
        return True
    return str(value).strip() in ("", EMPTY_TOKEN)


class RegistryVerdict(str, Enum):
    """Outcome of asking a registry whether an image tag already exists.

    Only ABSENT authorizes a first-time create/push. ERROR means the lookup
    itself failed (auth/network/5xx/malformed) and must NOT be treated as
    absent -- that is the exact class of bug that caused tag overwrites.
    """

    PRESENT = "present"
    ABSENT = "absent"
    ERROR = "error"


class SubmitDecision(str, Enum):
    """Safe action to take before (or instead of) a leaderboard submission."""

    REUSE_EXISTING = "reuse_existing"     # a task already exists -> track it, do not submit
    SUBMIT_FRESH = "submit_fresh"         # no prior evidence -> submit (with the stable attempt)
    ABORT_IDENTITY = "abort_identity"     # conflicting identity -> human reconciliation required
    ABORT_UNCERTAIN = "abort_uncertain"   # prior result unknown + cannot confirm -> reconcile first


class ContractResult(str, Enum):
    """Outcome of verifying a Feishu inference contract against expectations."""

    MATCH = "match"            # every expected anchor present, no conflicts -> verified
    MISMATCH = "mismatch"      # an expected anchor conflicts with the doc -> block + report diffs
    INCOMPLETE = "incomplete"  # doc missing/unfetched/key anchors absent -> NOT verified; block


class GateVerdict(str, Enum):
    """Effective state of a gate after merging local markers with external evidence."""

    SKIP_DONE = "skip_done"            # externally confirmed done -> skip
    RECONCILE = "reconcile"            # external evidence available -> act on it (no blind redo)
    PROCEED = "proceed"                # externally confirmed safe to run this gate
    QUERY = "query"                    # no authoritative evidence yet -> obtain it first
    HALT = "halt"                      # external state unknown/conflicting -> stop, report
