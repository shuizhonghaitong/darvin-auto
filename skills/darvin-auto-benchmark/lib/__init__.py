"""General Darvin auto-benchmark decision core.

Pure, offline-testable, leaderboard-agnostic. See the module docstrings for the
five recorded defects fixed here:

  registry.py  FIX 1  tag-absent vs registry-query-error
  dedup.py     FIX 2+5 submission dedup after interruption + task identity
  contract.py  FIX 3  Feishu inference-contract verification
  status.py    FIX 4  precise status parsing + terminal detection
  state.py     FIX (recovery) atomic/locked state + non-idempotent reconciliation
  pipeline.py  the planner that composes the above (disabled-by-default writes)
"""

from . import _common, contract, dedup, pipeline, registry, report, state, status
from .pipeline import Evidence, GatePlan, Pipeline, PipelineConfig
from .registry import RegistryResult, classify_lookup
from .status import TaskStatus, parse_status_report
from .dedup import SubmitPlan, decide_submit
from .contract import ContractPlan, build_anchors, verify_contract
from .state import PipelineState, ReconcileResult, reconcile_non_idempotent_gate

__all__ = [
    "_common", "contract", "dedup", "pipeline", "registry", "report", "state", "status",
    "Evidence", "GatePlan", "Pipeline", "PipelineConfig",
    "RegistryResult", "classify_lookup",
    "TaskStatus", "parse_status_report",
    "SubmitPlan", "decide_submit",
    "ContractPlan", "build_anchors", "verify_contract",
    "PipelineState", "ReconcileResult", "reconcile_non_idempotent_gate",
]
