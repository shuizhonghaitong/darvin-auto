"""FIX 2 + FIX 5 -- Submission de-duplication and true task identity.

Recorded defects this fixes:
  * "uncertain submission outcomes leading to duplicate submissions"
  * "a solution name alone cannot establish submission identity"

Ground truth (darvin-cli 0.5.2):
  * A submission's STABLE identity is its `trace_id` (= the payload field
    `model_solution_hash_id`), a deterministic fingerprint of
    (base_url, wallet, subnet_address, name, submission_type, task_name,
     auto_create_task, authors, training_image, content_digest, attempt).
    See models/solution.py: SolutionConfig.build_trace_id.
  * The receipt store is keyed by trace_id, but a model-only receipt can lead
    the CLI to call create_model again. Backend deduplication is not verified.
  * An UNCERTAIN submit (5xx / exception) leaves receipt.stage == "unknown" and
    raises PartialSuccessError("retry with the same input and attempt").
    Changing `attempt` changes trace_id and therefore creates a DUPLICATE.
  * The `solution submit --json` report has section "Submission" with fields
    stage/model_saved/task_created/job_created/model_app_id/submission_id/
    task_id/trace_id/historical, plus artifact "receipt_path".
  * `solution list --subnet A --json` table "My solutions" columns:
    Model ID, Submission ID, Name, Subnet, Status, Evaluation, Score, Raw score,
    Task ID, Local receipt.

Rules encoded below:
  R1  A confirmed task_id (local receipt or submit report) => REUSE_EXISTING.
  R2  historical submit (CLI returned a prior receipt)      => REUSE/ABORT,
      never a fresh submit.
  R3  local receipt with a model but no task (model_saved / uploaded)
      => ABORT_UNCERTAIN pending authoritative reconciliation.
  R4  local receipt stage == "unknown" (interrupted, result unknown)
      => reconcile; never automatically retry an uncertain write.
  R5  Attempt-change guard: any prior non-fresh receipt whose attempt differs
      from the intended attempt => ABORT_IDENTITY (would create a duplicate).
  R6  Fresh (no local/submit evidence): a remote row matching ONLY by name on
      our subnet => ABORT_IDENTITY (name is not identity; reconcile manually).
  R7  An empty list alone cannot prove a fresh submission is safe.
  R8  Strong-id remote matches: one distinct task_id => REUSE_EXISTING;
      several distinct task_ids => ABORT_IDENTITY (conflict).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import _common, report

# receipt stages that mean "something was already written to the backend"
_NON_FRESH_STAGES = {"uploaded", "model_saved", "task_created", "job_created", "unknown"}


@dataclass
class SubmitPlan:
    decision: _common.SubmitDecision
    reason: str
    track_task_id: Optional[str] = None
    rerun_same_attempt: bool = False
    matched_rows: List[Dict[str, str]] = field(default_factory=list)


def _clean(val: Optional[str]) -> Optional[str]:
    if val is None:
        return None
    v = str(val).strip()
    return None if _common.is_empty(v) else v


def identity_from_receipt(receipt: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    if not receipt:
        return {}
    return {
        "model_app_id": _clean(receipt.get("model_app_id")),
        "submission_id": _clean(receipt.get("submission_id")),
        "task_id": _clean(receipt.get("task_id")),
        "trace_id": _clean(receipt.get("trace_id")),
        "subnet_address": _clean(receipt.get("subnet_address")),
        "attempt": _clean(receipt.get("attempt")),
        "stage": _clean(receipt.get("stage")),
        "historical": None,
    }


def identity_from_submit_report(rep: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    if not rep:
        return {}
    f = report.section_fields(rep, _common.SECTION_SUBMISSION)
    out = {
        "model_app_id": _clean(f.get("model_app_id")),
        "submission_id": _clean(f.get("submission_id")),
        "task_id": _clean(f.get("task_id")),
        "trace_id": _clean(f.get("trace_id")),
        "subnet_address": None,
        "attempt": None,
        "stage": _clean(f.get("stage")),
        "historical": _clean(f.get("historical")),
        "task_created": _clean(f.get("task_created")),
    }
    # A PARTIAL_SUCCESS (uncertain) submit has no Submission section; it carries
    # error_code=PARTIAL_SUCCESS and artifacts trace_id/receipt_path.
    if not f and str(rep.get("error_code") or "") == "PARTIAL_SUCCESS":
        out["stage"] = "unknown"
        out["trace_id"] = report.artifact_value(rep, "trace_id")
    return out


def parse_remote_rows(rep: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not rep:
        return []
    return report.table_rows_as_dicts(rep, _common.TABLE_MY_SOLUTIONS)


def _row_task_id(row: Dict[str, str]) -> Optional[str]:
    return _clean(row.get(_common.SOLUTIONS_COL_TASK_ID))


def _row_model_id(row: Dict[str, str]) -> Optional[str]:
    return _clean(row.get(_common.SOLUTIONS_COL_MODEL_ID))


def _row_submission_id(row: Dict[str, str]) -> Optional[str]:
    return _clean(row.get(_common.SOLUTIONS_COL_SUBMISSION_ID))


def _row_name(row: Dict[str, str]) -> Optional[str]:
    return _clean(row.get(_common.SOLUTIONS_COL_NAME))


def _row_subnet(row: Dict[str, str]) -> Optional[str]:
    return _clean(row.get(_common.SOLUTIONS_COL_SUBNET))


def _strong_match(row: Dict[str, str], our: Dict[str, Optional[str]]) -> bool:
    """True only when the row shares a STRONG identity anchor with us."""
    if our.get("model_app_id") and _row_model_id(row) == str(our["model_app_id"]):
        return True
    if our.get("task_id") and _row_task_id(row) == our["task_id"]:
        return True
    if our.get("submission_id") and _row_submission_id(row) == str(our["submission_id"]):
        return True
    return False


def _name_match(row: Dict[str, str], our: Dict[str, Optional[str]]) -> bool:
    """Match by NAME only (a WEAK signal).

    The caller lists solutions for ONE subnet (`solution list --subnet <addr>`),
    so the returned rows are already subnet-scoped. We only reject when the row's
    subnet column and ours both look like 0x addresses and clearly differ.
    """
    name = our.get("name")
    if not name or _row_name(row) != name:
        return False
    row_subnet = _row_subnet(row)
    our_subnet = our.get("subnet_address")
    if (
        row_subnet
        and our_subnet
        and row_subnet.lower().startswith("0x")
        and our_subnet.lower().startswith("0x")
        and row_subnet.lower() != our_subnet.lower()
    ):
        return False
    return True


def decide_submit(
    local_receipt: Optional[Dict[str, Any]],
    submit_report: Optional[Dict[str, Any]] = None,
    remote_rows: Optional[List[Dict[str, str]]] = None,
    *,
    expected_name: Optional[str] = None,
    expected_subnet: Optional[str] = None,
    expected_attempt: Optional[str] = None,
) -> SubmitPlan:
    """Choose the safe action for a submission, given all known evidence.

    `remote_rows` is the parsed "My solutions" table for our subnet.
    `expected_*` describe what we are trying to (re)submit.

    Priority (fail closed):
      R1 confirmed task id            -> REUSE_EXISTING (track, never submit)
      R5 attempt changed on prior work -> ABORT_IDENTITY (would duplicate)
      R2 historical, no model          -> ABORT_IDENTITY (reconcile)
      RM we know a model, no confirmed  -> REUSE (1 task) / ABORT otherwise
      RF no anchors at all               -> ABORT_IDENTITY (name collision) / ABORT_UNCERTAIN
    """
    rows = list(remote_rows or [])
    our = identity_from_receipt(local_receipt)
    sub = identity_from_submit_report(submit_report)
    for k, v in sub.items():
        if our.get(k) in (None, "") and v not in (None, ""):
            our[k] = v
    our["name"] = our.get("name") or expected_name
    our["subnet_address"] = our.get("subnet_address") or expected_subnet

    prior_stage = (local_receipt or {}).get("stage") or sub.get("stage") or ""
    prior_attempt = (local_receipt or {}).get("attempt")  # only reliably known from the receipt

    # R1 -- a confirmed task id means we already submitted: track, never re-submit.
    confirmed_task = _clean(our.get("task_id"))
    if confirmed_task:
        matched = [r for r in rows if _row_task_id(r) == confirmed_task]
        return SubmitPlan(
            _common.SubmitDecision.REUSE_EXISTING,
            f"task {confirmed_task} already exists; track it, do not submit again.",
            track_task_id=confirmed_task, matched_rows=matched,
        )

    # R5 -- changing the attempt on a non-fresh submission would create a duplicate.
    if (
        prior_attempt
        and expected_attempt
        and prior_attempt != expected_attempt
        and prior_stage in _NON_FRESH_STAGES
    ):
        return SubmitPlan(
            _common.SubmitDecision.ABORT_IDENTITY,
            f"attempt changed from '{prior_attempt}' to '{expected_attempt}' on a "
            f"non-fresh submission (stage={prior_stage}); that would create a duplicate. "
            f"Re-run with the SAME attempt or reconcile first.",
        )

    # R2 -- CLI returned a historical receipt but we cannot resolve a model to backfill.
    if sub.get("historical") == "true" and not _clean(our.get("model_app_id")):
        return SubmitPlan(
            _common.SubmitDecision.ABORT_IDENTITY,
            "submit returned a historical receipt without a resolvable model id; "
            "reconcile (solution list / status --trace) before acting.",
        )

    # RM -- we know a model/submission/trace anchor but have no confirmed task yet.
    model_anchor = bool(_clean(our.get("model_app_id")) or _clean(our.get("submission_id"))
                        or _clean(our.get("trace_id")))
    if model_anchor:
        strong = [r for r in rows if _strong_match(r, our)]
        task_ids = {t for t in (_row_task_id(r) for r in strong) if t}
        if len(task_ids) == 1:
            tid = next(iter(task_ids))
            return SubmitPlan(
                _common.SubmitDecision.REUSE_EXISTING,
                f"remote row for our model already has task {tid}; track it.",
                track_task_id=tid, matched_rows=strong,
            )
        if len(task_ids) > 1:
            return SubmitPlan(
                _common.SubmitDecision.ABORT_IDENTITY,
                f"remote shows multiple distinct tasks for our model: {sorted(task_ids)}; "
                f"reconcile before acting (possible duplicate).",
                matched_rows=strong,
            )
        return SubmitPlan(
            _common.SubmitDecision.ABORT_UNCERTAIN,
            "a prior model or trace exists without a confirmed task; backend retry "
            "idempotency is unverified. Reconcile authoritatively before any write.",
            matched_rows=strong,
        )

    if prior_stage in _NON_FRESH_STAGES or sub.get("historical") == "true":
        return SubmitPlan(
            _common.SubmitDecision.ABORT_UNCERTAIN,
            "prior submission may have written externally but has no usable identity; "
            "reconcile before any write.",
        )

    # RF -- no strong anchors of our own: rely on remote, where name is only a WEAK signal.
    name_only = [r for r in rows if _name_match(r, our)]
    if name_only:
        return SubmitPlan(
            _common.SubmitDecision.ABORT_IDENTITY,
            f"a solution named '{expected_name}' already exists on this subnet "
            f"(rows={name_only}); a NAME is not submission identity. Confirm it is ours "
            f"(content digest / description / created time) before tracking or resubmitting.",
            matched_rows=name_only,
        )
    return SubmitPlan(
        _common.SubmitDecision.ABORT_UNCERTAIN,
        "no matching submission found, but missing or unverified list evidence cannot "
        "prove the account history is complete; reconcile before any write.",
    )
