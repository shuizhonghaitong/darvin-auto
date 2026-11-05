"""General, leaderboard-AGNOSTIC planner for the Darvin auto-benchmark flow.

This module is a PURE decision engine. It reads local state plus *injected*
authoritative external evidence and returns the single safe next action per
gate. It performs NO real writes: no docker, no darvin, no push, no submit. The
shell that would actually invoke those tools
is deliberately kept DISABLED; wiring a real executor behind an explicit,
off-by-default authorization gate is the only way it may run.

Why a planner (not a runner) lives in the skill:
  * the 5 recorded defects are all DECISION defects (what to do given evidence),
    so they are exactly what can be unit-tested offline and safely;
  * future leaderboards reuse the same decision logic with a different config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from . import _common, contract, dedup, registry, status
from .state import PipelineState

# Actions a gate can be told to take.
A_QUERY = "query_external"          # need authoritative evidence before deciding
A_STOP = "stop"                     # terminal / conflict / unknown -> stop + report
A_POLL = "poll"                     # non-terminal task -> keep polling within bounds
A_TRACK = "track_existing"          # do not write; track the identified task
A_COMPARE_DIGEST = "compare_digest" # tag exists; compare digests, do NOT overwrite
A_FIRST_PUBLISH = "first_publish"   # registry ABSENT (confirmed) -> safe to create
A_SUBMIT_FRESH = "submit_fresh"     # no prior evidence -> submit once
A_CONTRACT_VERIFIED = "contract_verified"

_WRITE_ACTIONS = (A_FIRST_PUBLISH, A_SUBMIT_FRESH)


@dataclass
class GatePlan:
    gate: str
    verdict: _common.GateVerdict
    action: str
    reason: str
    task_id: Optional[str] = None
    digest: Optional[str] = None
    requires_authorization: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_write(self) -> bool:
        return self.action in _WRITE_ACTIONS


@dataclass
class PipelineConfig:
    leaderboard_id: str
    solution_name: str
    image_ref: str                      # "repo:tag" for the self-published registry path
    subnet_keyword: str
    subnet_address: Optional[str] = None
    attempt: str = "default"
    contract_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Evidence:
    """Authoritative external inputs for one planning pass (all injectable)."""
    registry: Optional[registry.RegistryResult] = None
    solution_list_rows: Optional[List[Dict[str, str]]] = None
    local_receipt: Optional[Dict[str, Any]] = None
    last_submit_report: Optional[Dict[str, Any]] = None
    feishu_doc_text: Optional[str] = None
    status_report: Optional[Dict[str, Any]] = None


class Pipeline:
    def __init__(self, cfg: PipelineConfig, state: PipelineState, *, enabled: bool = False,
                 authorize: Optional[Callable[[GatePlan], bool]] = None) -> None:
        self.cfg = cfg
        self.state = state
        self.enabled = enabled
        self.authorize = authorize or (lambda plan: False)

    # ---- FIX 1: push (registry tag existence) ------------------------------
    def plan_push(self, ev: Evidence) -> GatePlan:
        reg = ev.registry
        if reg is None:
            return GatePlan("push", _common.GateVerdict.QUERY, A_QUERY,
                            "no registry evidence; run an authoritative tag lookup first")
        if reg.verdict == _common.RegistryVerdict.ERROR:
            return GatePlan("push", _common.GateVerdict.HALT, A_STOP,
                            f"registry lookup ERROR ({reg.error_kind}): a query failure does "
                            f"not prove the tag is absent; will not publish. {reg.reason}")
        if reg.verdict == _common.RegistryVerdict.PRESENT:
            local_digest = self.state.get("identity.image_digest")
            if local_digest and reg.digest and local_digest == reg.digest:
                return GatePlan("push", _common.GateVerdict.SKIP_DONE, A_STOP,
                                f"tag exists and digest matches our build ({reg.digest}); no publish needed",
                                digest=reg.digest)
            return GatePlan("push", _common.GateVerdict.HALT, A_COMPARE_DIGEST,
                            "tag already exists; compare digests and do NOT overwrite without "
                            "verified authorization", digest=reg.digest, requires_authorization=True)
        # ABSENT -> first publish is safe (real write still needs executor authorization)
        return GatePlan("push", _common.GateVerdict.PROCEED, A_FIRST_PUBLISH,
                        "registry confirms tag absent; first publish is safe",
                        requires_authorization=True)

    # ---- FIX 3: contract (feishu) ------------------------------------------
    def plan_contract(self, ev: Evidence) -> GatePlan:
        anchors = contract.build_anchors(self.cfg.contract_spec)
        plan = contract.verify_contract(ev.feishu_doc_text, anchors)
        if plan.verified:
            return GatePlan("contract", _common.GateVerdict.PROCEED, A_CONTRACT_VERIFIED,
                            plan.summary())
        return GatePlan("contract", _common.GateVerdict.HALT, A_STOP,
                        "contract not verified; push/submit blocked. " + plan.summary(),
                        details={"findings": [f.__dict__ for f in plan.findings]})

    # ---- FIX 2+5: submit (dedup + identity) --------------------------------
    def plan_submit(self, ev: Evidence) -> GatePlan:
        plan = dedup.decide_submit(
            ev.local_receipt,
            ev.last_submit_report,
            ev.solution_list_rows,
            expected_name=self.cfg.solution_name,
            expected_subnet=self.cfg.subnet_address,
            expected_attempt=self.cfg.attempt,
        )
        d = plan.decision
        if d == _common.SubmitDecision.REUSE_EXISTING:
            return GatePlan("submit", _common.GateVerdict.SKIP_DONE, A_TRACK,
                            plan.reason, task_id=plan.track_task_id)
        if d == _common.SubmitDecision.SUBMIT_FRESH:
            return GatePlan("submit", _common.GateVerdict.PROCEED, A_SUBMIT_FRESH,
                            plan.reason, requires_authorization=True)
        return GatePlan("submit", _common.GateVerdict.HALT, A_STOP, plan.reason)  # ABORT_*

    # ---- FIX 4: track (status parsing) -------------------------------------
    def plan_track(self, ev: Evidence) -> GatePlan:
        if ev.status_report is None:
            return GatePlan("track", _common.GateVerdict.QUERY, A_QUERY,
                            "no status report; query the task first")
        st = status.parse_status_report(ev.status_report)
        if not st.command_ok:
            return GatePlan("track", _common.GateVerdict.RECONCILE, A_POLL,
                            f"status command failed (state={st.command_state}, code={st.error_code}); "
                            f"retry the read, do not conclude", task_id=st.task_id)
        if st.model_only and not st.has_task:
            return GatePlan("track", _common.GateVerdict.HALT, A_STOP,
                            "solution has no training task (model-only); create one if a task was "
                            "intended, or treat as saved-not-run", task_id=st.task_id)
        if st.terminal:
            if st.task_state == "success":
                return GatePlan("track", _common.GateVerdict.SKIP_DONE, A_STOP,
                                "evaluation complete (task success)", task_id=st.task_id,
                                details={"score": st.score, "logs": False})
            return GatePlan("track", _common.GateVerdict.HALT, A_STOP,
                            f"task ended non-success: {st.task_state}; pull logs for evidence",
                            task_id=st.task_id, details={"score": st.score, "logs": True})
        return GatePlan("track", _common.GateVerdict.RECONCILE, A_POLL,
                        f"task non-terminal ({st.task_state}); keep polling within bounds",
                        task_id=st.task_id)

    # ---- boundary guard (documents the off-by-default write gate) ----------
    def write_allowed(self, plan: GatePlan) -> bool:
        """A real write requires the executor to be enabled, the plan to be a
        write action, and an explicit authorizer to grant it. All are False by
        default, so this planner can never trigger a write on its own."""
        return self.enabled and plan.is_write and self.authorize(plan)
