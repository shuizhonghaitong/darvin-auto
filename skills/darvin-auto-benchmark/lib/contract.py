"""FIX 3 -- Verify the Feishu inference contract; never assume it is unchanged.

The original `g_contract` gate fetched the doc and only saved it, printing
"verify manually before push/submit". That is not a check: nothing blocks a
wrong endpoint/field/label set from reaching evaluation (where any inference
failure scores 0).

This module turns the fetched document text into an OBSERVED contract and diffs
it against the EXPECTED contract the solution was built for. Three outcomes:

  MATCH      reserved for independently verified semantic evidence
  MISMATCH   an expected anchor conflicts with the document          -> block, show diffs
  INCOMPLETE the document is missing / unfetched / key anchors absent -> NOT verified;

"A link or title is not evidence of the inference contract." Only MATCH authorizes
push/submit; INCOMPLETE is treated as a block (fail closed), not a pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import _common

_ANCHOR_STATUS = {"ok", "conflict", "missing"}


@dataclass
class Finding:
    key: str
    label: str
    expected: str
    observed: str
    status: str  # "ok" | "conflict" | "missing"

    @property
    def good(self) -> bool:
        return self.status == "ok"


@dataclass
class ContractPlan:
    result: _common.ContractResult
    findings: List[Finding] = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return self.result == _common.ContractResult.MATCH

    @property
    def conflicts(self) -> List[Finding]:
        return [f for f in self.findings if f.status == "conflict"]

    @property
    def missing(self) -> List[Finding]:
        return [f for f in self.findings if f.status == "missing"]

    def summary(self) -> str:
        if self.verified:
            return f"contract verified: {len(self.findings)} anchor(s) matched"
        bits = []
        if self.conflicts:
            bits.append("conflict: " + "; ".join(f"{f.key}={f.observed!r}" for f in self.conflicts))
        if self.missing:
            bits.append("missing: " + "; ".join(f.key for f in self.missing))
        return f"contract NOT verified ({self.result.value}): " + " | ".join(bits)


def _esc(text: str) -> str:
    return re.escape(text)


def build_anchors(spec: Dict) -> List[Dict]:
    """Build a general, leaderboard-agnostic anchor set from a small spec dict.

    spec keys (all optional except endpoint):
      endpoint: "/api/v1/..."
      method: "POST"
      request_fields: ["image_base64", ...]
      response_fields: ["success", "result", ...]
      valid_results: ["1", "0_1", ...]   # tokens that must each appear
      sample_count: 120
    """
    anchors: List[Dict] = []
    endpoint = spec.get("endpoint")
    if endpoint:
        # conflict = some other /api/... path present and ours absent (checked in verify)
        anchors.append({"key": "endpoint", "label": "inference endpoint", "expected": endpoint,
                        "kind": "endpoint"})
    if spec.get("method"):
        anchors.append({"key": "method", "label": "HTTP method", "expected": spec["method"],
                        "kind": "method"})
    for f in spec.get("request_fields") or []:
        anchors.append({"key": f"req:{f}", "label": f"request field {f}", "expected": f,
                        "kind": "token"})
    for f in spec.get("response_fields") or []:
        anchors.append({"key": f"resp:{f}", "label": f"response field {f}", "expected": f,
                        "kind": "token"})
    for v in spec.get("valid_results") or []:
        anchors.append({"key": f"result:{v}", "label": f"valid result {v}", "expected": v,
                        "kind": "token", "word": True})
    if spec.get("sample_count") is not None:
        anchors.append({"key": "sample_count", "label": "sample count",
                        "expected": str(spec["sample_count"]), "kind": "token", "word": True})
    return anchors


def verify_contract(doc_text: Optional[str], anchors: List[Dict]) -> ContractPlan:
    """Diff the document text against the expected anchors (fail closed)."""
    if not doc_text or not doc_text.strip():
        findings = [
            Finding(a["key"], a["label"], a["expected"], "(no document text)", "missing")
            for a in anchors
        ]
        return ContractPlan(_common.ContractResult.INCOMPLETE, findings)

    text = doc_text
    findings: List[Finding] = []

    for a in anchors:
        kind = a["kind"]
        expected = a["expected"]
        if kind == "endpoint":
            if _esc(expected) and expected in text:
                findings.append(Finding(a["key"], a["label"], expected, expected, "ok"))
            else:
                # Any other /api/<ver>/<path> present => conflict, else missing.
                other = re.search(r"/api/v\d+/[A-Za-z0-9_\-/]+", text)
                if other and other.group(0) != expected:
                    findings.append(Finding(a["key"], a["label"], expected, other.group(0), "conflict"))
                else:
                    findings.append(Finding(a["key"], a["label"], expected, "(not found)", "missing"))
        elif kind == "method":
            want = expected.upper()
            opposite = "GET" if want == "POST" else "POST"
            if re.search(rf"\b{want}\b", text):
                findings.append(Finding(a["key"], a["label"], want, want, "ok"))
            elif re.search(rf"\b{opposite}\b", text) and not re.search(rf"\b{want}\b", text):
                findings.append(Finding(a["key"], a["label"], want, opposite, "conflict"))
            else:
                findings.append(Finding(a["key"], a["label"], want, "(not found)", "missing"))
        else:  # token
            if a.get("word"):
                present = re.search(rf"(?<![A-Za-z0-9]){_esc(expected)}(?![A-Za-z0-9])", text) is not None
            else:
                present = expected in text
            if present:
                findings.append(Finding(a["key"], a["label"], expected, expected, "ok"))
            else:
                findings.append(Finding(a["key"], a["label"], expected, "(not found)", "missing"))

    if any(f.status == "conflict" for f in findings):
        result = _common.ContractResult.MISMATCH
    elif any(f.status == "missing" for f in findings):
        result = _common.ContractResult.INCOMPLETE
    else:
        # Token presence alone cannot associate fields with request/response,
        # establish endpoint semantics, or rule out conflicting instructions.
        findings.append(Finding("semantic_verification", "semantic verification", "verified contract",
                                "string matches do not establish semantics", "missing"))
        result = _common.ContractResult.INCOMPLETE
    return ContractPlan(result, findings)
