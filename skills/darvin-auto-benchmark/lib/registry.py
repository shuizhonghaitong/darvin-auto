"""FIX 1 -- Distinguish a registry 'tag absent' from a registry *query error*.

The original pipeline did:

    if docker manifest inspect IMG >/dev/null 2>&1; then present
    else absent   # <-- BUG: ANY failure (auth, DNS, timeout, 5xx) read as "absent"

and then pushed, silently overwriting a tag it could not actually see. "Registry
lookup errors do not establish that a tag is absent" is the invariant this module
enforces.

The classifier is conservative by construction:
  * exit 0  + a digest in the output      -> PRESENT
  * exit 0  + no digest                   -> ERROR (cannot confirm; do not assume)
  * exit !=0                              -> ERROR (including ambiguous not-found)

`ABSENT` is the ONLY verdict that authorizes a first-time create/push. `ERROR`
must halt the gate with actionable evidence, never fall through to a write.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from . import _common

# Error sub-types, used only to produce actionable guidance. None of these is
# ever treated as ABSENT.
_AUTH_RE = re.compile(
    r"unauthorized|authentication required|auth config|login required|"
    r"denied: requested access|permission denied.*registry|\b401\b",
    re.IGNORECASE,
)
_NETWORK_RE = re.compile(
    r"connection (?:refused|reset)|no such host|temporary failure in name resolution|"
    r"dial tcp|lookup .*|i\/o timeout|context deadline exceeded|tls:|x509|"
    r"network is unreachable|connection timed out|\b503\b|\b502\b|\b504\b",
    re.IGNORECASE,
)


@dataclass
class RegistryResult:
    image_ref: str
    verdict: _common.RegistryVerdict
    digest: Optional[str] = None
    error_kind: Optional[str] = None      # "auth" | "network" | "other" | None
    reason: str = ""
    evidence: str = ""                    # redacted command output for the run record

    @property
    def safe_to_first_publish(self) -> bool:
        return self.verdict == _common.RegistryVerdict.ABSENT

    @property
    def requires_reconcile(self) -> bool:
        """True when we know the tag exists (compare digests) or cannot confirm it."""
        return self.verdict in (
            _common.RegistryVerdict.PRESENT,
            _common.RegistryVerdict.ERROR,
        )


def _extract_digest(stdout: str) -> Optional[str]:
    """Extract only an explicitly labelled top-level manifest digest.

    A manifest's config and layer digests identify different objects.  In
    particular, recursively searching `docker manifest inspect` JSON can
    mistake one of these for the remote manifest digest.
    """
    text = (stdout or "").strip()
    if not text:
        return None
    # Preferred: valid JSON with a "digest" key (docker manifest inspect / buildx).
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        value = data.get("digest")
        if isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value, re.IGNORECASE):
            return value.lower()
        return None
    # buildx imagetools inspect explicitly labels the manifest digest.
    m = re.search(r"^Digest:\s*(sha256:[0-9a-f]{64})\s*$", text, re.IGNORECASE | re.MULTILINE)
    if m:
        return m.group(1).lower()
    return None


def _redact(text: str) -> str:
    """Strip credentials/URLs that might ride along in registry error text."""
    if not text:
        return text
    text = re.sub(r"(\btoken\s*=\s*)[^\s&]+", r"\1<redacted>", text, flags=re.IGNORECASE)
    text = re.sub(r"([?&][^\s]*token[^\s]*=)[^\s&]+", r"\1<redacted>", text, flags=re.IGNORECASE)
    text = re.sub(r"(Authorization:\s*)[^\s]+", r"\1<redacted>", text, flags=re.IGNORECASE)
    text = re.sub(r"(bearer\s+)[A-Za-z0-9._\-]+", r"\1<redacted>", text, flags=re.IGNORECASE)
    return text


def classify_lookup(
    image_ref: str,
    *,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
) -> RegistryResult:
    """Classify the outcome of a single registry tag-existence lookup.

    `stdout`/`stderr`/`exit_code` are the raw result of the lookup command
    (e.g. `docker manifest inspect <ref>` or `docker buildx imagetools inspect`).
    """
    combined_err = f"{stderr or ''}\n{stdout or ''}"
    evidence = _redact(combined_err).strip()[:2000]

    if exit_code == 0:
        digest = _extract_digest(stdout)
        if digest:
            return RegistryResult(
                image_ref, _common.RegistryVerdict.PRESENT, digest=digest,
                reason=f"tag exists (digest {digest})", evidence=evidence,
            )
        return RegistryResult(
            image_ref, _common.RegistryVerdict.ERROR, error_kind="other",
            reason="lookup succeeded but no digest was returned; cannot confirm tag state",
            evidence=evidence,
        )

    # Free-form CLI errors do not establish which server or reference produced
    # a 404, and some registries conceal authorization failures as not-found.
    # Never authorize a write based on text-pattern matching alone.

    kind = "auth" if _AUTH_RE.search(combined_err) else (
        "network" if _NETWORK_RE.search(combined_err) else "other"
    )
    return RegistryResult(
        image_ref, _common.RegistryVerdict.ERROR, error_kind=kind,
        reason=f"registry lookup failed ({kind}); this does NOT prove the tag is absent",
        evidence=evidence,
    )
