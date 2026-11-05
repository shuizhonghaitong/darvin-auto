"""Durable, atomic, file-locked pipeline state + non-idempotent gate reconciliation.

Recorded defect this fixes: "unreliable gate recovery". The original pipeline
stored a bare `{"done": <ts>}` boolean per gate. If the process died AFTER a
non-idempotent write (push/submit) but BEFORE marking the gate done, a re-run
would silently redo the write (duplicate / overwrite). Conversely a stale
"done" marker would skip a gate whose external outcome was actually uncertain.

Two invariants are enforced here (mirroring darvin_cli's SubmissionReceiptStore):
  * State is written atomically (mkstemp + fsync + os.replace) under an flock, so
    a crash never leaves a half-written state file.
  * A non-idempotent gate's effective state is derived by MERGING the local
    marker with AUTHORITATIVE EXTERNAL evidence -- not from the local marker
    alone. When external evidence is inconclusive, the gate HALTs (fail closed).
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from . import _common

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineState:
    """A small JSON state store with atomic, locked read-modify-write."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            if not self.path.exists():
                self._write({"gates": {}, "identity": {}, "evidence": {}})

    # --- low level -----------------------------------------------------------
    @contextmanager
    def _lock(self) -> Iterator[None]:
        if fcntl is None:
            raise RuntimeError("cross-process file locking is unavailable")
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.touch(exist_ok=True)
        handle = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    @contextmanager
    def operation_lock(self) -> Iterator[None]:
        """Serialize an entire external operation for this state file.

        Hold from before external reconciliation until its outcome is persisted.
        Every process acting on the same remote identity must use the same state
        file; this lock cannot coordinate different state files or hosts without
        shared flock semantics. Never remove or replace the lock file while in use.
        """
        if fcntl is None:
            raise RuntimeError("cross-process file locking is unavailable")
        lock_path = self.path.with_suffix(self.path.suffix + ".operation.lock")
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write(self, obj: dict) -> None:
        fd, tmp = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as h:
                json.dump(obj, h, ensure_ascii=False, indent=1)
                h.write("\n")
                h.flush()
                os.fsync(h.fileno())
            os.replace(tmp, self.path)
            # Persist the directory entry too: fsync(file) alone does not make
            # the rename durable across a power loss.
            dir_fd = os.open(str(self.path.parent), os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def _read(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("state root must be an object")
            return data
        except (ValueError, OSError):
            # A corrupt state file must not be silently trusted or overwritten.
            raise StateCorrupt(self.path) from None

    # --- public API ----------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        cur: Any = self._read()
        for part in dotted_key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted_key: str, value: Any) -> None:
        with self._lock():
            data = self._read()
            parts = dotted_key.split(".")
            cur = data
            for part in parts[:-1]:
                cur = cur.setdefault(part, {})
            cur[parts[-1]] = value
            self._write(data)

    def update(self, mutator: Callable[[dict], None]) -> None:
        with self._lock():
            data = self._read()
            mutator(data)
            self._write(data)

    # --- gate markers --------------------------------------------------------
    def gate_marker(self, gate: str) -> Optional[str]:
        """Return the stored marker for a gate: 'done' | 'uncertain' | None."""
        data = self._read()
        gates = data.get("gates")
        if not isinstance(gates, dict):
            raise StateCorrupt(self.path)
        if gate not in gates:
            return None
        val = gates[gate]
        if not isinstance(val, dict) or val.get("marker") not in ("done", "uncertain"):
            raise StateCorrupt(self.path)
        return val["marker"]

    def mark_gate_done(self, gate: str, **evidence: Any) -> None:
        self.set(f"gates.{gate}", {"marker": "done", "at": _utcnow(), **evidence})

    def mark_gate_uncertain(self, gate: str, **evidence: Any) -> None:
        self.set(f"gates.{gate}", {"marker": "uncertain", "at": _utcnow(), **evidence})


class StateCorrupt(RuntimeError):
    pass


@dataclass
class ReconcileResult:
    verdict: _common.GateVerdict
    reason: str
    external: Optional[str] = None


def reconcile_non_idempotent_gate(
    state: PipelineState,
    gate: str,
    *,
    external_verdict: str,
    external_detail: str = "",
) -> ReconcileResult:
    """Merge a non-idempotent gate's local marker with external evidence.

    `external_verdict` is one of:
      "confirmed_done"   external proof the effect is already in place
                         (registry digest matches / task id exists / receipt present)
      "confirmed_safe"   external proof the effect is NOT in place and running is
                         safe (registry ABSENT / no remote anchor or name match)
      "conflict"         external evidence contradicts the local marker
      "unknown"          the external query itself failed (ERROR / NETWORK / no data)

    Rules (fail closed):
      local 'done'      + confirmed_done  -> SKIP_DONE
      local 'done'      + confirmed_safe  -> HALT (contradictory evidence)
      local 'done'      + conflict        -> HALT
      local 'done'      + unknown         -> HALT (never redo on unknown)
      local 'uncertain' + confirmed_done  -> SKIP_DONE (adopt external truth)
      local 'uncertain' + confirmed_safe  -> HALT (a prior write may still appear)
      local 'uncertain' + conflict        -> HALT
      local 'uncertain' + unknown         -> HALT
      local None          + confirmed_done-> SKIP_DONE
      local None          + confirmed_safe-> PROCEED
      local None          + conflict/unknown -> HALT
    """
    marker = state.gate_marker(gate)
    ext = external_verdict
    if ext == "confirmed_done":
        if marker in ("done", None, "uncertain"):
            return ReconcileResult(_common.GateVerdict.SKIP_DONE,
                                   "external evidence confirms the effect is already in place", ext)
        return ReconcileResult(_common.GateVerdict.HALT, "conflicting local marker", ext)
    if ext == "confirmed_safe":
        if marker == "done":
            return ReconcileResult(_common.GateVerdict.HALT,
                                   "local 'done' but external says not present; reconcile "
                                   "the contradiction before taking any action", ext)
        if marker == "uncertain":
            return ReconcileResult(_common.GateVerdict.HALT,
                                   "a previous write has an uncertain outcome; absence alone "
                                   "cannot prove that a delayed effect will not appear", ext)
        if marker is None:
            return ReconcileResult(_common.GateVerdict.PROCEED,
                                   "external evidence says safe to run exactly once", ext)
        return ReconcileResult(_common.GateVerdict.HALT, "unexpected local marker", ext)
    if ext == "conflict":
        return ReconcileResult(_common.GateVerdict.HALT,
                               f"external evidence conflicts with local state: {external_detail}", ext)
    # ext == "unknown" (or anything unrecognized) -> fail closed
    return ReconcileResult(_common.GateVerdict.HALT,
                           f"external state unknown ({external_detail or 'no data'}); "
                           f"failing closed, will not redo a non-idempotent write", ext)


def execute_non_idempotent_gate(
    state: PipelineState,
    gate: str,
    *,
    inspect: Callable[[], str],
    perform: Callable[[], None],
) -> ReconcileResult:
    """Run one effect only when authoritative checks permit it.

    The caller supplies identity-specific authoritative inspection and the
    write itself. Inspection must return a reconciliation verdict, not merely
    interpret a failed request as absence. Exceptions propagate; after the
    write starts the durable uncertain marker intentionally remains. All
    participants for an identity must share this state file and flock domain.
    """
    with state.operation_lock():
        result = reconcile_non_idempotent_gate(state, gate, external_verdict=inspect())
        if result.verdict != _common.GateVerdict.PROCEED:
            return result
        state.mark_gate_uncertain(gate)
        perform()
        result = reconcile_non_idempotent_gate(state, gate, external_verdict=inspect())
        if result.verdict == _common.GateVerdict.SKIP_DONE:
            state.mark_gate_done(gate)
        return result
