"""Recovery -- atomic/locked state + non-idempotent gate reconciliation.

The defect: a bare per-gate "done" boolean made gate recovery unreliable. These
tests pin (a) atomic, fail-closed state persistence and (b) the reconciliation
truth table that merges a local marker with authoritative external evidence.
"""
import json
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from lib import _common
from lib.state import (
    PipelineState,
    StateCorrupt,
    execute_non_idempotent_gate,
    reconcile_non_idempotent_gate,
)


def hold_operation(path, entered):
    state = PipelineState(Path(path))
    with state.operation_lock():
        entered.send(True)
        time.sleep(30)


def concurrent_gate(path, remote_path, start, results):
    state = PipelineState(Path(path))
    remote = Path(remote_path)
    start.wait(5)
    def inspect():
        return 'confirmed_done' if remote.exists() else 'confirmed_safe'
    def perform():
        time.sleep(0.1)
        with remote.open('a', encoding='utf-8') as handle:
            handle.write('created\n')
    result = execute_non_idempotent_gate(state, 'submit', inspect=inspect, perform=perform)
    results.put(result.verdict.value)


class StateAtomicity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "PIPELINE_STATE.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_set_get_roundtrip(self):
        st = PipelineState(self.path)
        st.set("identity.image_digest", "sha256:abc")
        st.set("gates.push.marker", "done")
        self.assertEqual(st.get("identity.image_digest"), "sha256:abc")
        self.assertEqual(st.get("gates.push.marker"), "done")

    def test_missing_lock_support_fails_closed(self):
        with patch('lib.state.fcntl', None):
            with self.assertRaisesRegex(RuntimeError, 'locking is unavailable'):
                PipelineState(self.path)

    def test_operation_lock_serializes_processes_and_survives_crash(self):
        state = PipelineState(self.path)
        ctx = multiprocessing.get_context('fork')
        recv1, send1 = ctx.Pipe(duplex=False)
        recv2, send2 = ctx.Pipe(duplex=False)
        first = ctx.Process(target=hold_operation, args=(str(self.path), send1))
        second = ctx.Process(target=hold_operation, args=(str(self.path), send2))
        try:
            first.start()
            self.assertTrue(recv1.poll(5))
            self.assertTrue(recv1.recv())
            second.start()
            self.assertFalse(recv2.poll(0.3))
            # A killed owner releases its kernel lock, without clearing markers.
            state.mark_gate_uncertain('submit')
            first.terminate()
            first.join(5)
            self.assertTrue(recv2.poll(5))
            self.assertTrue(recv2.recv())
            self.assertEqual(state.gate_marker('submit'), 'uncertain')
        finally:
            for process in (first, second):
                if process.pid is not None:
                    if process.is_alive():
                        process.terminate()
                    process.join(5)
            for connection in (recv1, send1, recv2, send2):
                connection.close()

    def test_get_missing_returns_default(self):
        st = PipelineState(self.path)
        self.assertIsNone(st.get("nope.nope"))
        self.assertEqual(st.get("nope.nope", "dflt"), "dflt")

    def test_no_partial_file_on_disk(self):
        st = PipelineState(self.path)
        for i in range(200):
            st.set("counter", i)
        # after many writes the on-disk file must always be valid JSON
        data = json.loads(self.path.read_text())
        self.assertEqual(data["counter"], 199)
        # and no stray temp files left behind
        leftovers = [f for f in os.listdir(self.path.parent) if f.startswith(".state-")]
        self.assertEqual(leftovers, [])

    def test_corrupt_state_fails_closed(self):
        st = PipelineState(self.path)
        st.set("x", 1)
        self.path.write_text("{ this is not json")
        with self.assertRaises(StateCorrupt):
            st.get("x")

    def test_non_object_and_missing_state_fail_closed(self):
        st = PipelineState(self.path)
        for content in ('[]', 'null'):
            self.path.write_text(content)
            with self.assertRaises(StateCorrupt):
                st.set('x', 1)
        self.path.unlink()
        with self.assertRaises(StateCorrupt):
            st.get('x')

    def test_malformed_gate_never_looks_unexecuted(self):
        st = PipelineState(self.path)
        for gates in (None, [], {'submit': None}, {'submit': {}},
                      {'submit': {'marker': None}}, {'submit': {'marker': 'typo'}}):
            self.path.write_text(json.dumps({'gates': gates}))
            with self.subTest(gates=gates), self.assertRaises(StateCorrupt):
                execute_non_idempotent_gate(st, 'submit',
                    inspect=lambda: 'confirmed_safe',
                    perform=lambda: self.fail('must not write'))

    def test_concurrent_initialization_does_not_overwrite_update(self):
        entered = threading.Event()
        second_write = threading.Event()
        release = threading.Event()
        real_write = PipelineState._write
        errors = []

        def delayed_write(instance, obj):
            if not entered.is_set():
                entered.set()
                if not release.wait(5):
                    raise RuntimeError('test timed out')
            else:
                second_write.set()
            real_write(instance, obj)

        def initialize():
            try:
                PipelineState(self.path)
            except Exception as exc:
                errors.append(exc)

        with patch.object(PipelineState, '_write', delayed_write):
            first = threading.Thread(target=initialize)
            first.start()
            self.assertTrue(entered.wait(5))
            second = threading.Thread(target=initialize)
            second.start()
            self.assertFalse(second_write.wait(0.2))
            release.set()
            first.join(5)
            second.join(5)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(json.loads(self.path.read_text())['gates'], {})


class ReconcileTable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "S.json"
        self.st = PipelineState(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def _r(self, gate, ext):
        return reconcile_non_idempotent_gate(self.st, gate, external_verdict=ext).verdict

    def _mark(self, gate, marker):
        if marker == "done":
            self.st.mark_gate_done(gate)
        elif marker == "uncertain":
            self.st.mark_gate_uncertain(gate)

    def test_done_confirmed_done_skips(self):
        self._mark("push", "done")
        self.assertEqual(self._r("push", "confirmed_done"), _common.GateVerdict.SKIP_DONE)

    def test_done_confirmed_safe_halts_on_contradiction(self):
        self._mark("push", "done")
        self.assertEqual(self._r("push", "confirmed_safe"), _common.GateVerdict.HALT)

    def test_done_conflict_halts(self):
        self._mark("push", "done")
        self.assertEqual(self._r("push", "conflict"), _common.GateVerdict.HALT)

    def test_done_unknown_halts(self):
        self._mark("push", "done")
        self.assertEqual(self._r("push", "unknown"), _common.GateVerdict.HALT)

    def test_uncertain_confirmed_done_adopts(self):
        self._mark("submit", "uncertain")
        self.assertEqual(self._r("submit", "confirmed_done"), _common.GateVerdict.SKIP_DONE)

    def test_uncertain_confirmed_safe_halts(self):
        self._mark("submit", "uncertain")
        self.assertEqual(self._r("submit", "confirmed_safe"), _common.GateVerdict.HALT)

    def test_uncertain_unknown_halts(self):
        self._mark("submit", "uncertain")
        self.assertEqual(self._r("submit", "unknown"), _common.GateVerdict.HALT)

    def test_uncertain_conflict_halts(self):
        self._mark("submit", "uncertain")
        self.assertEqual(self._r("submit", "conflict"), _common.GateVerdict.HALT)

    def test_none_confirmed_safe_proceeds(self):
        self.assertEqual(self._r("push", "confirmed_safe"), _common.GateVerdict.PROCEED)

    def test_none_confirmed_done_skips(self):
        self.assertEqual(self._r("push", "confirmed_done"), _common.GateVerdict.SKIP_DONE)

    def test_none_unknown_halts(self):
        self.assertEqual(self._r("push", "unknown"), _common.GateVerdict.HALT)

    def test_unrecognized_external_halts(self):
        self._mark("push", "done")
        self.assertEqual(self._r("push", "wat"), _common.GateVerdict.HALT)


class OperationExecution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = PipelineState(Path(self.tmp.name) / 'state.json')

    def tearDown(self):
        self.tmp.cleanup()

    def test_confirmed_write_and_rerun(self):
        remote = []
        inspect = lambda: 'confirmed_done' if remote else 'confirmed_safe'
        perform = lambda: remote.append('created')
        first = execute_non_idempotent_gate(self.state, 'submit', inspect=inspect, perform=perform)
        second = execute_non_idempotent_gate(self.state, 'submit', inspect=inspect, perform=perform)
        self.assertEqual(first.verdict, _common.GateVerdict.SKIP_DONE)
        self.assertEqual(second.verdict, _common.GateVerdict.SKIP_DONE)
        self.assertEqual(remote, ['created'])
        self.assertEqual(self.state.gate_marker('submit'), 'done')

    def test_ambiguous_write_is_not_retried(self):
        calls = []
        def fail():
            calls.append(1)
            raise RuntimeError('connection lost')
        with self.assertRaises(RuntimeError):
            execute_non_idempotent_gate(self.state, 'submit', inspect=lambda: 'confirmed_safe', perform=fail)
        result = execute_non_idempotent_gate(self.state, 'submit', inspect=lambda: 'confirmed_safe', perform=fail)
        self.assertEqual(result.verdict, _common.GateVerdict.HALT)
        self.assertEqual(calls, [1])
        self.assertEqual(self.state.gate_marker('submit'), 'uncertain')

    def test_failed_confirmation_remains_uncertain(self):
        checks = iter(['confirmed_safe', 'unknown'])
        result = execute_non_idempotent_gate(self.state, 'push', inspect=lambda: next(checks), perform=lambda: None)
        self.assertEqual(result.verdict, _common.GateVerdict.HALT)
        self.assertEqual(self.state.gate_marker('push'), 'uncertain')

    def test_unknown_initial_state_never_writes(self):
        calls = []
        result = execute_non_idempotent_gate(self.state, 'push', inspect=lambda: 'unknown', perform=lambda: calls.append(1))
        self.assertEqual(result.verdict, _common.GateVerdict.HALT)
        self.assertEqual(calls, [])

    def test_concurrent_processes_execute_once(self):
        ctx = multiprocessing.get_context('fork')
        start = ctx.Event()
        results = ctx.Queue()
        remote = Path(self.tmp.name) / 'remote'
        processes = [ctx.Process(target=concurrent_gate,
                     args=(str(self.state.path), str(remote), start, results)) for _ in range(2)]
        try:
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(5)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(remote.read_text().splitlines(), ['created'])
            self.assertEqual([results.get(timeout=1) for _ in processes],
                             [_common.GateVerdict.SKIP_DONE.value] * 2)
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            results.close()


if __name__ == "__main__":
    unittest.main()
