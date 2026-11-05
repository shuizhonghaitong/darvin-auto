"""Planner integration -- the five fixes composed, all offline.

Proves the planner:
  * never emits a real write unless the executor is enabled AND authorizes it
  * halts on registry query errors (FIX 1)
  * tracks existing tasks instead of resubmitting (FIX 2/5)
  * blocks push/submit unless the Feishu contract verifies (FIX 3)
  * reads the task state, not the command state, to decide terminality (FIX 4)
"""
import tempfile
import unittest
from pathlib import Path

from lib import _common, pipeline
from lib.registry import classify_lookup, RegistryResult
from tests import _util

SPEC = {
    "endpoint": "/api/v1/image-authenticity",
    "method": "POST",
    "request_fields": ["image_base64", "image_format"],
    "response_fields": ["success", "result", "message"],
}
GOOD_DOC = ("POST /api/v1/image-authenticity "
            "image_base64 image_format success result message")


def make_pipeline(test_case, enabled=False, authorize=None, subnet="0xabc"):
    tmp = tempfile.TemporaryDirectory()
    test_case.addCleanup(tmp.cleanup)
    st = pipeline.PipelineState(Path(tmp.name) / "S.json")
    cfg = pipeline.PipelineConfig(
        leaderboard_id="196", solution_name="image-auth",
        image_ref="harbor-contest.4pd.io/x/image-auth:0.0.2",
        subnet_keyword="图像真伪", subnet_address=subnet, attempt="a1",
        contract_spec=SPEC,
    )
    return pipeline.Pipeline(cfg, st, enabled=enabled, authorize=authorize)


class PushGate(unittest.TestCase):
    def test_no_evidence_queries(self):
        pl = make_pipeline(self)
        p = pl.plan_push(pipeline.Evidence())
        self.assertEqual(p.action, pipeline.A_QUERY)
        self.assertFalse(p.is_write)

    def test_registry_error_halts_not_publish(self):
        reg = classify_lookup("x:1", exit_code=1, stderr="unauthorized: authentication required")
        pl = make_pipeline(self)
        p = pl.plan_push(pipeline.Evidence(registry=reg))
        self.assertEqual(p.verdict, _common.GateVerdict.HALT)
        self.assertEqual(p.action, pipeline.A_STOP)
        self.assertFalse(p.is_write)

    def test_absent_proceeds_first_publish_but_needs_auth(self):
        # Exercise the planner branch, not a claim that CLI 404 proves absence.
        reg = RegistryResult("x:1", _common.RegistryVerdict.ABSENT)
        pl = make_pipeline(self)
        p = pl.plan_push(pipeline.Evidence(registry=reg))
        self.assertEqual(p.action, pipeline.A_FIRST_PUBLISH)
        self.assertTrue(p.requires_authorization)
        self.assertFalse(pl.write_allowed(p))  # executor disabled by default

    def test_present_same_digest_skips(self):
        reg = classify_lookup("x:1", exit_code=0, stdout='{"digest":"sha256:' + "a" * 64 + '"}')
        pl = make_pipeline(self)
        pl.state.set("identity.image_digest", "sha256:" + "a" * 64)
        p = pl.plan_push(pipeline.Evidence(registry=reg))
        self.assertEqual(p.verdict, _common.GateVerdict.SKIP_DONE)
        self.assertFalse(p.is_write)

    def test_present_different_digest_halts_compare(self):
        reg = classify_lookup("x:1", exit_code=0, stdout='{"digest":"sha256:' + "b" * 64 + '"}')
        pl = make_pipeline(self)
        pl.state.set("identity.image_digest", "sha256:" + "a" * 64)
        p = pl.plan_push(pipeline.Evidence(registry=reg))
        self.assertEqual(p.verdict, _common.GateVerdict.HALT)
        self.assertEqual(p.action, pipeline.A_COMPARE_DIGEST)
        self.assertFalse(p.is_write)


class SubmitGate(unittest.TestCase):
    def test_confirmed_task_tracks_not_submits(self):
        pl = make_pipeline(self)
        ev = pipeline.Evidence(local_receipt={"stage": "task_created", "task_id": "T-EXIST",
                                              "model_app_id": 123, "attempt": "a1"})
        p = pl.plan_submit(ev)
        self.assertEqual(p.action, pipeline.A_TRACK)
        self.assertEqual(p.task_id, "T-EXIST")
        self.assertFalse(p.is_write)

    def test_unknown_with_remote_task_tracks(self):
        pl = make_pipeline(self)
        rows = _util.solution_rows("solution_list_one_task.json")  # model 123 -> T-EXIST
        ev = pipeline.Evidence(local_receipt={"stage": "unknown", "model_app_id": 123, "attempt": "a1"},
                               solution_list_rows=rows)
        p = pl.plan_submit(ev)
        self.assertEqual(p.action, pipeline.A_TRACK)
        self.assertEqual(p.task_id, "T-EXIST")

    def test_model_saved_known_attempt_halts(self):
        pl = make_pipeline(self)
        ev = pipeline.Evidence(local_receipt={"stage": "model_saved", "model_app_id": 131,
                                              "attempt": "a1"}, solution_list_rows=[])
        p = pl.plan_submit(ev)
        self.assertEqual(p.action, pipeline.A_STOP)
        self.assertFalse(p.is_write)
        self.assertFalse(pl.write_allowed(p))

    def test_empty_unverified_list_stops(self):
        pl = make_pipeline(self)
        ev = pipeline.Evidence(solution_list_rows=_util.solution_rows("solution_list_empty.json"))
        p = pl.plan_submit(ev)
        self.assertEqual(p.action, pipeline.A_STOP)

    def test_missing_list_stops(self):
        self.assertEqual(make_pipeline(self).plan_submit(pipeline.Evidence()).action,
                         pipeline.A_STOP)

    def test_name_collision_stops(self):
        pl = make_pipeline(self)
        ev = pipeline.Evidence(solution_list_rows=_util.solution_rows("solution_list_name_collision.json"))
        p = pl.plan_submit(ev)
        self.assertEqual(p.verdict, _common.GateVerdict.HALT)
        self.assertEqual(p.action, pipeline.A_STOP)
        self.assertFalse(p.is_write)

    def test_write_never_allowed_when_disabled(self):
        pl = make_pipeline(self, enabled=False)
        p = pipeline.GatePlan("submit", _common.GateVerdict.PROCEED,
                              pipeline.A_SUBMIT_FRESH, "hypothetical")
        self.assertTrue(p.is_write)
        self.assertFalse(pl.write_allowed(p))

    def test_write_allowed_only_when_enabled_and_authorized(self):
        pl = make_pipeline(self, enabled=True, authorize=lambda p: True)
        p = pipeline.GatePlan("submit", _common.GateVerdict.PROCEED,
                              pipeline.A_SUBMIT_FRESH, "hypothetical")
        self.assertTrue(pl.write_allowed(p))
        # but a non-write plan is never a "write allowed"
        pl2 = make_pipeline(self, enabled=True, authorize=lambda p: True)
        track = pl2.plan_submit(pipeline.Evidence(local_receipt={"stage": "task_created",
                                                                "task_id": "T", "attempt": "a1"}))
        self.assertFalse(pl2.write_allowed(track))


class ContractGate(unittest.TestCase):
    def test_matching_anchors_do_not_prove_semantics(self):
        pl = make_pipeline(self)
        p = pl.plan_contract(pipeline.Evidence(feishu_doc_text=GOOD_DOC))
        self.assertEqual(p.action, pipeline.A_STOP)
        self.assertEqual(p.verdict, _common.GateVerdict.HALT)

    def test_missing_field_halts(self):
        pl = make_pipeline(self)
        p = pl.plan_contract(pipeline.Evidence(feishu_doc_text="POST /api/v1/image-authenticity image_base64"))
        self.assertEqual(p.verdict, _common.GateVerdict.HALT)
        self.assertEqual(p.action, pipeline.A_STOP)

    def test_unfetched_doc_halts(self):
        pl = make_pipeline(self)
        p = pl.plan_contract(pipeline.Evidence(feishu_doc_text=None))
        self.assertEqual(p.verdict, _common.GateVerdict.HALT)


class TrackGate(unittest.TestCase):
    def test_success_is_complete(self):
        pl = make_pipeline(self)
        p = pl.plan_track(pipeline.Evidence(status_report=_util.load("status_task_success.json")))
        self.assertEqual(p.details.get("logs"), False)
        self.assertEqual(p.verdict, _common.GateVerdict.SKIP_DONE)

    def test_command_ok_task_failed_ends_not_success(self):
        pl = make_pipeline(self)
        p = pl.plan_track(pipeline.Evidence(status_report=_util.load("status_task_failed.json")))
        self.assertEqual(p.details.get("logs"), True)
        self.assertEqual(p.verdict, _common.GateVerdict.HALT)

    def test_running_polls(self):
        pl = make_pipeline(self)
        p = pl.plan_track(pipeline.Evidence(status_report=_util.load("status_task_running.json")))
        self.assertEqual(p.action, pipeline.A_POLL)

    def test_command_failed_retries_read(self):
        pl = make_pipeline(self)
        p = pl.plan_track(pipeline.Evidence(status_report=_util.load("status_command_failed.json")))
        self.assertEqual(p.action, pipeline.A_POLL)
        self.assertEqual(p.verdict, _common.GateVerdict.RECONCILE)

    def test_model_only_stops(self):
        pl = make_pipeline(self)
        p = pl.plan_track(pipeline.Evidence(status_report=_util.load("status_model_only.json")))
        self.assertEqual(p.action, pipeline.A_STOP)


if __name__ == "__main__":
    unittest.main()
