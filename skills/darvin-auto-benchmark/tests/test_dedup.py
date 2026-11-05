"""FIX 2 + FIX 5 -- submission dedup after interruption + true task identity.

These pin the corrected decision rules, including the cases that previously led
to duplicate submissions:
  * interrupted submit, result unknown  -> halt until authoritative reconciliation
  * model saved but no task             -> halt; retry idempotency is unverified
  * confirmed task anywhere             -> track, never re-submit
  * attempt changed on prior work       -> ABORT (would duplicate)
  * remote name collision, no strong id -> ABORT (name is not identity)
"""
import unittest

from lib import _common, dedup
from tests import _util


def receipt(stage, *, model_app_id=None, task_id=None, submission_id=None,
            attempt="default", subnet="0xabc"):
    return {
        "stage": stage, "model_app_id": model_app_id, "task_id": task_id,
        "submission_id": submission_id, "attempt": attempt,
        "subnet_address": subnet, "trace_id": "v1:trace", "historical": False,
    }


class Dedup(unittest.TestCase):
    D = _common.SubmitDecision

    # R1 -- confirmed task id => track, never resubmit
    def test_confirmed_task_from_receipt_reuses(self):
        plan = dedup.decide_submit(receipt("task_created", model_app_id=123, task_id="T-EXIST"))
        self.assertEqual(plan.decision, self.D.REUSE_EXISTING)
        self.assertEqual(plan.track_task_id, "T-EXIST")

    def test_confirmed_task_from_submit_report_reuses(self):
        rep = _util.load("submit_receipt_task_created.json")
        plan = dedup.decide_submit(None, rep)
        self.assertEqual(plan.decision, self.D.REUSE_EXISTING)
        self.assertEqual(plan.track_task_id, "T-NEW")

    # R5 -- attempt changed on prior work => abort (duplicate guard)
    def test_attempt_change_on_prior_aborts(self):
        plan = dedup.decide_submit(
            receipt("model_saved", model_app_id=131, attempt="a1"),
            expected_attempt="a2",
        )
        self.assertEqual(plan.decision, self.D.ABORT_IDENTITY)
        self.assertIn("duplicate", plan.reason.lower())

    def test_attempt_same_on_prior_does_not_abort(self):
        plan = dedup.decide_submit(
            receipt("model_saved", model_app_id=131, attempt="a1"),
            expected_attempt="a1",
        )
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)
        self.assertFalse(plan.rerun_same_attempt)

    # R2 -- historical submit, model present => reconcile
    def test_historical_model_halts(self):
        rep = _util.load("submit_receipt_historical_model.json")
        plan = dedup.decide_submit(receipt("model_saved", model_app_id=131, attempt="a1"), rep,
                                   expected_attempt="a1")
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)
        self.assertFalse(plan.rerun_same_attempt)

    # RM -- interrupted 'unknown', remote now shows the model WITH a task => reuse
    def test_unknown_interruption_remote_has_task_reuses(self):
        rows = _util.solution_rows("solution_list_one_task.json")  # model 123 -> task T-EXIST
        plan = dedup.decide_submit(receipt("unknown", model_app_id=123, attempt="a1"), remote_rows=rows,
                                   expected_attempt="a1")
        self.assertEqual(plan.decision, self.D.REUSE_EXISTING)
        self.assertEqual(plan.track_task_id, "T-EXIST")

    # RM -- a model without a confirmed task requires reconciliation
    def test_unknown_interruption_remote_no_task_halts(self):
        rows = _util.solution_rows("solution_list_model_no_task.json")  # model 123, task "-"
        plan = dedup.decide_submit(receipt("unknown", model_app_id=123, attempt="a1"), remote_rows=rows,
                                   expected_attempt="a1")
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)
        self.assertFalse(plan.rerun_same_attempt)

    # RM -- an empty listing does not prove that a write failed
    def test_unknown_interruption_remote_empty_halts(self):
        plan = dedup.decide_submit(receipt("unknown", model_app_id=123, attempt="a1"), remote_rows=[],
                                   expected_attempt="a1")
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)
        self.assertFalse(plan.rerun_same_attempt)

    def test_unknown_without_anchor_halts(self):
        plan = dedup.decide_submit({"stage": "unknown"}, remote_rows=[])
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)
        self.assertFalse(plan.rerun_same_attempt)

    # RM -- model anchor but NO known attempt (receipt unread) => abort uncertain (no guess)
    def test_model_known_but_attempt_unknown_aborts_uncertain(self):
        # submit report says a model was saved, but we have no receipt with an attempt
        rep = _util.load("submit_receipt_task_created.json")
        rep = dict(rep)  # shallow copy
        # strip task id so we're in the model-only branch, keep model_app_id
        for sec in rep["sections"]:
            for f in sec.get("fields", []):
                if f["name"] == "task_id":
                    f["value"] = "-"
        plan = dedup.decide_submit(None, rep)
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)

    # RM -- two distinct tasks for the same model => conflict, abort
    def test_two_tasks_for_same_model_aborts(self):
        rows = _util.solution_rows("solution_list_two_tasks.json")  # model 123 -> T-A and T-B
        plan = dedup.decide_submit(receipt("unknown", model_app_id=123, attempt="a1"), remote_rows=rows,
                                   expected_attempt="a1")
        self.assertEqual(plan.decision, self.D.ABORT_IDENTITY)

    # RF -- fresh, remote NAME collision with a task, but no strong id => abort (name != identity)
    def test_fresh_name_collision_aborts(self):
        rows = _util.solution_rows("solution_list_name_collision.json")  # name image-auth, model 999, task T-OTHERS
        plan = dedup.decide_submit(None, None, rows, expected_name="image-auth",
                                   expected_subnet="0xabc")
        self.assertEqual(plan.decision, self.D.ABORT_IDENTITY)
        self.assertIn("name is not", plan.reason.lower())

    # A bare list does not establish account identity or complete pagination.
    def test_empty_list_does_not_authorize_submission(self):
        rows = _util.solution_rows("solution_list_empty.json")
        plan = dedup.decide_submit(None, None, rows, expected_name="brand-new-solution",
                                   expected_subnet="0xabc")
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)

    def test_missing_list_does_not_authorize_submission(self):
        plan = dedup.decide_submit(None, expected_name="brand-new-solution",
                                   expected_subnet="0xabc")
        self.assertEqual(plan.decision, self.D.ABORT_UNCERTAIN)

    # identity extraction
    def test_identity_from_receipt_ignores_sentinels(self):
        ident = dedup.identity_from_receipt(receipt("model_saved", model_app_id=131, task_id="-"))
        self.assertEqual(ident["model_app_id"], "131")
        self.assertIsNone(ident["task_id"])


if __name__ == "__main__":
    unittest.main()
