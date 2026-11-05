"""FIX 4 -- precise task-status parsing.

The defect: a blind recursive walk for the first `name=="status"` conflated the
top-level COMMAND state with the TASK state and could be misled by the Jobs
table or the model-only Solution section. These tests pin the section-aware
parser.
"""
import unittest

from lib import status
from tests import _util


class ParseStatus(unittest.TestCase):
    def test_success_task(self):
        info = status.parse_status_report(_util.load("status_task_success.json"))
        self.assertTrue(info.command_ok)
        self.assertTrue(info.has_task)
        self.assertEqual(info.task_id, "T-SUCC")
        self.assertEqual(info.task_state, "success")
        self.assertTrue(info.terminal)
        self.assertEqual(info.outcome, "evaluation_complete")
        self.assertEqual(info.model_app_id, "123")

    def test_command_success_but_task_failed_is_terminal_not_success(self):
        # The KEY trap: report top-level status == "success" (command ran) while
        # the Task section status == "failed". The parser must NOT read the
        # top-level "success" as the task state.
        info = status.parse_status_report(_util.load("status_task_failed.json"))
        self.assertEqual(info.command_state, "success")
        self.assertTrue(info.command_ok)
        self.assertEqual(info.task_state, "failed")
        self.assertTrue(info.terminal)
        self.assertEqual(info.outcome, "ended_not_success:failed")
        self.assertTrue(status.should_download_logs(info))

    def test_running_is_not_terminal(self):
        info = status.parse_status_report(_util.load("status_task_running.json"))
        self.assertFalse(info.terminal)
        self.assertEqual(info.task_state, "running")
        self.assertEqual(info.outcome, "non_terminal:running")
        self.assertFalse(status.should_download_logs(info))

    def test_mixed_is_terminal_non_success(self):
        info = status.parse_status_report(_util.load("status_task_mixed.json"))
        self.assertTrue(info.terminal)
        self.assertEqual(info.task_state, "mixed")
        self.assertEqual(info.outcome, "ended_not_success:mixed")

    def test_model_only_has_no_task(self):
        info = status.parse_status_report(_util.load("status_model_only.json"))
        self.assertTrue(info.model_only)
        self.assertFalse(info.has_task)
        self.assertIsNone(info.task_state)  # task state must be None, not the Solution status
        self.assertEqual(info.outcome, "model_only")
        self.assertEqual(info.solution_status, "MODEL_SAVED")
        self.assertFalse(status.should_download_logs(info))

    def test_command_failed_is_not_terminal(self):
        # A failed READ (network) must not be mistaken for a terminal task.
        info = status.parse_status_report(_util.load("status_command_failed.json"))
        self.assertFalse(info.command_ok)
        self.assertEqual(info.error_code, "NETWORK_ERROR")
        self.assertFalse(info.has_task)
        self.assertFalse(info.terminal)

    def test_jobs_status_column_does_not_leak_into_task_state(self):
        # The Jobs table has a "Status" column (e.g. SUCCEEDED/FAILED) that must
        # never be read as the task state.
        info = status.parse_status_report(_util.load("status_task_failed.json"))
        self.assertEqual(info.task_state, "failed")  # from Task section, not Jobs column
        # and score is taken from the Jobs table for task mode
        self.assertIsNone(info.score) or self.assertNotEqual(info.task_state, "SUCCEEDED")

    def test_terminal_set_matches_cli(self):
        self.assertEqual(
            status.is_terminal("success"), True)
        for s in ("failed", "killed", "cancelled", "mixed"):
            self.assertTrue(status.is_terminal(s))
        for s in ("running", "waiting_for_jobs", "paused", "not_killable", "unknown"):
            self.assertFalse(status.is_terminal(s))
        self.assertFalse(status.is_terminal(None))


if __name__ == "__main__":
    unittest.main()
