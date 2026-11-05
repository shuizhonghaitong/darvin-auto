"""FIX 3 -- Feishu inference-contract verification (fail closed).

The defect: the gate fetched the doc and just saved it ("verify manually"). These
tests pin a real check: parse the document, diff against the expected contract,
and only MATCH authorizes push/submit. Missing doc / missing anchors / a
conflicting value all BLOCK.
"""
import unittest

from lib import _common, contract

SPEC = {
    "endpoint": "/api/v1/image-authenticity",
    "method": "POST",
    "request_fields": ["image_base64", "image_format"],
    "response_fields": ["success", "result", "message"],
    "valid_results": ["1", "0_1", "0_2"],
    "sample_count": 120,
}

GOOD_DOC = """
## 接口
- 方法: POST
- 路径: /api/v1/image-authenticity
- 请求体: {"image_base64": "<base64>", "image_format": "jpg"}
- 响应: {"success": true, "result": "1|0_1|0_2", "message": ""}
- 测试样本: 120
"""


class ContractVerify(unittest.TestCase):
    R = _common.ContractResult

    def test_good_doc_requires_semantic_verification(self):
        plan = contract.verify_contract(GOOD_DOC, contract.build_anchors(SPEC))
        self.assertEqual(plan.result, self.R.INCOMPLETE)
        self.assertFalse(plan.verified)
        self.assertEqual(plan.conflicts, [])
        self.assertEqual([f.key for f in plan.missing], ["semantic_verification"])

    def test_empty_anchors_cannot_verify(self):
        self.assertEqual(contract.verify_contract(GOOD_DOC, []).result, self.R.INCOMPLETE)

    def test_wrong_endpoint_is_mismatch(self):
        doc = GOOD_DOC.replace("/api/v1/image-authenticity", "/api/v2/other-endpoint")
        # keep the good endpoint absent, only the bad one present
        doc = "POST /api/v2/other-endpoint body image_base64 image_format resp success result message results 1 0_1 0_2 120"
        plan = contract.verify_contract(doc, contract.build_anchors(SPEC))
        self.assertEqual(plan.result, self.R.MISMATCH)
        self.assertTrue(any(f.key == "endpoint" and f.status == "conflict" for f in plan.findings))

    def test_wrong_method_is_mismatch(self):
        doc = GOOD_DOC.replace("POST", "GET")
        plan = contract.verify_contract(doc, contract.build_anchors(SPEC))
        self.assertEqual(plan.result, self.R.MISMATCH)
        self.assertTrue(any(f.key == "method" and f.status == "conflict" for f in plan.findings))

    def test_missing_response_field_is_incomplete(self):
        doc = GOOD_DOC.replace('"success": true, ', "")
        plan = contract.verify_contract(doc, contract.build_anchors(SPEC))
        self.assertEqual(plan.result, self.R.INCOMPLETE)
        self.assertIn("resp:success", [f.key for f in plan.missing])
        self.assertFalse(plan.verified)

    def test_missing_doc_is_incomplete_not_match(self):
        plan = contract.verify_contract(None, contract.build_anchors(SPEC))
        self.assertEqual(plan.result, self.R.INCOMPLETE)
        self.assertFalse(plan.verified)

    def test_empty_doc_is_incomplete(self):
        plan = contract.verify_contract("   \n  ", contract.build_anchors(SPEC))
        self.assertEqual(plan.result, self.R.INCOMPLETE)

    def test_summary_reports_diffs(self):
        doc = GOOD_DOC.replace('"success": true, ', "")
        plan = contract.verify_contract(doc, contract.build_anchors(SPEC))
        self.assertIn("resp:success", plan.summary())

    def test_valid_results_word_boundary(self):
        # "10" should not satisfy the required token "1"; "0_10" not satisfy "0_1"
        doc = "POST /api/v1/image-authenticity image_base64 image_format success result message 10 0_10 120"
        anchors = contract.build_anchors({k: v for k, v in SPEC.items() if k != "sample_count"})
        plan = contract.verify_contract(doc, anchors)
        self.assertEqual(plan.result, self.R.INCOMPLETE)
        self.assertTrue(any(f.key in ("result:1", "result:0_1", "result:0_2") for f in plan.missing))


if __name__ == "__main__":
    unittest.main()
