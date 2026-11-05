"""FIX 1 -- registry 'tag absent' vs 'registry query error'.

The defect: any non-zero exit from `docker manifest inspect` was read as "tag
absent", so an auth/network/5xx failure led to a blind push that overwrote a tag
we could not see. These tests pin the corrected, fail-closed classification.
"""
import unittest

from lib import _common
from lib.registry import classify_lookup

IMG = "registry.example.invalid/user/image:0.0.2"


class RegistryAbsentVsError(unittest.TestCase):
    def test_exit0_with_digest_is_present(self):
        out = '{"schemaVersion":2,"digest":"sha256:' + "a" * 64 + '"}'
        r = classify_lookup(IMG, exit_code=0, stdout=out)
        self.assertEqual(r.verdict, _common.RegistryVerdict.PRESENT)
        self.assertTrue(r.digest)
        self.assertFalse(r.safe_to_first_publish)
        self.assertTrue(r.requires_reconcile)

    def test_exit0_raw_digest_token_is_present(self):
        out = "Name:  " + IMG + "\nMediaType: ...\nDigest: sha256:" + "b" * 64
        r = classify_lookup(IMG, exit_code=0, stdout=out)
        self.assertEqual(r.verdict, _common.RegistryVerdict.PRESENT)
        self.assertEqual(r.digest, "sha256:" + "b" * 64)

    def test_exit0_without_digest_is_error_not_absent(self):
        r = classify_lookup(IMG, exit_code=0, stdout="ok")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertFalse(r.safe_to_first_publish)

    def test_notfound_is_uncertain(self):
        r = classify_lookup(IMG, exit_code=1, stderr="Error: manifest for " + IMG + " not found")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertFalse(r.safe_to_first_publish)

    def test_name_unknown_is_uncertain(self):
        r = classify_lookup(IMG, exit_code=1,
                            stderr='error parsing HTTP 404: unexpected status: 404 NAME_UNKNOWN: name unknown')
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)

    def test_repo_does_not_exist_is_uncertain(self):
        r = classify_lookup(IMG, exit_code=1, stderr="Error: repository does not exist")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)

    # --- the critical cases: a QUERY ERROR must NEVER be read as absent ---
    def test_auth_error_is_error(self):
        r = classify_lookup(IMG, exit_code=1,
                            stderr="unauthorized: authentication required")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "auth")
        self.assertFalse(r.safe_to_first_publish)
        self.assertTrue(r.requires_reconcile)

    def test_denied_error_is_error(self):
        r = classify_lookup(IMG, exit_code=1,
                            stderr="denied: requested access to the resource is denied")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "auth")
        self.assertFalse(r.safe_to_first_publish)

    def test_network_error_is_error(self):
        r = classify_lookup(IMG, exit_code=1,
                            stderr="dial tcp 1.2.3.4:443: connect: connection refused")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "network")
        self.assertFalse(r.safe_to_first_publish)

    def test_dns_error_is_error(self):
        r = classify_lookup(IMG, exit_code=1,
                            stderr="Get https://x/: dial tcp: lookup x on 1.1.1.1:53: no such host")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "network")
        self.assertFalse(r.safe_to_first_publish)

    def test_timeout_is_error(self):
        r = classify_lookup(IMG, exit_code=1, stderr="context deadline exceeded (Client.Timeout)")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "network")
        self.assertFalse(r.safe_to_first_publish)

    def test_server_5xx_is_error(self):
        r = classify_lookup(IMG, exit_code=1, stderr="server gave HTTP response with unexpected status: 503 Service Unavailable")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "network")
        self.assertFalse(r.safe_to_first_publish)

    def test_unrecognized_error_fails_closed_to_error(self):
        # No "not found", no auth/network signature -> must be ERROR, never ABSENT.
        r = classify_lookup(IMG, exit_code=1, stderr="boom: totally unhandled thing")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "other")
        self.assertFalse(r.safe_to_first_publish)

    def test_credentials_are_redacted_in_evidence(self):
        r = classify_lookup(IMG, exit_code=1,
                            stderr="unauthorized: authentication required token=SECRETTOKEN123")
        self.assertNotIn("SECRETTOKEN123", r.evidence)

    def test_nested_layer_digest_is_not_manifest_digest(self):
        out = '{"config":{"digest":"sha256:' + "a" * 64 + '"},"layers":[{"digest":"sha256:' + "b" * 64 + '"}]}'
        self.assertEqual(classify_lookup(IMG, exit_code=0, stdout=out).verdict, _common.RegistryVerdict.ERROR)

    def test_mixed_not_found_and_auth_is_error(self):
        r = classify_lookup(IMG, exit_code=1, stderr="404 not found: unauthorized")
        self.assertEqual(r.verdict, _common.RegistryVerdict.ERROR)
        self.assertEqual(r.error_kind, "auth")


if __name__ == "__main__":
    unittest.main()
