import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import handoff


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'code.py').touch()
        (self.root / 'weights.pth').touch()
        self.data = dict(candidate='baseline', code_path='code.py', weights_path='weights.pth',
                         license_status='unknown', interface={'input': 'image'},
                         evidence={'source': 'local'}, adaptation_gaps=['license'])

    def parse(self, data):
        return handoff.read_handoff(json.dumps(data), self.root)

    def test_plain_and_fenced_json(self):
        raw = json.dumps(self.data)
        self.assertEqual(self.parse(self.data), self.data)
        self.assertEqual(handoff.read_handoff('```json\n' + raw + '\n```', self.root), self.data)

    def test_malformed_and_duplicate(self):
        for raw in ('text\n{}', '```json\n{}\n```\n```json\n{}\n```', '{bad}',
                    '{"candidate":"a","candidate":"b"}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                handoff.read_handoff(raw, self.root)

    def test_missing_field_and_bad_paths(self):
        cases = [dict(self.data, candidate=''), dict(self.data, code_path='/etc/passwd'),
                 dict(self.data, weights_path='../outside'), dict(self.data, weights_path='missing')]
        (self.root / 'escape').symlink_to('/etc/passwd')
        cases.append(dict(self.data, weights_path='escape'))
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.parse(data)

    def test_unreviewed_entry_rejected(self):
        with self.assertRaisesRegex(ValueError, 'reviewed'):
            handoff.smoke(self.data, self.root)

    @patch('handoff.subprocess.Popen')
    @patch('handoff.socket.socket')
    def test_selected_weights_passed_to_service(self, socket, popen):
        socket.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
        proc = popen.return_value
        proc.poll.return_value = 1
        proc.stderr.read.return_value = b'failed to load'
        with self.assertRaisesRegex(RuntimeError, 'failed to load'):
            handoff.smoke(dict(self.data, code_path=handoff.ENTRY), self.root)
        self.assertEqual(popen.call_args.kwargs['env']['MODEL_PATH'], str(self.root / 'weights.pth'))
        proc.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
