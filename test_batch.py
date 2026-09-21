import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import batch_run


class BatchSafetyTests(unittest.TestCase):
    def check_terminal_status(self,prior_status,expected_state,expected_code):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            previous=root/'prior'
            previous.mkdir()
            (previous/'summary.json').write_text(json.dumps({'status':prior_status,'reason':'test'}),encoding='utf-8')
            args=SimpleNamespace(after=str(previous),max_hours=1,max_segments=1,segment_rows=5)
            with patch.object(batch_run,'BASE',root),patch.object(batch_run,'audit',return_value=0),patch.object(batch_run.subprocess,'run') as run:
                self.assertEqual(batch_run.main(args),expected_code)
                run.assert_not_called()
                result=json.loads((root/'batch-status.json').read_text(encoding='utf-8'))
                self.assertEqual(result['state'],expected_state)

    def test_user_stop_is_not_automatically_resumed(self):
        self.check_terminal_status('stopped','blocked',1)

    def test_failed_capture_is_not_retried(self):
        self.check_terminal_status('recording_error','blocked',1)

    def test_end_candidate_is_not_claimed_complete(self):
        self.check_terminal_status('end_candidate','end_candidate',0)

    def test_audit_failure_prevents_new_capture(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp)
            previous=root/'prior'; previous.mkdir()
            (previous/'summary.json').write_text('{"status":"limit_reached"}',encoding='utf-8')
            args=SimpleNamespace(after=str(previous),max_hours=1,max_segments=1,segment_rows=5)
            with patch.object(batch_run,'BASE',root),patch.object(batch_run,'audit',return_value=1),patch.object(batch_run.subprocess,'run') as run:
                self.assertEqual(batch_run.main(args),1)
                run.assert_not_called()


if __name__=='__main__':
    unittest.main()
