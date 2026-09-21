import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from assemble import reviewed_tail


class TailReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        files={'capture.mkv':b'video','tail-frame.png':b'frame','000001.png':b'png',
               'events.jsonl':b'{"kind":"visited","index":1,"elapsed":10}',
               'summary.json':b'{"elapsed_seconds":15}'}
        for name,data in files.items():
            (self.folder/name).write_bytes(data)
        self.review={'decision':'confirmed_visits_present_unconfirmed_tail_missing',
                     'sha256':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()}}
        (self.folder/'tail-review.json').write_text(json.dumps(self.review),encoding='utf-8')
        self.report={'checks':{'video_duration_matches_run':False,'snapshots_contiguous':True},
                     'run_status':'blocked','media':{'format':{'duration':'10.5'}}}

    def test_exact_reviewed_tail_accepted(self):
        self.assertIsNotNone(reviewed_tail(self.folder,self.report))

    def test_other_failure_not_waived(self):
        self.report['checks']['snapshots_contiguous']=False
        self.assertIsNone(reviewed_tail(self.folder,self.report))

    def test_changed_video_rejected(self):
        (self.folder/'capture.mkv').write_bytes(b'different video')
        self.assertIsNone(reviewed_tail(self.folder,self.report))

    def test_missing_last_visit_rejected(self):
        self.report['media']['format']['duration']='9.9'
        self.assertIsNone(reviewed_tail(self.folder,self.report))
