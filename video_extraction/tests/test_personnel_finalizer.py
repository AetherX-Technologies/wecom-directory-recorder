# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import unittest
from unittest.mock import patch

from rapid_videocr.personnel_finalizer import apply_manual_visual_corrections


class PersonnelFinalizerTest(unittest.TestCase):
    def setUp(self):
        corrections = (("示例集团 > 生产经营单位 > 城建与交通工程院 > 工程管理部", "副主任", "测试人3", "测试人4"),)
        self.names = patch('rapid_videocr.personnel_finalizer.MANUAL_NAME_CORRECTIONS', corrections)
        self.departments = patch('rapid_videocr.personnel_finalizer.DEPARTMENT_SEGMENT_CORRECTIONS', (("电与抽水蓄能工程院", "水电与抽水蓄能工程院"),))
        self.names.start()
        self.departments.start()
        self.addCleanup(self.names.stop)
        self.addCleanup(self.departments.stop)

    def test_applies_exact_name_correction_and_preserves_real_people(self):
        base = {
            "department_path": "示例集团 > 生产经营单位 > 城建与交通工程院 > 工程管理部",
            "position": "副主任",
            "video": "one.mp4",
            "timestamp_seconds": 1.0,
            "timestamp": "00:00:01.000",
            "source_frame": "one.jpg",
            "evidence_count": 1,
        }
        rows = apply_manual_visual_corrections(
            [{**base, "name": "测试人3"}, {**base, "name": "测试人4", "evidence_count": 4}]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "测试人4")
        self.assertEqual(rows[0]["evidence_count"], 5)

    def test_corrects_and_collapses_repeated_department_segment(self):
        rows = apply_manual_visual_corrections(
            [
                {
                    "department_path": "示例集团 > 生产经营单位 > 水电与抽水蓄能工程院 > 电与抽水蓄能工程院",
                    "name": "测试人6",
                    "position": "主任",
                    "video": "one.mp4",
                    "timestamp_seconds": 1.0,
                    "timestamp": "00:00:01.000",
                    "source_frame": "one.jpg",
                    "evidence_count": 2,
                }
            ]
        )

        self.assertEqual(
            rows[0]["department_path"],
            "示例集团 > 生产经营单位 > 水电与抽水蓄能工程院",
        )

    def test_only_replaces_complete_department_segments(self):
        rows = apply_manual_visual_corrections(
            [
                {
                    "department_path": "示例集团 > 驻外经营机构 > 南方总部/示例企业南方有限公司",
                    "name": "测试人6",
                    "position": "主任",
                    "video": "one.mp4",
                    "timestamp_seconds": 1.0,
                    "timestamp": "00:00:01.000",
                    "source_frame": "one.jpg",
                    "evidence_count": 1,
                }
            ]
        )

        self.assertEqual(
            rows[0]["department_path"],
            "示例集团 > 驻外经营机构 > 南方总部/示例企业南方有限公司",
        )


if __name__ == "__main__":
    unittest.main()
