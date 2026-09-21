# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rapid_videocr.personnel_extractor import (
    build_personnel_rows,
    compose_personnel_roi,
    deduplicate_personnel_rows,
    parse_personnel_batch_response,
    parse_personnel_response,
)


class PersonnelResponseTest(unittest.TestCase):
    def test_parses_fenced_json_and_drops_people_without_position(self):
        parsed = parse_personnel_response(
            """```json
{"page_type":"people","page_title":"党委工作部","breadcrumb":[],"people":[{"name":"测试人17","position":"团委副书记、副主任"},{"name":"测试人12","position":""}]}
```"""
        )

        self.assertEqual(parsed["page_title"], "党委工作部")
        self.assertEqual(
            parsed["people"],
            [{"name": "测试人17", "position": "团委副书记、副主任"}],
        )

    def test_builds_hierarchy_from_preceding_folder_frame(self):
        frames = [
            {
                "timestamp_seconds": 1.0,
                "timestamp": "00:00:01.000",
                "frame": "frames/folder.jpg",
                "parsed": {
                    "page_type": "folder",
                    "page_title": "职能部门",
                    "breadcrumb": ["企业通讯录", "示例集团", "职能部门"],
                    "people": [],
                },
            },
            {
                "timestamp_seconds": 2.0,
                "timestamp": "00:00:02.000",
                "frame": "frames/people.jpg",
                "parsed": {
                    "page_type": "people",
                    "page_title": "党委工作部",
                    "breadcrumb": [],
                    "people": [{"name": "测试人17", "position": "副主任"}],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows = build_personnel_rows(
                "recording.mp4", Path(temporary_directory), frames
            )

        self.assertEqual(
            rows[0]["department_path"], "示例集团 > 职能部门 > 党委工作部"
        )

    def test_backfills_common_root_for_people_at_video_start(self):
        frames = [
            {
                "timestamp_seconds": 0.0,
                "timestamp": "00:00:00.000",
                "frame": "frames/people.jpg",
                "parsed": {
                    "page_type": "people",
                    "page_title": "公司领导",
                    "breadcrumb": ["错误的人员页层级"],
                    "people": [{"name": "测试人16", "position": "总经理"}],
                },
            },
            {
                "timestamp_seconds": 1.0,
                "timestamp": "00:00:01.000",
                "frame": "frames/folder.jpg",
                "parsed": {
                    "page_type": "folder",
                    "page_title": "示例集团",
                    "breadcrumb": ["企业通讯录", "示例集团"],
                    "people": [],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows = build_personnel_rows(
                "recording.mp4", Path(temporary_directory), frames
            )

        self.assertEqual(rows[0]["department_path"], "示例集团 > 公司领导")

    def test_parses_batch_by_image_index(self):
        parsed = parse_personnel_batch_response(
            '{"frames":[{"image_index":1,"page_type":"folder","page_title":"职能部门","breadcrumb":["示例集团"],"people":[]},{"image_index":0,"page_type":"people","page_title":"公司领导","breadcrumb":[],"people":[{"name":"测试人16","position":"总经理"}]}]}',
            expected_count=2,
        )

        self.assertEqual(parsed[0]["people"][0]["name"], "测试人16")
        self.assertEqual(parsed[1]["breadcrumb"], ["示例集团"])

    def test_ignores_incomplete_folder_breadcrumb(self):
        frames = [
            {
                "timestamp_seconds": 1.0,
                "timestamp": "00:00:01.000",
                "frame": "frames/root.jpg",
                "parsed": {
                    "page_type": "folder",
                    "page_title": "职能部门",
                    "breadcrumb": ["企业通讯录", "示例集团", "职能部门"],
                    "people": [],
                },
            },
            {
                "timestamp_seconds": 2.0,
                "timestamp": "00:00:02.000",
                "frame": "frames/partial.jpg",
                "parsed": {
                    "page_type": "folder",
                    "page_title": "职能部门",
                    "breadcrumb": ["职能部门"],
                    "people": [],
                },
            },
            {
                "timestamp_seconds": 3.0,
                "timestamp": "00:00:03.000",
                "frame": "frames/people.jpg",
                "parsed": {
                    "page_type": "people",
                    "page_title": "办公室",
                    "breadcrumb": [],
                    "people": [{"name": "测试人6", "position": "主任"}],
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            rows = build_personnel_rows(
                "recording.mp4", Path(temporary_directory), frames
            )

        self.assertEqual(
            rows[0]["department_path"], "示例集团 > 职能部门 > 办公室"
        )

    def test_deduplicates_by_department_name_and_position(self):
        source = {
            "department_path": "示例集团 > 公司领导",
            "name": "测试人16",
            "position": "党委副书记、董事、总经理",
            "video": "one.mp4",
            "timestamp_seconds": 1.0,
            "timestamp": "00:00:01.000",
            "source_frame": "frame.jpg",
        }
        rows = deduplicate_personnel_rows([source, source])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evidence_count"], 2)

    def test_sums_existing_evidence_counts_when_merging_aggregated_rows(self):
        source = {
            "department_path": "示例集团 > 公司领导",
            "name": "测试人16",
            "position": "党委副书记、董事、总经理",
            "video": "one.mp4",
            "timestamp_seconds": 1.0,
            "timestamp": "00:00:01.000",
            "source_frame": "frame.jpg",
        }
        rows = deduplicate_personnel_rows(
            [
                {**source, "evidence_count": 4},
                {**source, "evidence_count": 3},
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evidence_count"], 7)

    def test_normalizes_rare_one_character_root_alias(self):
        common = {
            "department_path": "示例集团 > 公司领导",
            "name": "测试人16",
            "position": "总经理",
            "video": "one.mp4",
            "timestamp_seconds": 1.0,
            "timestamp": "00:00:01.000",
            "source_frame": "one.jpg",
        }
        alias = {
            **common,
            "department_path": "示例集困 > 公司领导",
            "source_frame": "two.jpg",
        }
        rows = deduplicate_personnel_rows([common] * 10 + [alias])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["department_path"], "示例集团 > 公司领导")
        self.assertEqual(rows[0]["evidence_count"], 11)

    def test_merges_truncated_position_into_longer_position(self):
        base = {
            "department_path": "示例集团 > 办公室",
            "name": "测试人6",
            "video": "one.mp4",
            "timestamp_seconds": 1.0,
            "timestamp": "00:00:01.000",
            "source_frame": "one.jpg",
        }
        rows = deduplicate_personnel_rows(
            [
                {**base, "position": "四级管理专"},
                {**base, "position": "四级管理专家、主管"},
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["position"], "四级管理专家、主管")
        self.assertEqual(rows[0]["evidence_count"], 2)

    def test_merges_position_punctuation_variant(self):
        base = {
            "department_path": "示例集团 > 海外区域部",
            "name": "测试人13",
            "video": "one.mp4",
            "timestamp_seconds": 1.0,
            "timestamp": "00:00:01.000",
            "source_frame": "one.jpg",
        }
        rows = deduplicate_personnel_rows(
            [
                {**base, "position": "海外部副总经理 亚太区域部总经理"},
                {**base, "position": "海外部副总经理，亚太区域部总经理"},
            ]
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["position"], "海外部副总经理，亚太区域部总经理"
        )
        self.assertEqual(rows[0]["evidence_count"], 2)


class PersonnelROITest(unittest.TestCase):
    def test_composes_fixed_header_and_list_roi(self):
        frame = np.zeros((1668, 2388, 3), dtype=np.uint8)
        roi = compose_personnel_roi(frame)

        self.assertEqual(roi.shape[1], 900)
        self.assertGreater(roi.shape[0], 1400)
        self.assertLess(roi.shape[0], 1700)


if __name__ == "__main__":
    unittest.main()
