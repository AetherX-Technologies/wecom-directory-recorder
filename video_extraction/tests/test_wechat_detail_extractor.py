# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from rapid_videocr.wechat_detail_extractor import (
    analyze_selected_detail_frames,
    build_stable_candidates,
    build_wechat_outputs,
    compose_wechat_detail_roi,
    group_change_events,
    parse_wechat_detail_batch_response,
)


class WeChatDetailSelectionTest(unittest.TestCase):
    def test_crops_only_right_detail_panel(self):
        frame = np.zeros((794, 984, 3), dtype=np.uint8)
        roi = compose_wechat_detail_roi(frame)

        self.assertGreater(roi.shape[0], 600)
        self.assertGreater(roi.shape[1], 400)
        self.assertLess(roi.shape[1], 500)

    def test_groups_loading_changes_then_selects_stable_midpoints(self):
        events = [
            {"source_frame_index": 4, "timestamp_seconds": 1.0},
            {"source_frame_index": 5, "timestamp_seconds": 1.25},
            {"source_frame_index": 16, "timestamp_seconds": 4.0},
        ]
        groups = group_change_events(events, group_gap_seconds=1.0)
        candidates = build_stable_candidates(
            groups,
            decoded_frame_count=28,
            source_fps=4.0,
            min_stable_seconds=0.75,
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(
            [row["source_frame_index"] for row in candidates], [2, 10, 22]
        )


class WeChatDetailResponseTest(unittest.TestCase):
    def test_parses_multiple_images_and_preserves_all_departments(self):
        parsed = parse_wechat_detail_batch_response(
            '{"frames":['
            '{"image_index":1,"page_type":"folder","name":"","position":"","user_code":"","departments":[]},'
            '{"image_index":0,"page_type":"person","name":"测试人1","position":"党委副书记、工会主席、职工董事","user_code":"08000","departments":["示例集团 / 公司领导","示例集团技术中心/技术中心领导"]}'
            "]}",
            expected_count=2,
        )

        self.assertEqual(parsed[0]["user_code"], "08000")
        self.assertEqual(
            parsed[0]["departments"],
            ["示例集团/公司领导", "示例集团技术中心/技术中心领导"],
        )
        self.assertEqual(parsed[1]["page_type"], "folder")

    def test_accepts_first_complete_object_when_model_appends_extra_json(self):
        parsed = parse_wechat_detail_batch_response(
            '{"frames":[{"image_index":0,"page_type":"person","name":"测试人9","position":"","user_code":"DEMO102","departments":["示例集团/项目管理部"]}]} '
            '{"note":"duplicate trailing object"}',
            expected_count=1,
        )

        self.assertEqual(parsed[0]["name"], "测试人9")
        self.assertEqual(parsed[0]["user_code"], "DEMO102")

    def test_resume_cache_keeps_equal_images_at_two_selection_indexes(self):
        class FakeClient:
            def __init__(self):
                self.calls = 0

            def analyze_images(self, images, prompt, max_tokens):
                self.calls += 1
                frames = [
                    {
                        "image_index": index,
                        "page_type": "folder",
                        "name": "",
                        "position": "",
                        "user_code": "",
                        "departments": [],
                    }
                    for index in range(len(images))
                ]
                return SimpleNamespace(
                    content=__import__("json").dumps({"frames": frames}),
                    raw={"choices": []},
                )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "frames").mkdir()
            ok, encoded = cv2.imencode(
                ".jpg", np.zeros((16, 16, 3), dtype=np.uint8)
            )
            self.assertTrue(ok)
            for name in ("one.jpg", "two.jpg"):
                (root / "frames" / name).write_bytes(encoded.tobytes())
            selected = [
                {
                    "selection_index": index,
                    "source_frame_index": index,
                    "timestamp_seconds": float(index),
                    "timestamp": "00:00:0{}.000".format(index),
                    "frame": "frames/{}".format(name),
                    "frame_sha256": "same-image-hash",
                }
                for index, name in enumerate(("one.jpg", "two.jpg"))
            ]
            first_client = FakeClient()
            analyzed, _ = analyze_selected_detail_frames(
                root, selected, first_client, workers=1, batch_size=2
            )
            second_client = FakeClient()
            resumed, _ = analyze_selected_detail_frames(
                root, selected, second_client, workers=1, batch_size=2
            )

        self.assertEqual([row["selection_index"] for row in analyzed], [0, 1])
        self.assertEqual([row["selection_index"] for row in resumed], [0, 1])
        self.assertEqual(first_client.calls, 1)
        self.assertEqual(second_client.calls, 0)


class WeChatDetailFinalizerTest(unittest.TestCase):
    def _frame(self, code, name, position, departments, timestamp, frame):
        return {
            "timestamp_seconds": timestamp,
            "timestamp": "00:00:{:02d}.000".format(timestamp),
            "frame": frame,
            "frame_sha256": frame,
            "parsed": {
                "page_type": "person",
                "user_code": code,
                "name": name,
                "position": position,
                "departments": departments,
            },
        }

    def test_uses_employee_code_and_keeps_multiple_memberships(self):
        frames = [
            self._frame(
                "08000",
                "测试人1",
                "党委副书记、工会主席、职工董事",
                ["示例集团/公司领导", "示例集团技术中心/技术中心领导"],
                1,
                "one.jpg",
            ),
            self._frame(
                "DEMO999",
                "测试人1",
                "专家",
                ["示例集团/技术中心"],
                2,
                "two.jpg",
            ),
        ]
        outputs = build_wechat_outputs("capture.mkv", frames)

        self.assertEqual(len(outputs["people"]), 2)
        self.assertEqual(len(outputs["memberships"]), 3)
        first_memberships = [
            row
            for row in outputs["memberships"]
            if row["user_code"] == "08000"
        ]
        self.assertEqual(len(first_memberships), 2)

    def test_keeps_blank_position_out_of_filtered_people_table(self):
        outputs = build_wechat_outputs(
            "capture.mkv",
            [
                self._frame(
                    "DEMO009",
                    "测试人员甲",
                    "",
                    ["示例集团/外部董事"],
                    1,
                    "one.jpg",
                )
            ],
        )

        self.assertEqual(len(outputs["people"]), 1)
        self.assertEqual(outputs["people_with_positions"], [])

    def test_routes_missing_code_to_review(self):
        outputs = build_wechat_outputs(
            "capture.mkv",
            [
                self._frame(
                    "",
                    "测试人6",
                    "主任",
                    ["示例集团/办公室"],
                    1,
                    "one.jpg",
                )
            ],
        )

        self.assertEqual(outputs["people"], [])
        self.assertEqual(outputs["review"][0]["reason"], "missing_identity")

    def test_keeps_position_without_code_in_all_positions_output(self):
        outputs = build_wechat_outputs(
            "capture.mkv",
            [
                self._frame(
                    "",
                    "测试人员乙",
                    "经理",
                    ["示例集团/船舶设计与建造事业部"],
                    1,
                    "one.jpg",
                )
            ],
        )

        self.assertEqual(len(outputs["position_records"]), 1)
        self.assertEqual(
            outputs["position_records"][0]["identity_status"],
            "missing_user_code",
        )

    def test_excludes_folder_people_count_from_all_positions_output(self):
        outputs = build_wechat_outputs(
            "capture.mkv",
            [
                self._frame(
                    "",
                    "项目领导班",
                    "~0人",
                    ["示例集团/项目部"],
                    1,
                    "one.jpg",
                )
            ],
        )

        self.assertEqual(outputs["position_records"], [])

    def test_reviewed_override_resolves_model_spelling_conflict(self):
        frames = [
            self._frame(
                "08001",
                "测试错字丙",
                "副总工程师",
                ["示例集团/水利院"],
                1,
                "one.jpg",
            ),
            self._frame(
                "08001",
                "测试人5",
                "副总工程师",
                ["示例集团/水利院"],
                2,
                "two.jpg",
            ),
        ]
        outputs = build_wechat_outputs(
            "capture.mkv",
            frames,
            overrides=[
                {
                    "user_code": "08001",
                    "name": "测试人5",
                    "position": "",
                    "resolution_note": "reviewed screenshot",
                    "evidence": "one.jpg | two.jpg",
                }
            ],
        )

        self.assertEqual(outputs["people"][0]["name"], "测试人5")
        self.assertEqual(outputs["people"][0]["review_status"], "ok")
        self.assertEqual(outputs["people"][0]["override_applied"], "yes")
        self.assertEqual(outputs["review"], [])

    def test_rejects_folder_count_as_employee_code(self):
        outputs = build_wechat_outputs(
            "capture.mkv",
            [
                self._frame(
                    "~0人",
                    "项目领导班",
                    "",
                    ["示例集团/项目部"],
                    1,
                    "one.jpg",
                )
            ],
        )

        self.assertEqual(outputs["people"], [])
        self.assertEqual(outputs["review"][0]["reason"], "invalid_user_code")


if __name__ == "__main__":
    unittest.main()
