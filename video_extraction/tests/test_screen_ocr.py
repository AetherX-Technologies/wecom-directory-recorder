# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import unittest
import re
from unittest.mock import patch

import numpy as np

from rapid_videocr.screen_ocr import (
    RegionChangeDetector,
    format_timestamp,
    normalize_ocr_line,
    unique_text_lines,
)
from rapid_videocr.screen_ocr_all_frames import (
    build_text_consensus,
    choose_sharpest_frame,
    clean_watermark_text,
)


class RegionChangeDetectorTest(unittest.TestCase):
    def test_selects_only_regions_that_changed(self):
        detector = RegionChangeDetector(split_ratio=0.25, threshold=1.5)
        first = np.zeros((40, 80, 3), dtype=np.uint8)

        self.assertEqual(set(detector.changed_regions(first)), {"left", "main"})
        self.assertEqual(detector.changed_regions(first), {})

        main_changed = first.copy()
        main_changed[:, 20:] = 255
        self.assertEqual(set(detector.changed_regions(main_changed)), {"main"})

        left_changed = main_changed.copy()
        left_changed[:, :20] = 255
        self.assertEqual(set(detector.changed_regions(left_changed)), {"left"})


class TextOutputTest(unittest.TestCase):
    def test_normalizes_html_and_location_tokens(self):
        self.assertEqual(
            normalize_ocr_line("  &lt;部门<|LOC_432|>   名称  "),
            "<部门 名称",
        )

    def test_keeps_first_occurrence_of_each_line(self):
        self.assertEqual(
            unique_text_lines(["办公室\n市场开发部", "办公室\n财务资金部"]),
            ["办公室", "市场开发部", "财务资金部"],
        )

    def test_formats_timestamp(self):
        self.assertEqual(format_timestamp(62.345), "00:01:02.345")

    def test_removes_raster_watermark_from_ocr_text(self):
        with patch('rapid_videocr.screen_ocr_all_frames.WATERMARK_RE', re.compile(r'测试水印@示例企业')), patch('rapid_videocr.screen_ocr_all_frames.WATERMARK_DATE_RE', re.compile(r'2099-1-1')):
            self.assertEqual(clean_watermark_text("公司领导 测试水印@示例企业 2099-1-1"), "公司领导")

    def test_consensus_rejects_unique_gibberish(self):
        records = [
            {"timestamp": "00:00:00.000", "text": "办公室\nASWnucleus ནིཧ"},
            {"timestamp": "00:00:00.250", "text": "办公室\n市场开发部"},
        ]

        _, accepted = build_text_consensus(records)

        self.assertIn("办公室", accepted)
        self.assertIn("市场开发部", accepted)
        self.assertNotIn("ASWnucleus ནིཧ", accepted)

    def test_selects_sharpest_frame_from_dynamic_window(self):
        blurred = np.full((40, 80, 3), 127, dtype=np.uint8)
        sharp = blurred.copy()
        sharp[:, 40::2] = 255
        frames = [(0, 0.0, blurred), (1, 0.1, sharp)]

        selected = choose_sharpest_frame(frames, 20, "main")

        self.assertEqual(selected[0], 1)


if __name__ == "__main__":
    unittest.main()
