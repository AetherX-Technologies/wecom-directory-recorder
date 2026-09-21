# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import unittest

from rapid_videocr.conflict_verifier import (
    parse_verification_response,
    resolve_visibility_results,
)


class ConflictVerifierTest(unittest.TestCase):
    def test_parses_fenced_response(self):
        parsed = parse_verification_response(
            '```json\n{"candidate_a_visible":true,"candidate_b_visible":false,"visible_names":["测试人员丁"]}\n```'
        )

        self.assertTrue(parsed["candidate_a_visible"])
        self.assertFalse(parsed["candidate_b_visible"])
        self.assertEqual(parsed["visible_names"], ["测试人员丁"])

    def test_resolves_only_one_visible_candidate(self):
        resolution = resolve_visibility_results(
            [
                {"candidate_a_visible": True, "candidate_b_visible": False},
                {"candidate_a_visible": True, "candidate_b_visible": False},
            ]
        )

        self.assertEqual(resolution, "merge_b_into_a")

    def test_keeps_both_when_each_has_evidence(self):
        resolution = resolve_visibility_results(
            [
                {"candidate_a_visible": True, "candidate_b_visible": False},
                {"candidate_a_visible": False, "candidate_b_visible": True},
            ]
        )

        self.assertEqual(resolution, "keep_both")


if __name__ == "__main__":
    unittest.main()
