# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
import unittest

from rapid_videocr.personnel_identity_finalizer import build_identity_outputs


class PersonnelIdentityFinalizerTest(unittest.TestCase):
    def test_one_person_can_have_multiple_department_memberships(self):
        matched = [
            {
                "user_code": "DEMO103",
                "user_name": "测试人20",
                "title": "副总经理",
                "sso_dept_path": "旧路径",
            }
        ]
        overrides = [
            {
                "user_code": "DEMO103",
                "user_name": "测试人20",
                "title": "副总经理",
                "sso_dept_path": "部门甲",
                "override_mode": "replace_user",
            },
            {
                "user_code": "DEMO103",
                "user_name": "测试人20",
                "title": "副总经理",
                "sso_dept_path": "部门乙",
                "override_mode": "replace_user",
            },
        ]

        result = build_identity_outputs(matched, overrides)

        self.assertEqual(len(result["people"]), 1)
        self.assertEqual(result["people"][0]["user_code"], "DEMO103")
        self.assertEqual(result["people"][0]["department_count"], 2)
        self.assertEqual(
            {row["sso_dept_path"] for row in result["memberships"]},
            {"部门甲", "部门乙"},
        )

    def test_same_name_with_different_user_codes_is_not_merged(self):
        matched = [
            {
                "user_code": "001",
                "user_name": "测试人7",
                "title": "主任",
                "sso_dept_path": "部门甲",
            },
            {
                "user_code": "002",
                "user_name": "测试人7",
                "title": "主任",
                "sso_dept_path": "部门乙",
            },
        ]

        result = build_identity_outputs(matched, [])

        self.assertEqual(len(result["people"]), 2)
        self.assertEqual({row["user_code"] for row in result["people"]}, {"001", "002"})

    def test_repeated_observation_is_collapsed_without_losing_source_count(self):
        repeated = {
            "user_code": "001",
            "user_name": "测试人6",
            "title": "主任",
            "sso_dept_path": "部门甲",
        }

        result = build_identity_outputs([repeated, repeated], [])

        self.assertEqual(len(result["memberships"]), 1)
        self.assertEqual(result["memberships"][0]["source_count"], 2)

    def test_conflicting_names_for_one_user_code_fail_closed(self):
        matched = [
            {
                "user_code": "001",
                "user_name": "测试人6",
                "title": "主任",
                "sso_dept_path": "部门甲",
            },
            {
                "user_code": "001",
                "user_name": "测试人11",
                "title": "主任",
                "sso_dept_path": "部门乙",
            },
        ]

        with self.assertRaisesRegex(ValueError, "001"):
            build_identity_outputs(matched, [])

    def test_missing_identity_fields_are_rejected(self):
        result = build_identity_outputs(
            [
                {
                    "user_code": "",
                    "user_name": "待核对",
                    "title": "主任",
                    "sso_dept_path": "部门甲",
                }
            ],
            [],
        )

        self.assertEqual(result["people"], [])
        self.assertEqual(result["memberships"], [])
        self.assertEqual(len(result["rejected"]), 1)


if __name__ == "__main__":
    unittest.main()
