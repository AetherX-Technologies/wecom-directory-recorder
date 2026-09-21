import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REQUIRED_FIELDS = ("user_code", "user_name", "title", "sso_dept_path")
PEOPLE_FIELDS = [
    "工号",
    "姓名",
    "职务列表",
    "部门数量",
    "任职关系数",
    "部门完整路径列表",
]
MEMBERSHIP_FIELDS = [
    "工号",
    "姓名",
    "职务",
    "部门完整路径",
    "来源",
    "证据",
    "合并来源数",
]


def _clean(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_membership(
    row: Mapping[str, Any], source_origin: str
) -> Tuple[Dict[str, Any], str]:
    normalized = {field: _clean(row.get(field, "")) for field in REQUIRED_FIELDS}
    missing = [field for field in REQUIRED_FIELDS if not normalized[field]]
    if missing:
        return {}, "missing_{}".format("_and_".join(missing))

    source_reference = _clean(row.get("source_reference", row.get("source_row", "")))
    normalized.update(
        {
            "source_origin": source_origin,
            "source_reference": source_reference,
            "source_count": 1,
        }
    )
    return normalized, ""


def _append_unique_reference(target: Dict[str, Any], reference: str) -> None:
    if not reference:
        return
    existing = [
        item.strip()
        for item in str(target.get("source_reference", "")).split(" | ")
        if item.strip()
    ]
    if reference not in existing:
        existing.append(reference)
    target["source_reference"] = " | ".join(existing)


def build_identity_outputs(
    matched_rows: Iterable[Mapping[str, Any]],
    override_rows: Iterable[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Build one-person and person-to-department tables using employee code.

    Repeated observations are collapsed only when employee code, name, title,
    and department path are all identical.  An override marked
    ``replace_user`` replaces every matched membership for that employee code;
    this is intended for a profile screen that displays the complete current
    department list.
    """

    matched_materialized = [dict(row) for row in matched_rows]
    override_materialized = [dict(row) for row in override_rows]
    replace_codes = {
        _clean(row.get("user_code", ""))
        for row in override_materialized
        if _clean(row.get("override_mode", "")).lower() == "replace_user"
        and _clean(row.get("user_code", ""))
    }

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for index, row in enumerate(matched_materialized, start=1):
        normalized, reason = _normalize_membership(row, "identity_match")
        if reason:
            rejected.append(
                {
                    "source_origin": "identity_match",
                    "source_index": index,
                    "reason": reason,
                    **row,
                }
            )
            continue
        if normalized["user_code"] not in replace_codes:
            accepted.append(normalized)

    for index, row in enumerate(override_materialized, start=1):
        normalized, reason = _normalize_membership(row, "manual_profile")
        if reason:
            rejected.append(
                {
                    "source_origin": "manual_profile",
                    "source_index": index,
                    "reason": reason,
                    **row,
                }
            )
            continue
        accepted.append(normalized)

    unique: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in accepted:
        key = tuple(str(row[field]) for field in REQUIRED_FIELDS)
        if key in unique:
            unique[key]["source_count"] = int(unique[key]["source_count"]) + 1
            _append_unique_reference(unique[key], str(row["source_reference"]))
        else:
            unique[key] = dict(row)

    names_by_code: Dict[str, set] = {}
    for row in unique.values():
        names_by_code.setdefault(str(row["user_code"]), set()).add(
            str(row["user_name"])
        )
    conflicts = {
        code: sorted(names) for code, names in names_by_code.items() if len(names) > 1
    }
    if conflicts:
        details = "; ".join(
            "{}={}".format(code, "|".join(names))
            for code, names in sorted(conflicts.items())
        )
        raise ValueError("Conflicting names for user_code: {}".format(details))

    memberships = sorted(
        unique.values(),
        key=lambda row: (
            str(row["user_code"]),
            str(row["sso_dept_path"]),
            str(row["title"]),
        ),
    )
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in memberships:
        grouped.setdefault(str(row["user_code"]), []).append(row)

    people = []
    for user_code, rows in sorted(grouped.items()):
        titles = sorted({str(row["title"]) for row in rows})
        departments = sorted({str(row["sso_dept_path"]) for row in rows})
        people.append(
            {
                "user_code": user_code,
                "user_name": str(rows[0]["user_name"]),
                "titles": " || ".join(titles),
                "department_count": len(departments),
                "membership_count": len(rows),
                "department_paths": " || ".join(departments),
            }
        )

    return {"people": people, "memberships": memberships, "rejected": rejected}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _write_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finalize_identity_outputs(
    matches_csv: Path,
    overrides_csv: Path,
    output_root: Path,
    review_csv: Path = None,
) -> Dict[str, Any]:
    matched_rows = _read_csv(matches_csv)
    override_rows = _read_csv(overrides_csv) if Path(overrides_csv).is_file() else []
    result = build_identity_outputs(matched_rows, override_rows)
    output_root = Path(output_root)
    people_path = output_root / "personnel_people_fixed.csv"
    memberships_path = output_root / "personnel_memberships_fixed.csv"
    rejected_path = output_root / "personnel_identity_rejected.csv"

    _write_csv(
        people_path,
        PEOPLE_FIELDS,
        [
            {
                "工号": row["user_code"],
                "姓名": row["user_name"],
                "职务列表": row["titles"],
                "部门数量": row["department_count"],
                "任职关系数": row["membership_count"],
                "部门完整路径列表": row["department_paths"],
            }
            for row in result["people"]
        ],
    )
    _write_csv(
        memberships_path,
        MEMBERSHIP_FIELDS,
        [
            {
                "工号": row["user_code"],
                "姓名": row["user_name"],
                "职务": row["title"],
                "部门完整路径": row["sso_dept_path"],
                "来源": row["source_origin"],
                "证据": row["source_reference"],
                "合并来源数": row["source_count"],
            }
            for row in result["memberships"]
        ],
    )

    rejected_fields = sorted({key for row in result["rejected"] for key in row.keys()})
    if rejected_fields:
        _write_csv(rejected_path, rejected_fields, result["rejected"])
    elif rejected_path.exists():
        rejected_path.unlink()

    review_count = 0
    review_output = output_root / "personnel_identity_review_fixed.csv"
    if review_csv is not None and Path(review_csv).is_file():
        review_rows = _read_csv(review_csv)
        review_count = len(review_rows)
        review_fields = list(review_rows[0].keys()) if review_rows else []
        if review_fields:
            _write_csv(review_output, review_fields, review_rows)

    duplicate_name_groups = 0
    codes_by_name: Dict[str, set] = {}
    for row in result["people"]:
        codes_by_name.setdefault(str(row["user_name"]), set()).add(
            str(row["user_code"])
        )
    duplicate_name_groups = sum(1 for codes in codes_by_name.values() if len(codes) > 1)
    manual_profile_codes = {
        str(row["user_code"])
        for row in result["memberships"]
        if row["source_origin"] == "manual_profile"
    }

    summary = {
        "identity_key": "user_code",
        "person_count": len(result["people"]),
        "membership_count": len(result["memberships"]),
        "people_with_multiple_departments": sum(
            1 for row in result["people"] if int(row["department_count"]) > 1
        ),
        "duplicate_name_groups_kept_separate_by_user_code": duplicate_name_groups,
        "manual_profile_override_person_count": len(manual_profile_codes),
        "rejected_match_row_count": len(result["rejected"]),
        "review_row_count": review_count,
        "people_csv": str(people_path.resolve()),
        "memberships_csv": str(memberships_path.resolve()),
        "review_csv": str(review_output.resolve()) if review_count else "",
        "matches_source": str(Path(matches_csv).resolve()),
        "overrides_source": str(Path(overrides_csv).resolve()),
    }
    summary_path = output_root / "personnel_identity_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build employee-code-based person and department membership CSVs."
    )
    parser.add_argument("--matches", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_identity_outputs(
        matches_csv=args.matches,
        overrides_csv=args.overrides,
        review_csv=args.review,
        output_root=args.output_root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
