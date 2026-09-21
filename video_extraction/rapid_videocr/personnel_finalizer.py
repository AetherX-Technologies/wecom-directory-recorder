# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .conflict_verifier import apply_name_resolutions
from .personnel_extractor import (
    deduplicate_personnel_rows,
    find_review_conflicts,
    write_personnel_csv,
    write_review_conflicts,
)


# Public distribution: populate only with your own reviewed evidence.
MANUAL_NAME_CORRECTIONS = ()
DEPARTMENT_SEGMENT_CORRECTIONS = ()
DEPARTMENT_EVIDENCE_FRAMES = {}


def apply_manual_visual_corrections(
    rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    name_mappings = {
        (department_path, position, original): corrected
        for department_path, position, original, corrected in MANUAL_NAME_CORRECTIONS
    }
    segment_mappings = dict(DEPARTMENT_SEGMENT_CORRECTIONS)
    corrected_rows = []
    for source_row in rows:
        row = dict(source_row)
        name_key = (
            str(row["department_path"]),
            str(row["position"]),
            str(row["name"]),
        )
        row["name"] = name_mappings.get(name_key, str(row["name"]))

        path_parts = [
            segment_mappings.get(part.strip(), part.strip())
            for part in str(row["department_path"]).split(">")
            if part.strip()
        ]
        collapsed_parts = []
        for part in path_parts:
            if not collapsed_parts or collapsed_parts[-1] != part:
                collapsed_parts.append(part)
        row["department_path"] = " > ".join(collapsed_parts)
        corrected_rows.append(row)
    return deduplicate_personnel_rows(corrected_rows)


def build_manual_resolution_audit(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    audit = []
    for department_path, position, original, corrected in MANUAL_NAME_CORRECTIONS:
        affected = [
            row
            for row in rows
            if str(row["department_path"]) == department_path
            and str(row["position"]) == position
            and str(row["name"]) == original
        ]
        audit.append(
            {
                "correction_type": "name",
                "scope": department_path,
                "position": position,
                "original": original,
                "corrected": corrected,
                "affected_csv_rows": len(affected),
                "affected_evidence_count": sum(
                    int(row.get("evidence_count", 1)) for row in affected
                ),
                "source_frame": (
                    str(affected[0].get("source_frame", "")) if affected else ""
                ),
                "basis": "manual_visual_confirmation",
            }
        )

    for original, corrected in DEPARTMENT_SEGMENT_CORRECTIONS:
        affected = [
            row
            for row in rows
            if original
            in [part.strip() for part in str(row["department_path"]).split(">")]
        ]
        audit.append(
            {
                "correction_type": "department_segment",
                "scope": "all_department_paths",
                "position": "",
                "original": original,
                "corrected": corrected,
                "affected_csv_rows": len(affected),
                "affected_evidence_count": sum(
                    int(row.get("evidence_count", 1)) for row in affected
                ),
                "source_frame": (
                    str(affected[0].get("source_frame", "")) if affected else ""
                ),
                "basis": "manual_visual_confirmation",
            }
        )
    return audit


def finalize_personnel_outputs(output_root: Path) -> Dict[str, Any]:
    output_root = Path(output_root)
    summary_path = output_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    automated_resolutions = _read_csv(output_root / "conflict_resolutions.csv")

    all_rows_before_manual = []
    all_final_rows = []
    video_summaries = []
    accepted_single_frame_analyses = 0
    ignored_non_single_records = 0
    for video_summary in summary.get("videos", []):
        video_stem = Path(str(video_summary["video"])).stem
        video_output = output_root / video_stem
        video_csv = video_output / "personnel_positions.csv"
        prefinalization_csv = video_output / "personnel_positions_prefinalization.csv"
        source_rows = _read_csv(
            prefinalization_csv if prefinalization_csv.is_file() else video_csv
        )
        if not prefinalization_csv.is_file():
            write_personnel_csv(prefinalization_csv, source_rows)
        automatically_resolved = apply_name_resolutions(
            source_rows, automated_resolutions
        )
        all_rows_before_manual.extend(automatically_resolved)
        finalized_rows = apply_manual_visual_corrections(automatically_resolved)
        write_personnel_csv(video_csv, finalized_rows)
        all_final_rows.extend(finalized_rows)

        child_summary_path = video_output / "summary.json"
        child_summary = json.loads(
            child_summary_path.read_text(encoding="utf-8-sig")
        )
        request_mode_counts = _count_request_modes(video_output / "raw_model.jsonl")
        accepted_count = request_mode_counts.get("single", 0)
        ignored_count = sum(request_mode_counts.values()) - accepted_count
        if accepted_count != int(child_summary["analyzed_frame_count"]):
            raise ValueError(
                "Accepted single-frame count does not match analyzed frames for {}"
                .format(video_stem)
            )
        child_summary["unique_personnel_count"] = len(finalized_rows)
        child_summary["accepted_single_frame_analysis_count"] = accepted_count
        child_summary["ignored_non_single_raw_record_count"] = ignored_count
        child_summary["postprocessing_complete"] = True
        child_summary_path.write_text(
            json.dumps(child_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        video_summaries.append(child_summary)
        accepted_single_frame_analyses += accepted_count
        ignored_non_single_records += ignored_count

    final_rows = deduplicate_personnel_rows(all_final_rows)
    detailed_csv = output_root / "personnel_positions.csv"
    simple_csv = output_root / "personnel_positions_simple.csv"
    write_personnel_csv(detailed_csv, final_rows)
    _write_simple_csv(simple_csv, final_rows)

    conflicts = find_review_conflicts(final_rows)
    conflict_path = output_root / "review_conflicts.csv"
    write_review_conflicts(conflict_path, conflicts)

    audit = build_manual_resolution_audit(all_rows_before_manual)
    for record in audit:
        if record["correction_type"] != "department_segment":
            continue
        relative_frame = DEPARTMENT_EVIDENCE_FRAMES.get(str(record["original"]))
        if relative_frame:
            record["source_frame"] = str((output_root / relative_frame).resolve())
    audit_path = output_root / "manual_visual_resolutions.csv"
    _write_audit_csv(audit_path, audit)

    summary["videos"] = video_summaries
    summary["unique_personnel_count"] = len(final_rows)
    summary["review_conflict_count"] = len(conflicts)
    summary["csv"] = str(detailed_csv.resolve())
    summary["simple_csv"] = str(simple_csv.resolve())
    summary["review_conflicts_csv"] = str(conflict_path.resolve())
    summary["manual_visual_resolutions_csv"] = str(audit_path.resolve())
    summary["manual_name_correction_rule_count"] = len(
        MANUAL_NAME_CORRECTIONS
    )
    summary["manual_name_affected_row_count"] = sum(
        int(record["affected_csv_rows"])
        for record in audit
        if record["correction_type"] == "name"
    )
    summary["manual_department_correction_rule_count"] = len(
        DEPARTMENT_SEGMENT_CORRECTIONS
    )
    summary["manual_department_affected_row_count"] = len(
        {
            (
                str(row["department_path"]),
                str(row["name"]),
                str(row["position"]),
            )
            for row in all_rows_before_manual
            if any(
                original
                in [
                    part.strip()
                    for part in str(row["department_path"]).split(">")
                ]
                for original, _ in DEPARTMENT_SEGMENT_CORRECTIONS
            )
        }
    )
    summary["manual_department_rule_application_count"] = sum(
        int(record["affected_csv_rows"])
        for record in audit
        if record["correction_type"] == "department_segment"
    )
    summary["accepted_single_frame_analysis_count"] = (
        accepted_single_frame_analyses
    )
    summary["ignored_non_single_raw_record_count"] = ignored_non_single_records
    summary["accepted_request_mode"] = "single"
    summary["postprocessing_complete"] = True
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _count_request_modes(path: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    with Path(path).open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            request_mode = str(record.get("request_mode", "unknown"))
            counts[request_mode] = counts.get(request_mode, 0) + 1
    return counts


def _write_simple_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["部门层级", "人名", "职务"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {
                "部门层级": row["department_path"],
                "人名": row["name"],
                "职务": row["position"],
            }
            for row in rows
        )


def _write_audit_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "correction_type",
        "scope",
        "position",
        "original",
        "corrected",
        "affected_csv_rows",
        "affected_evidence_count",
        "source_frame",
        "basis",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize personnel CSV files without calling any model service."
    )
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    result = finalize_personnel_outputs(args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
