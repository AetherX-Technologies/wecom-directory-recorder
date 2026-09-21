# -*- encoding: utf-8 -*-
"""Extract WeCom contact-detail records from a long screen recording.

This adapter is intentionally separate from ``personnel_extractor``.  The old
adapter reads a portrait personnel list; this one reads the fixed detail panel
of the desktop WeCom contact directory.
"""

import argparse
import csv
import hashlib
import json
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .screen_ocr import format_timestamp
from .vision_client import QwenVisionClient


WECHAT_DETAIL_PROMPT = """你是企业微信通讯录个人详情页的信息抽取器。每张图片只包含屏幕右侧详情区域。

逐张判断并抽取：
1. 只有清楚显示个人姓名及“工号”“部门”等详情字段时，page_type 才是 person；目录页、空白页、加载页均不是 person。
2. name：姓名原文。
3. position：姓名正下方的灰色职务原文；画面没有灰色职务时必须返回空字符串，不得从部门名推测。
4. user_code：“工号”字段右侧的原文。这是人员唯一标识，不能改写或补零。
5. departments：“部门”字段右侧实际显示的全部部门路径，按从上到下顺序返回。长路径可能因列宽自动折行，应把折行片段无空格拼回同一条路径；只有明显开始的新路径才另列一项。
6. 忽略邮箱、座机、手机、办公地点、企业、头像、按钮、左侧名单、鼠标指针、水印和日期。
7. 看不清的内容留空，不得根据常识补全。

严格只返回一个 JSON 对象，不要 Markdown，不要解释：
{"frames":[{"image_index":0,"page_type":"person|folder|blank|other","name":"","position":"","user_code":"","departments":[]}]}"""

SPACE_RE = re.compile(r"\s+")
DEPARTMENT_SPACE_RE = re.compile(r"\s*(/)\s*")
USER_CODE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
PERSON_COUNT_RE = re.compile(r"^[~≈]?\s*\d+\s*人$")


def compose_wechat_detail_roi(frame: np.ndarray) -> np.ndarray:
    """Crop the fixed right-side personal detail panel from a 984x794 layout."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame is empty")
    height, width = frame.shape[:2]
    if width < 320 or height < 320:
        raise ValueError("frame is too small")

    left = max(0, round(width * 0.415))
    right = min(width, round(width * 0.875))
    top = max(0, round(height * 0.075))
    bottom = min(height, round(height * 0.91))
    roi = frame[top:bottom, left:right]
    if roi.size == 0:
        raise ValueError("detail ROI is empty")
    return roi


def _visual_signature(frame: np.ndarray) -> np.ndarray:
    roi = compose_wechat_detail_roi(frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (172, 116), interpolation=cv2.INTER_AREA)


def group_change_events(
    events: Sequence[Mapping[str, Any]], group_gap_seconds: float
) -> List[List[Dict[str, Any]]]:
    if group_gap_seconds < 0:
        raise ValueError("group_gap_seconds cannot be negative")
    groups: List[List[Dict[str, Any]]] = []
    for event in sorted(events, key=lambda item: int(item["source_frame_index"])):
        materialized = dict(event)
        if not groups:
            groups.append([materialized])
            continue
        gap = float(materialized["timestamp_seconds"]) - float(
            groups[-1][-1]["timestamp_seconds"]
        )
        if gap <= group_gap_seconds + 1e-9:
            groups[-1].append(materialized)
        else:
            groups.append([materialized])
    return groups


def build_stable_candidates(
    change_groups: Sequence[Sequence[Mapping[str, Any]]],
    decoded_frame_count: int,
    source_fps: float,
    min_stable_seconds: float,
) -> List[Dict[str, Any]]:
    if decoded_frame_count <= 0:
        return []
    if source_fps <= 0 or min_stable_seconds <= 0:
        raise ValueError("frame rate and minimum stable duration must be positive")

    groups = [list(group) for group in change_groups if group]
    intervals: List[Tuple[int, int, Optional[int]]] = []
    if not groups:
        intervals.append((0, decoded_frame_count - 1, None))
    else:
        first_start = int(groups[0][0]["source_frame_index"])
        intervals.append((0, first_start - 1, None))
        for index, group in enumerate(groups):
            start = int(group[-1]["source_frame_index"])
            if index + 1 < len(groups):
                end = int(groups[index + 1][0]["source_frame_index"]) - 1
            else:
                end = decoded_frame_count - 1
            intervals.append((start, end, index))

    candidates = []
    for start, end, group_index in intervals:
        if end < start:
            continue
        stable_seconds = (end - start + 1) / source_fps
        if stable_seconds + 1e-9 < min_stable_seconds:
            continue
        source_frame_index = int(round((start + end) / 2.0))
        timestamp_seconds = source_frame_index / source_fps
        candidates.append(
            {
                "selection_index": len(candidates),
                "source_frame_index": source_frame_index,
                "timestamp_seconds": round(timestamp_seconds, 6),
                "timestamp": format_timestamp(timestamp_seconds),
                "stable_start_frame": start,
                "stable_end_frame": end,
                "stable_duration_seconds": round(stable_seconds, 6),
                "preceding_change_group": group_index,
            }
        )
    return candidates


def select_stable_detail_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 4.0,
    change_threshold: float = 0.2,
    group_gap_seconds: float = 1.0,
    min_stable_seconds: float = 0.75,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError("Video does not exist: {}".format(video_path))
    if sample_fps <= 0 or change_threshold < 0:
        raise ValueError("sampling rate must be positive and threshold non-negative")

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open video: {}".format(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    metadata_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or metadata_frame_count <= 0:
        capture.release()
        raise RuntimeError("Video metadata is invalid")

    source_step = source_fps / sample_fps
    next_sample_frame = 0.0
    source_frame_index = 0
    sampled_frame_count = 0
    previous_signature: Optional[np.ndarray] = None
    events: List[Dict[str, Any]] = []
    try:
        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            if source_frame_index + 1e-9 >= next_sample_frame:
                ok, frame = capture.retrieve()
                if ok:
                    sampled_frame_count += 1
                    signature = _visual_signature(frame)
                    if previous_signature is not None:
                        difference = float(
                            np.mean(cv2.absdiff(signature, previous_signature))
                        )
                        if difference >= change_threshold:
                            timestamp_seconds = source_frame_index / source_fps
                            events.append(
                                {
                                    "source_frame_index": source_frame_index,
                                    "timestamp_seconds": round(
                                        timestamp_seconds, 6
                                    ),
                                    "timestamp": format_timestamp(
                                        timestamp_seconds
                                    ),
                                    "change_score": round(difference, 6),
                                }
                            )
                    previous_signature = signature
                while next_sample_frame <= source_frame_index + 1e-9:
                    next_sample_frame += source_step
            source_frame_index += 1
    finally:
        capture.release()

    groups = group_change_events(events, group_gap_seconds)
    candidates = build_stable_candidates(
        groups,
        decoded_frame_count=source_frame_index,
        source_fps=source_fps,
        min_stable_seconds=min_stable_seconds,
    )
    _write_jsonl(output_dir / "change_events.jsonl", events)
    _write_jsonl(
        output_dir / "change_groups.jsonl",
        [
            {
                "group_index": index,
                "first_source_frame_index": group[0]["source_frame_index"],
                "last_source_frame_index": group[-1]["source_frame_index"],
                "first_timestamp_seconds": group[0]["timestamp_seconds"],
                "last_timestamp_seconds": group[-1]["timestamp_seconds"],
                "event_count": len(group),
                "max_change_score": max(
                    float(event["change_score"]) for event in group
                ),
            }
            for index, group in enumerate(groups)
        ],
    )

    wanted = {int(item["source_frame_index"]): item for item in candidates}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to reopen video: {}".format(video_path))
    second_pass_count = 0
    written = 0
    try:
        while wanted and True:
            grabbed = capture.grab()
            if not grabbed:
                break
            record = wanted.get(second_pass_count)
            if record is not None:
                ok, frame = capture.retrieve()
                if ok:
                    roi = compose_wechat_detail_roi(frame)
                    frame_name = "s{:05d}_f{:06d}_t{:010.3f}.jpg".format(
                        int(record["selection_index"]),
                        second_pass_count,
                        float(record["timestamp_seconds"]),
                    )
                    frame_path = frames_dir / frame_name
                    _write_jpeg(frame_path, roi)
                    record["frame"] = (Path("frames") / frame_name).as_posix()
                    record["frame_sha256"] = hashlib.sha256(
                        frame_path.read_bytes()
                    ).hexdigest()
                    written += 1
                wanted.pop(second_pass_count, None)
            second_pass_count += 1
    finally:
        capture.release()

    selected = [item for item in candidates if item.get("frame_sha256")]
    _write_jsonl(output_dir / "selected_frames.jsonl", selected)
    summary = {
        "video": str(video_path.resolve()),
        "source_fps": round(source_fps, 6),
        "metadata_frame_count": metadata_frame_count,
        "decoded_frame_count": source_frame_index,
        "metadata_frame_count_delta": metadata_frame_count - source_frame_index,
        "metadata_frame_count_matches_decoded": (
            metadata_frame_count == source_frame_index
        ),
        "all_decodable_frames_scanned": True,
        "sample_fps": sample_fps,
        "sampled_frame_count": sampled_frame_count,
        "change_threshold": change_threshold,
        "change_event_count": len(events),
        "group_gap_seconds": group_gap_seconds,
        "change_group_count": len(groups),
        "min_stable_seconds": min_stable_seconds,
        "stable_candidate_count": len(candidates),
        "selected_frame_count": len(selected),
        "second_pass_written_count": written,
        "unwritten_candidate_count": len(candidates) - written,
    }
    _write_json(output_dir / "selection_summary.json", summary)
    return selected, summary


def parse_wechat_detail_response(content: str) -> Dict[str, Any]:
    return parse_wechat_detail_batch_response(content, 1)[0]


def parse_wechat_detail_batch_response(
    content: str, expected_count: int
) -> List[Dict[str, Any]]:
    if expected_count <= 0:
        raise ValueError("expected_count must be greater than zero")
    value = _parse_json_object(content)
    raw_frames = value.get("frames")
    if not isinstance(raw_frames, list):
        if expected_count == 1:
            return [_normalize_detail_object(value)]
        raise ValueError("Batch model response does not contain frames")

    by_index: Dict[int, Dict[str, Any]] = {}
    for fallback_index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            continue
        try:
            image_index = int(raw_frame.get("image_index", fallback_index))
        except (TypeError, ValueError):
            continue
        if 0 <= image_index < expected_count and image_index not in by_index:
            by_index[image_index] = _normalize_detail_object(raw_frame)
    missing = [index for index in range(expected_count) if index not in by_index]
    if missing:
        raise ValueError(
            "Batch model response is missing image indexes: {}".format(
                ", ".join(str(index) for index in missing)
            )
        )
    return [by_index[index] for index in range(expected_count)]


def _normalize_detail_object(value: Mapping[str, Any]) -> Dict[str, Any]:
    page_type = _clean(value.get("page_type", "other")).lower()
    if page_type not in {"person", "folder", "blank", "other"}:
        page_type = "other"
    name = _clean(value.get("name", ""))
    position = _clean(value.get("position", ""))
    user_code = _clean(value.get("user_code", ""))
    departments = []
    seen = set()
    raw_departments = value.get("departments", [])
    if isinstance(raw_departments, str):
        raw_departments = [raw_departments]
    if isinstance(raw_departments, list):
        for raw_department in raw_departments:
            department = _clean_department(raw_department)
            if department and department not in seen:
                departments.append(department)
                seen.add(department)
    if page_type == "person" and not (name or user_code or departments):
        page_type = "other"
    return {
        "page_type": page_type,
        "name": name,
        "position": position,
        "user_code": user_code,
        "departments": departments,
    }


def analyze_selected_detail_frames(
    output_dir: Path,
    selected_frames: Sequence[Mapping[str, Any]],
    client: Any,
    workers: int = 4,
    batch_size: int = 4,
    limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output_dir = Path(output_dir)
    if workers <= 0 or batch_size <= 0:
        raise ValueError("workers and batch_size must be greater than zero")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    selected = list(selected_frames)
    if limit is not None:
        selected = selected[:limit]
    raw_path = output_dir / "raw_model.jsonl"
    failure_path = output_dir / "model_failures.jsonl"
    existing = {
        int(record.get("selection_index")): record
        for record in _load_jsonl(raw_path)
        if str(record.get("selection_index", "")).isdigit()
        and isinstance(record.get("parsed"), dict)
    }
    pending = [
        dict(record)
        for record in selected
        if int(record["selection_index"]) not in existing
    ]
    failures: List[Dict[str, Any]] = []
    lock = threading.Lock()
    batches = [
        pending[index : index + batch_size]
        for index in range(0, len(pending), batch_size)
    ]

    def run(batch: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        images = [
            _read_image(output_dir / str(record["frame"])) for record in batch
        ]
        response = client.analyze_images(
            images,
            WECHAT_DETAIL_PROMPT,
            max_tokens=max(2048, 768 * len(images)),
        )
        request_mode = "batch" if len(images) > 1 else "single"
        try:
            parsed_frames = parse_wechat_detail_batch_response(
                response.content, len(images)
            )
            raw_responses = [response.raw] * len(images)
        except (ValueError, json.JSONDecodeError):
            if len(images) == 1:
                raise
            request_mode = "single_retry"
            parsed_frames = []
            raw_responses = []
            for image in images:
                single_response = client.analyze_images(
                    [image], WECHAT_DETAIL_PROMPT, max_tokens=2048
                )
                parsed_frames.append(
                    parse_wechat_detail_batch_response(
                        single_response.content, 1
                    )[0]
                )
                raw_responses.append(single_response.raw)
        return [
            {
                **dict(record),
                "parsed": parsed,
                "request_mode": request_mode,
                "model_response": raw_response,
            }
            for record, parsed, raw_response in zip(
                batch, parsed_frames, raw_responses
            )
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run, batch): batch for batch in batches}
        completed = 0
        for future in as_completed(future_map):
            source_batch = future_map[future]
            try:
                records = future.result()
            except Exception as exc:
                for source in source_batch:
                    failure = {
                        **dict(source),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    with lock:
                        _append_jsonl(failure_path, failure)
            else:
                for record in records:
                    existing[int(record["selection_index"])] = record
                    with lock:
                        _append_jsonl(raw_path, record)
            completed += 1
            failed = future.exception() is not None
            if failed or completed == 1 or completed % 25 == 0 or completed == len(batches):
                print(
                    "WeCom model batch {}/{} timestamps={} status={}".format(
                        completed,
                        len(batches),
                        ",".join(str(item["timestamp"]) for item in source_batch),
                        "failed" if failed else "ok",
                    ),
                    flush=True,
                )

    ordered = [
        existing[int(record["selection_index"])]
        for record in selected
        if int(record["selection_index"]) in existing
    ]
    return ordered, failures


def build_wechat_outputs(
    video_name: str,
    analyzed_frames: Iterable[Mapping[str, Any]],
    overrides: Optional[Iterable[Mapping[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    overrides_by_code = {
        _clean(row.get("user_code", "")): dict(row)
        for row in (overrides or [])
        if _clean(row.get("user_code", ""))
    }
    observations = []
    ignored_count = 0
    for frame in sorted(
        analyzed_frames, key=lambda item: float(item["timestamp_seconds"])
    ):
        parsed = frame.get("parsed", {})
        if not isinstance(parsed, dict) or parsed.get("page_type") != "person":
            ignored_count += 1
            continue
        observations.append(
            {
                "user_code": _clean(parsed.get("user_code", "")),
                "name": _clean(parsed.get("name", "")),
                "position": _clean(parsed.get("position", "")),
                "departments": list(parsed.get("departments", [])),
                "video": video_name,
                "timestamp_seconds": frame["timestamp_seconds"],
                "timestamp": frame["timestamp"],
                "source_frame": frame["frame"],
                "frame_sha256": frame["frame_sha256"],
            }
        )

    review: List[Dict[str, Any]] = []
    by_code: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        code = observation["user_code"]
        if not code or not observation["name"]:
            review.append(
                _review_row("missing_identity", code, [observation])
            )
            continue
        if not USER_CODE_RE.fullmatch(code):
            review.append(
                _review_row("invalid_user_code", code, [observation])
            )
            continue
        by_code[code].append(observation)

    people = []
    memberships = []
    for code in sorted(by_code):
        records = by_code[code]
        names = [_clean(record["name"]) for record in records if record["name"]]
        positions = [
            _clean(record["position"])
            for record in records
            if record["position"]
        ]
        selected_name, name_conflict = _select_consensus(names)
        selected_position, position_conflict = _select_consensus(positions)
        override = overrides_by_code.get(code, {})
        override_name = _clean(override.get("name", ""))
        override_position = _clean(override.get("position", ""))
        override_note = _clean(override.get("resolution_note", ""))
        override_evidence = _clean(override.get("evidence", ""))
        override_applied = False
        if override_name:
            selected_name = override_name
            name_conflict = False
            override_applied = True
        if override_position:
            selected_position = override_position
            position_conflict = False
            override_applied = True
        if name_conflict:
            review.append(_review_row("name_conflict", code, records))
        if position_conflict:
            review.append(_review_row("position_conflict", code, records))

        department_evidence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            for department in record["departments"]:
                normalized = _clean_department(department)
                if normalized:
                    department_evidence[normalized].append(record)
        if not department_evidence:
            review.append(_review_row("missing_department", code, records))

        first = records[0]
        people.append(
            {
                "user_code": code,
                "name": selected_name,
                "position": selected_position,
                "department_count": len(department_evidence),
                "departments": " | ".join(sorted(department_evidence)),
                "evidence_count": len(records),
                "first_timestamp": first["timestamp"],
                "first_source_frame": first["source_frame"],
                "review_status": (
                    "needs_review"
                    if name_conflict or position_conflict or not department_evidence
                    else "ok"
                ),
                "override_applied": "yes" if override_applied else "no",
                "override_note": override_note if override_applied else "",
                "override_evidence": override_evidence if override_applied else "",
            }
        )
        for department in sorted(department_evidence):
            evidence = department_evidence[department]
            memberships.append(
                {
                    "user_code": code,
                    "name": selected_name,
                    "position": selected_position,
                    "department_path": department,
                    "evidence_count": len(evidence),
                    "first_timestamp": evidence[0]["timestamp"],
                    "first_source_frame": evidence[0]["source_frame"],
                    "override_applied": "yes" if override_applied else "no",
                }
            )

    position_records = [
        {
            "user_code": row["user_code"],
            "name": row["name"],
            "position": row["position"],
            "department_count": row["department_count"],
            "departments": row["departments"],
            "evidence_count": row["evidence_count"],
            "first_timestamp": row["first_timestamp"],
            "first_source_frame": row["first_source_frame"],
            "identity_status": "verified_user_code",
            "review_status": row["review_status"],
            "override_applied": row["override_applied"],
        }
        for row in people
        if row["position"]
    ]
    seen_unidentified_positions = set()
    for observation in observations:
        position = _clean(observation["position"])
        code = _clean(observation["user_code"])
        if (
            not observation["name"]
            or not position
            or PERSON_COUNT_RE.fullmatch(position)
            or (code and USER_CODE_RE.fullmatch(code))
        ):
            continue
        departments = tuple(
            _clean_department(item) for item in observation["departments"] if item
        )
        key = (observation["name"], position, departments)
        if key in seen_unidentified_positions:
            continue
        seen_unidentified_positions.add(key)
        position_records.append(
            {
                "user_code": code,
                "name": observation["name"],
                "position": position,
                "department_count": len(departments),
                "departments": " | ".join(departments),
                "evidence_count": 1,
                "first_timestamp": observation["timestamp"],
                "first_source_frame": observation["source_frame"],
                "identity_status": (
                    "invalid_user_code" if code else "missing_user_code"
                ),
                "review_status": "needs_review",
                "override_applied": "no",
            }
        )

    flat_observations = []
    for observation in observations:
        departments = observation.pop("departments")
        flat_observations.append(
            {
                **observation,
                "department_count": len(departments),
                "departments": " | ".join(departments),
            }
        )
    return {
        "observations": flat_observations,
        "people": people,
        "people_with_positions": [row for row in people if row["position"]],
        "position_records": position_records,
        "memberships": memberships,
        "review": review,
        "ignored": [{"ignored_frame_count": ignored_count}],
    }


def write_wechat_outputs(output_dir: Path, outputs: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    _write_csv(
        output_dir / "wechat_detail_observations.csv",
        outputs["observations"],
        [
            "user_code",
            "name",
            "position",
            "department_count",
            "departments",
            "video",
            "timestamp_seconds",
            "timestamp",
            "source_frame",
            "frame_sha256",
        ],
    )
    people_fields = [
        "user_code",
        "name",
        "position",
        "department_count",
        "departments",
        "evidence_count",
        "first_timestamp",
        "first_source_frame",
        "review_status",
        "override_applied",
        "override_note",
        "override_evidence",
    ]
    _write_csv(output_dir / "wechat_people.csv", outputs["people"], people_fields)
    _write_csv(
        output_dir / "wechat_people_with_positions.csv",
        outputs["people_with_positions"],
        people_fields,
    )
    _write_csv(
        output_dir / "wechat_positions_all.csv",
        outputs["position_records"],
        [
            "user_code",
            "name",
            "position",
            "department_count",
            "departments",
            "evidence_count",
            "first_timestamp",
            "first_source_frame",
            "identity_status",
            "review_status",
            "override_applied",
        ],
    )
    _write_csv(
        output_dir / "wechat_memberships.csv",
        outputs["memberships"],
        [
            "user_code",
            "name",
            "position",
            "department_path",
            "evidence_count",
            "first_timestamp",
            "first_source_frame",
            "override_applied",
        ],
    )
    _write_csv(
        output_dir / "wechat_review.csv",
        outputs["review"],
        [
            "reason",
            "user_code",
            "names",
            "positions",
            "departments",
            "timestamps",
            "source_frames",
        ],
    )


def _review_row(
    reason: str, code: str, records: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    return {
        "reason": reason,
        "user_code": code,
        "names": " | ".join(sorted({_clean(row.get("name", "")) for row in records if row.get("name")})),
        "positions": " | ".join(sorted({_clean(row.get("position", "")) for row in records if row.get("position")})),
        "departments": " | ".join(
            sorted(
                {
                    _clean_department(department)
                    for row in records
                    for department in row.get("departments", [])
                    if _clean_department(department)
                }
            )
        ),
        "timestamps": " | ".join(str(row.get("timestamp", "")) for row in records),
        "source_frames": " | ".join(
            str(row.get("source_frame", "")) for row in records
        ),
    }


def _select_consensus(values: Sequence[str]) -> Tuple[str, bool]:
    cleaned = [value for value in (_clean(item) for item in values) if value]
    if not cleaned:
        return "", False
    counts = Counter(cleaned)
    highest = max(counts.values())
    winners = sorted(value for value, count in counts.items() if count == highest)
    return winners[0], len(counts) > 1


def _clean(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").strip())


def _clean_department(value: Any) -> str:
    text = _clean(value)
    text = DEPARTMENT_SPACE_RE.sub(r"\1", text)
    return text.replace(" /", "/").replace("/ ", "/")


def _parse_json_object(content: str) -> Dict[str, Any]:
    text = _strip_code_fence(str(content).strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object")
    return value


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
    )
    if not ok:
        raise RuntimeError("Unable to encode frame: {}".format(path))
    path.write_bytes(encoded.tobytes())


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Unable to read image: {}".format(path))
    return image


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_selected_frames(path: Path) -> List[Dict[str, Any]]:
    return [record for record in _load_jsonl(path) if record.get("frame_sha256")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract WeCom personal details from a desktop screen recording."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--change-threshold", type=float, default=0.2)
    parser.add_argument("--group-gap", type=float, default=1.0)
    parser.add_argument("--min-stable", type=float, default=0.75)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--reuse-selection", action="store_true")
    parser.add_argument(
        "--overrides",
        type=Path,
        help="Optional reviewed CSV with user_code,name,position and evidence.",
    )
    args = parser.parse_args()

    selected_path = args.output_dir / "selected_frames.jsonl"
    if args.reuse_selection and selected_path.is_file():
        selected = _load_selected_frames(selected_path)
        selection_summary = json.loads(
            (args.output_dir / "selection_summary.json").read_text(
                encoding="utf-8"
            )
        )
    else:
        selected, selection_summary = select_stable_detail_frames(
            args.video,
            args.output_dir,
            sample_fps=args.sample_fps,
            change_threshold=args.change_threshold,
            group_gap_seconds=args.group_gap,
            min_stable_seconds=args.min_stable,
        )
    print(json.dumps(selection_summary, ensure_ascii=False, indent=2), flush=True)
    if args.selection_only:
        return

    client = QwenVisionClient.from_env()
    analyzed, failures = analyze_selected_detail_frames(
        args.output_dir,
        selected,
        client,
        workers=args.workers,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    override_rows = _read_csv(args.overrides) if args.overrides else []
    outputs = build_wechat_outputs(
        args.video.name, analyzed, overrides=override_rows
    )
    write_wechat_outputs(args.output_dir, outputs)
    analyzed_indexes = {
        int(record["selection_index"]) for record in analyzed
    }
    unresolved = [
        dict(record)
        for record in selected
        if int(record["selection_index"]) not in analyzed_indexes
    ]
    _write_jsonl(args.output_dir / "unresolved_model_frames.jsonl", unresolved)
    historical_failure_attempts = len(
        _load_jsonl(args.output_dir / "model_failures.jsonl")
    )
    summary = {
        **selection_summary,
        "analyzed_frame_count": len(analyzed),
        "model_failure_count_this_run": len(failures),
        "historical_failure_attempt_count": historical_failure_attempts,
        "unresolved_model_frame_count": len(unresolved),
        "person_observation_count": len(outputs["observations"]),
        "unique_person_count": len(outputs["people"]),
        "people_with_positions_count": len(outputs["people_with_positions"]),
        "position_record_count": len(outputs["position_records"]),
        "membership_count": len(outputs["memberships"]),
        "review_count": len(outputs["review"]),
        "override_count": len(override_rows),
        "ignored_model_frame_count": outputs["ignored"][0][
            "ignored_frame_count"
        ],
        "limited_run": args.limit is not None,
    }
    _write_json(args.output_dir / "wechat_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
