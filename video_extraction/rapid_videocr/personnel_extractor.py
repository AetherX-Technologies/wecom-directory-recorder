# -*- encoding: utf-8 -*-
import argparse
import csv
import hashlib
import json
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .screen_ocr import format_timestamp
from .vision_client import QwenVisionClient


PERSONNEL_PROMPT = """你是组织通讯录图像信息抽取器。输入图像由两部分拼接：最上方是当前页面标题条，下方是右侧内容列表。

任务：
1. 只提取人员行中同时完整显示的“人名”和人名下方的灰色职务文字。
2. 人名下面没有灰色职务、职务被裁断、只有头像或只有人名时，必须忽略该人。
3. 忽略文件夹名称、搜索框、状态栏、蓝色设备图标、人数、所有水印和日期。
4. 若当前是目录页，people 返回空数组，并提取画面中实际可见的面包屑 breadcrumb。
5. 不得根据常识补全画面中不可见的信息；不得改写、概括或扩充职务。

严格只返回一个 JSON 对象，不要 Markdown 代码块，不要解释：
{"page_type":"people|folder|other","page_title":"画面顶部标题","breadcrumb":["从左到右的可见层级"],"people":[{"name":"人名","position":"灰色职务原文"}]}"""

GENERIC_ROOTS = {"企业通讯录", "通讯录"}
SPACE_RE = re.compile(r"\s+")
PATH_SPLIT_RE = re.compile(r"\s*(?:>|›|/|→)\s*")


def compose_personnel_roi(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        raise ValueError("frame is empty")
    height, width = frame.shape[:2]
    if width < 100 or height < 100:
        raise ValueError("frame is too small")

    panel_x = round(width * 0.268)
    header_height = max(1, round(height * 0.084))
    list_y = round(height * 0.060)
    list_bottom = round(height * 0.947)
    list_width = min(round(width * 0.377), width - panel_x)
    output_width = 900

    header = frame[:header_height, panel_x:]
    header_output_height = max(1, round(header.shape[0] * output_width / header.shape[1]))
    header = cv2.resize(
        header,
        (output_width, header_output_height),
        interpolation=cv2.INTER_AREA,
    )
    content = frame[list_y:list_bottom, panel_x : panel_x + list_width]
    content = cv2.resize(
        content,
        (output_width, round(content.shape[0] * output_width / content.shape[1])),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.vconcat([header, content])


def select_dynamic_roi_frames(
    video_path: Path,
    output_dir: Path,
    sample_fps: float = 4.0,
    min_interval_seconds: float = 0.75,
    change_threshold: float = 1.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError("Video does not exist: {}".format(video_path))
    if sample_fps <= 0 or min_interval_seconds <= 0:
        raise ValueError("sampling values must be greater than zero")
    if change_threshold < 0:
        raise ValueError("change_threshold cannot be negative")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open video: {}".format(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if source_fps <= 0 or expected_frame_count <= 0:
        capture.release()
        raise RuntimeError("Video metadata is invalid")

    source_step = source_fps / sample_fps
    next_sample_frame = 0.0
    source_frame_index = 0
    sampled_frame_count = 0
    records: List[Dict[str, Any]] = []
    previous_selected_signature: Optional[np.ndarray] = None
    previous_selected_timestamp = -math.inf

    try:
        while True:
            grabbed = capture.grab()
            if not grabbed:
                break
            should_sample = source_frame_index + 1e-9 >= next_sample_frame
            if should_sample:
                ok, frame = capture.retrieve()
                if ok:
                    sampled_frame_count += 1
                    timestamp_seconds = source_frame_index / source_fps
                    roi = compose_personnel_roi(frame)
                    signature = _visual_signature(roi)
                    difference = (
                        float("inf")
                        if previous_selected_signature is None
                        else float(
                            np.mean(
                                cv2.absdiff(signature, previous_selected_signature)
                            )
                        )
                    )
                    interval = timestamp_seconds - previous_selected_timestamp
                    if (
                        previous_selected_signature is None
                        or (
                            interval >= min_interval_seconds
                            and difference >= change_threshold
                        )
                    ):
                        frame_name = "s{:05d}_f{:06d}_t{:010.3f}.jpg".format(
                            len(records), source_frame_index, timestamp_seconds
                        )
                        frame_path = frames_dir / frame_name
                        _write_jpeg(frame_path, roi)
                        digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
                        records.append(
                            {
                                "selection_index": len(records),
                                "source_frame_index": source_frame_index,
                                "timestamp_seconds": round(timestamp_seconds, 6),
                                "timestamp": format_timestamp(timestamp_seconds),
                                "frame": (Path("frames") / frame_name).as_posix(),
                                "frame_sha256": digest,
                                "change_score": (
                                    None
                                    if not math.isfinite(difference)
                                    else round(difference, 4)
                                ),
                            }
                        )
                        previous_selected_signature = signature
                        previous_selected_timestamp = timestamp_seconds
                while next_sample_frame <= source_frame_index + 1e-9:
                    next_sample_frame += source_step
            source_frame_index += 1
    finally:
        capture.release()

    _write_jsonl(output_dir / "selected_frames.jsonl", records)
    selection_summary = {
        "video": str(video_path.resolve()),
        "source_fps": round(source_fps, 6),
        "metadata_frame_count": expected_frame_count,
        "expected_frame_count": expected_frame_count,
        "decoded_frame_count": source_frame_index,
        "scanned_frame_count": source_frame_index,
        "metadata_frame_count_delta": expected_frame_count - source_frame_index,
        "metadata_frame_count_matches_decoded": (
            source_frame_index == expected_frame_count
        ),
        "all_decodable_frames_scanned": True,
        "all_source_frames_scanned": True,
        "sample_fps": sample_fps,
        "sampled_frame_count": sampled_frame_count,
        "min_interval_seconds": min_interval_seconds,
        "change_threshold": change_threshold,
        "selected_frame_count": len(records),
    }
    return records, selection_summary


def parse_personnel_response(content: str) -> Dict[str, Any]:
    return _normalize_personnel_object(_parse_json_object(content))


def parse_personnel_batch_response(
    content: str, expected_count: int
) -> List[Dict[str, Any]]:
    if expected_count <= 0:
        raise ValueError("expected_count must be greater than zero")
    value = _parse_json_object(content)
    raw_frames = value.get("frames")
    if not isinstance(raw_frames, list):
        if expected_count == 1:
            return [_normalize_personnel_object(value)]
        raise ValueError("Batch model response does not contain frames")

    by_index: Dict[int, Dict[str, Any]] = {}
    for fallback_index, raw_frame in enumerate(raw_frames):
        if not isinstance(raw_frame, dict):
            continue
        raw_index = raw_frame.get("image_index", fallback_index)
        try:
            image_index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if 0 <= image_index < expected_count and image_index not in by_index:
            by_index[image_index] = _normalize_personnel_object(raw_frame)
    missing = [index for index in range(expected_count) if index not in by_index]
    if missing:
        raise ValueError(
            "Batch model response is missing image indexes: {}".format(
                ", ".join(str(index) for index in missing)
            )
        )
    return [by_index[index] for index in range(expected_count)]


def _normalize_personnel_object(value: Mapping[str, Any]) -> Dict[str, Any]:

    page_type = _normalize_text(value.get("page_type", "other")).lower()
    if page_type not in {"people", "folder", "other"}:
        page_type = "other"
    page_title = _normalize_text(value.get("page_title", ""))
    breadcrumb = _normalize_breadcrumb(value.get("breadcrumb", []))
    people = []
    raw_people = value.get("people", [])
    if isinstance(raw_people, list):
        seen = set()
        for item in raw_people:
            if not isinstance(item, dict):
                continue
            name = _normalize_text(item.get("name", ""))
            position = _normalize_text(item.get("position", ""))
            key = (name, position)
            if not name or not position or key in seen:
                continue
            seen.add(key)
            people.append({"name": name, "position": position})
    if people and page_type == "other":
        page_type = "people"
    return {
        "page_type": page_type,
        "page_title": page_title,
        "breadcrumb": breadcrumb,
        "people": people,
    }


def analyze_selected_frames(
    output_dir: Path,
    selected_frames: Sequence[Mapping[str, Any]],
    client: Any,
    workers: int = 4,
    batch_size: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    output_dir = Path(output_dir)
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    raw_path = output_dir / "raw_model.jsonl"
    failure_path = output_dir / "model_failures.jsonl"
    _write_jsonl(failure_path, [])
    existing = {
        str(record.get("frame_sha256")): record
        for record in _load_jsonl(raw_path)
        if record.get("frame_sha256")
        and record.get("request_mode") == "single"
        and isinstance(record.get("parsed"), dict)
    }
    pending = [
        dict(frame)
        for frame in selected_frames
        if str(frame["frame_sha256"]) not in existing
    ]
    lock = threading.Lock()
    failures: List[Dict[str, Any]] = []

    batches = [
        pending[index : index + batch_size]
        for index in range(0, len(pending), batch_size)
    ]

    def run(batch: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        images = [
            _read_image(output_dir / str(frame_record["frame"]))
            for frame_record in batch
        ]
        if len(batch) == 1:
            response = client.analyze(
                images[0], PERSONNEL_PROMPT, max_tokens=2048
            )
            return [
                {
                    **dict(batch[0]),
                    "parsed": parse_personnel_response(response.content),
                    "request_mode": "single",
                    "model_response": response.raw,
                }
            ]

        batch_prompt = PERSONNEL_PROMPT + (
            "\n\n本次按顺序提供了 {} 张图像，索引从 0 开始。请严格返回："
            '{{"frames":[{{"image_index":0,"page_type":"people|folder|other",'
            '"page_title":"","breadcrumb":[],"people":[]}}]}}。'
            "frames 必须包含每个图像索引且不得合并不同图像的人员。"
        ).format(len(images))
        response = client.analyze_images(images, batch_prompt, max_tokens=4096)
        request_mode = "batch"
        raw_responses = [response.raw] * len(batch)
        try:
            parsed_frames = parse_personnel_batch_response(
                response.content, expected_count=len(batch)
            )
        except (ValueError, json.JSONDecodeError):
            if len(batch) == 1:
                raise
            request_mode = "single_retry"
            parsed_frames = []
            raw_responses = []
            for image in images:
                single_response = client.analyze(
                    image, PERSONNEL_PROMPT, max_tokens=2048
                )
                parsed_frames.append(
                    parse_personnel_response(single_response.content)
                )
                raw_responses.append(single_response.raw)
        return [
            {
                **dict(frame_record),
                "parsed": parsed,
                "request_mode": request_mode,
                "model_response": raw_response,
            }
            for frame_record, parsed, raw_response in zip(
                batch, parsed_frames, raw_responses
            )
        ]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run, batch): batch for batch in batches}
        completed = 0
        for future in as_completed(future_map):
            source_batch = future_map[future]
            try:
                batch_records = future.result()
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
                for record in batch_records:
                    existing[str(record["frame_sha256"])] = record
                    with lock:
                        _append_jsonl(raw_path, record)
            completed += 1
            print(
                "Qwen batch {}/{} timestamps={} status={}".format(
                    completed,
                    len(batches),
                    ",".join(str(item["timestamp"]) for item in source_batch),
                    "failed" if future.exception() is not None else "ok",
                ),
                flush=True,
            )

    ordered = [
        existing[str(frame["frame_sha256"])]
        for frame in selected_frames
        if str(frame["frame_sha256"]) in existing
    ]
    return ordered, failures


def build_personnel_rows(
    video_name: str,
    output_dir: Path,
    analyzed_frames: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    sorted_frames = sorted(
        analyzed_frames, key=lambda item: float(item["timestamp_seconds"])
    )
    raw_folder_breadcrumbs = []
    for frame in sorted_frames:
        parsed = frame.get("parsed", {})
        if not isinstance(parsed, dict) or parsed.get("people"):
            continue
        breadcrumb = _normalize_breadcrumb(parsed.get("breadcrumb", []))
        if breadcrumb:
            raw_folder_breadcrumbs.append(breadcrumb)
    known_roots = {
        breadcrumb[1]
        for breadcrumb in raw_folder_breadcrumbs
        if len(breadcrumb) >= 2 and breadcrumb[0] in GENERIC_ROOTS
    }
    accepted_by_time = []
    for frame in sorted_frames:
        parsed = frame.get("parsed", {})
        if not isinstance(parsed, dict) or parsed.get("people"):
            continue
        accepted = _accepted_breadcrumb(
            parsed.get("breadcrumb", []), known_roots
        )
        if accepted:
            accepted_by_time.append((float(frame["timestamp_seconds"]), accepted))
    current_breadcrumb: List[str] = []
    rows = []

    for frame in sorted_frames:
        parsed = frame.get("parsed", {})
        if not isinstance(parsed, dict):
            continue
        people = parsed.get("people", [])
        if not people:
            accepted = _accepted_breadcrumb(
                parsed.get("breadcrumb", []), known_roots
            )
            if accepted:
                current_breadcrumb = accepted
        if not isinstance(people, list) or not people:
            continue

        path_parts = list(current_breadcrumb)
        if not path_parts:
            timestamp_seconds = float(frame["timestamp_seconds"])
            path_parts = next(
                (
                    list(path)
                    for timestamp, path in accepted_by_time
                    if timestamp >= timestamp_seconds
                ),
                [],
            )
        page_title = _normalize_text(parsed.get("page_title", ""))
        if page_title and (not path_parts or path_parts[-1] != page_title):
            path_parts.append(page_title)
        department_path = " > ".join(path_parts)
        source_frame = str((output_dir / str(frame["frame"])).resolve())
        for person in people:
            if not isinstance(person, dict):
                continue
            name = _normalize_text(person.get("name", ""))
            position = _normalize_text(person.get("position", ""))
            if not name or not position:
                continue
            rows.append(
                {
                    "department_path": department_path,
                    "name": name,
                    "position": position,
                    "video": video_name,
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "timestamp": frame["timestamp"],
                    "source_frame": source_frame,
                    "evidence_count": 1,
                }
            )
    return rows


def deduplicate_personnel_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    materialized = [dict(row) for row in rows]
    root_counts: Dict[str, int] = {}
    for row in materialized:
        root = _path_root(row.get("department_path", ""))
        if root:
            root_counts[root] = root_counts.get(root, 0) + 1
    root_aliases = _find_root_aliases(root_counts)

    unique: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in materialized:
        department_path = _normalize_root_alias(
            row.get("department_path", ""), root_aliases
        )
        key = (
            department_path,
            _normalize_text(row.get("name", "")),
            _normalize_text(row.get("position", "")),
        )
        if not key[1] or not key[2]:
            continue
        incoming_evidence_count = int(row.get("evidence_count", 1))
        if key in unique:
            unique[key]["evidence_count"] = (
                int(unique[key]["evidence_count"]) + incoming_evidence_count
            )
        else:
            row["department_path"], row["name"], row["position"] = key
            row["evidence_count"] = incoming_evidence_count
            unique[key] = row

    by_person: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in unique.values():
        by_person.setdefault(
            (str(row["department_path"]), str(row["name"])), []
        ).append(row)
    consolidated = []
    for person_rows in by_person.values():
        kept: List[Dict[str, Any]] = []
        for row in sorted(
            person_rows,
            key=lambda item: (
                len(_position_comparison_key(item["position"])),
                _position_display_quality(item["position"]),
            ),
            reverse=True,
        ):
            short_key = _position_comparison_key(row["position"])
            target = next(
                (
                    candidate
                    for candidate in kept
                    if len(short_key) >= 3
                    and _position_comparison_key(candidate["position"]).startswith(
                        short_key
                    )
                ),
                None,
            )
            if target is None:
                kept.append(row)
            else:
                target["evidence_count"] = int(target["evidence_count"]) + int(
                    row["evidence_count"]
                )
        consolidated.extend(kept)

    return sorted(
        consolidated,
        key=lambda row: (
            str(row["department_path"]),
            str(row["name"]),
            str(row["position"]),
        ),
    )


def find_review_conflicts(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    conflicts = []
    by_person: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    by_department_position: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        by_person.setdefault(
            (str(row["department_path"]), str(row["name"])), []
        ).append(row)
        by_department_position.setdefault(
            (str(row["department_path"]), str(row["position"])), []
        ).append(row)

    for (department_path, name), person_rows in by_person.items():
        positions = sorted({str(row["position"]) for row in person_rows})
        if len(positions) <= 1:
            continue
        conflicts.append(
            {
                "conflict_type": "same_person_multiple_positions",
                "department_path": department_path,
                "name_a": name,
                "position_a": " | ".join(positions),
                "name_b": "",
                "position_b": "",
            }
        )

    for (department_path, position), position_rows in by_department_position.items():
        names = sorted({str(row["name"]) for row in position_rows})
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                if not _one_character_apart(left, right):
                    continue
                conflicts.append(
                    {
                        "conflict_type": "near_names_same_position",
                        "department_path": department_path,
                        "name_a": left,
                        "position_a": position,
                        "name_b": right,
                        "position_b": position,
                    }
                )
    return sorted(
        conflicts,
        key=lambda item: (
            item["conflict_type"],
            item["department_path"],
            item["name_a"],
        ),
    )


def write_review_conflicts(
    path: Path, conflicts: Sequence[Mapping[str, Any]]
) -> None:
    fieldnames = [
        "conflict_type",
        "department_path",
        "name_a",
        "position_a",
        "name_b",
        "position_b",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(conflicts)


def write_personnel_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "department_path",
        "name",
        "position",
        "video",
        "timestamp_seconds",
        "timestamp",
        "source_frame",
        "evidence_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def process_videos(
    videos: Sequence[Path],
    output_root: Path,
    client: Any,
    sample_fps: float = 4.0,
    min_interval_seconds: float = 0.75,
    change_threshold: float = 1.5,
    workers: int = 4,
    batch_size: int = 1,
) -> Dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    video_summaries = []
    total_failures = 0

    for video_index, video_path in enumerate(videos, start=1):
        video_path = Path(video_path)
        video_output = output_root / video_path.stem
        video_output.mkdir(parents=True, exist_ok=True)
        print(
            "Selecting video {}/{}: {}".format(
                video_index, len(videos), video_path.name
            ),
            flush=True,
        )
        selected, selection_summary = select_dynamic_roi_frames(
            video_path,
            video_output,
            sample_fps=sample_fps,
            min_interval_seconds=min_interval_seconds,
            change_threshold=change_threshold,
        )
        analyzed, failures = analyze_selected_frames(
            video_output,
            selected,
            client=client,
            workers=workers,
            batch_size=batch_size,
        )
        rows = build_personnel_rows(video_path.name, video_output, analyzed)
        unique_rows = deduplicate_personnel_rows(rows)
        write_personnel_csv(video_output / "personnel_positions.csv", unique_rows)
        summary = {
            **selection_summary,
            "analyzed_frame_count": len(analyzed),
            "model_failure_count": len(failures),
            "raw_personnel_observation_count": len(rows),
            "unique_personnel_count": len(unique_rows),
        }
        (video_output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        video_summaries.append(summary)
        all_rows.extend(rows)
        total_failures += len(failures)

    unique_all_rows = deduplicate_personnel_rows(all_rows)
    csv_path = output_root / "personnel_positions.csv"
    write_personnel_csv(csv_path, unique_all_rows)
    conflicts = find_review_conflicts(unique_all_rows)
    conflict_path = output_root / "review_conflicts.csv"
    write_review_conflicts(conflict_path, conflicts)
    overall = {
        "videos": video_summaries,
        "video_count": len(videos),
        "model_failure_count": total_failures,
        "unique_personnel_count": len(unique_all_rows),
        "review_conflict_count": len(conflicts),
        "csv": str(csv_path.resolve()),
        "review_conflicts_csv": str(conflict_path.resolve()),
    }
    (output_root / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return overall


def _visual_signature(image: np.ndarray) -> np.ndarray:
    content = image[round(image.shape[0] * 0.045) :]
    grayscale = cv2.cvtColor(content, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, (128, 192), interpolation=cv2.INTER_AREA)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value)).strip()


def _path_root(value: Any) -> str:
    path = _normalize_text(value)
    return path.split(" > ", 1)[0] if path else ""


def _find_root_aliases(root_counts: Mapping[str, int]) -> Dict[str, str]:
    if not root_counts:
        return {}
    dominant_root, dominant_count = max(
        root_counts.items(), key=lambda item: item[1]
    )
    limit = max(3, round(dominant_count * 0.02))
    aliases = {}
    for root, count in root_counts.items():
        if (
            root != dominant_root
            and count <= limit
            and _one_character_apart(root, dominant_root)
        ):
            aliases[root] = dominant_root
    return aliases


def _normalize_root_alias(value: Any, aliases: Mapping[str, str]) -> str:
    path = _normalize_text(value)
    root = _path_root(path)
    replacement = aliases.get(root)
    if not replacement:
        return path
    return replacement + path[len(root) :]


def _position_comparison_key(value: Any) -> str:
    text = _normalize_text(value)
    translation = str.maketrans(
        {
            " ": "",
            "(": "（",
            ")": "）",
            ",": "，",
            ";": "；",
        }
    )
    translated = text.translate(translation)
    return re.sub(r"[、，,。；;：:]", "", translated)


def _position_display_quality(value: Any) -> Tuple[int, int]:
    text = _normalize_text(value)
    punctuation_count = len(re.findall(r"[、，,。；;：:]", text))
    return punctuation_count, len(text)


def _one_character_apart(left: str, right: str) -> bool:
    return len(left) == len(right) and sum(
        left_char != right_char
        for left_char, right_char in zip(left, right)
    ) == 1


def _normalize_breadcrumb(value: Any) -> List[str]:
    if isinstance(value, str):
        parts = PATH_SPLIT_RE.split(value)
    elif isinstance(value, list):
        parts = value
    else:
        parts = []
    normalized = []
    for part in parts:
        text = _normalize_text(part)
        if text and (not normalized or normalized[-1] != text):
            normalized.append(text)
    return normalized


def _accepted_breadcrumb(value: Any, known_roots: set) -> List[str]:
    breadcrumb = _normalize_breadcrumb(value)
    while breadcrumb and breadcrumb[0] in GENERIC_ROOTS:
        breadcrumb = breadcrumb[1:]
    if not breadcrumb:
        return []
    if known_roots and breadcrumb[0] not in known_roots:
        return []
    return breadcrumb


def _parse_json_object(content: str) -> Dict[str, Any]:
    text = _strip_code_fence(str(content).strip())
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model response does not contain a JSON object")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON must be an object")
    return value


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    encoded, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
    )
    if not encoded:
        raise RuntimeError("Unable to encode frame: {}".format(path))
    buffer.tofile(str(path))


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError("Unable to read frame: {}".format(path))
    return image


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract department, person name, and gray position text from videos."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=4.0)
    parser.add_argument("--min-interval", type=float, default=0.75)
    parser.add_argument("--change-threshold", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    videos = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )
    if not videos:
        raise SystemExit("No MP4 files found in {}".format(args.input_dir))
    client = QwenVisionClient.from_env()
    summary = process_videos(
        videos,
        args.output_dir,
        client=client,
        sample_fps=args.sample_fps,
        min_interval_seconds=args.min_interval,
        change_threshold=args.change_threshold,
        workers=args.workers,
        batch_size=args.batch_size,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["model_failure_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
