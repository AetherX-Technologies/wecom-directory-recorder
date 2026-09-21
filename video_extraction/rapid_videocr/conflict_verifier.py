# -*- encoding: utf-8 -*-
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .personnel_extractor import deduplicate_personnel_rows


def build_verification_prompt(name_a: str, name_b: str, position: str) -> str:
    return """你只做人员姓名逐字核验，不补全、不猜测。图片是人员列表。
请分别检查候选姓名“{name_a}”和“{name_b}”是否在画面的人名列中逐字清晰出现；灰色职务“{position}”只用于定位。即使同职务有多人，也要分别判断两个候选。
严格只返回 JSON：{{"candidate_a_visible":true或false,"candidate_b_visible":true或false,"visible_names":["实际清晰读到的人名"]}}。""".format(
        name_a=name_a, name_b=name_b, position=position
    )


def parse_verification_response(content: str) -> Dict[str, Any]:
    text = str(content).strip()
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Verification response does not contain JSON")
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Verification response must be an object")
    visible_names = value.get("visible_names", [])
    if not isinstance(visible_names, list):
        visible_names = []
    return {
        "candidate_a_visible": value.get("candidate_a_visible") is True,
        "candidate_b_visible": value.get("candidate_b_visible") is True,
        "visible_names": [str(name).strip() for name in visible_names if str(name).strip()],
    }


def resolve_visibility_results(
    results: Sequence[Mapping[str, Any]],
) -> str:
    a_visible = any(result.get("candidate_a_visible") is True for result in results)
    b_visible = any(result.get("candidate_b_visible") is True for result in results)
    if a_visible and b_visible:
        return "keep_both"
    if a_visible:
        return "merge_b_into_a"
    if b_visible:
        return "merge_a_into_b"
    return "unresolved"


def verify_near_name_conflicts(
    rows: Sequence[Mapping[str, Any]],
    conflicts: Sequence[Mapping[str, Any]],
    client: Any,
    output_path: Path,
    workers: int = 8,
) -> List[Dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    row_index = {
        (
            str(row["department_path"]),
            str(row["name"]),
            str(row["position"]),
        ): row
        for row in rows
    }
    tasks = []
    for conflict_index, conflict in enumerate(conflicts):
        if conflict.get("conflict_type") != "near_names_same_position":
            continue
        department_path = str(conflict["department_path"])
        name_a = str(conflict["name_a"])
        name_b = str(conflict["name_b"])
        position = str(conflict["position_a"])
        for side, name in (("a", name_a), ("b", name_b)):
            row = row_index.get((department_path, name, position))
            if row is None:
                continue
            tasks.append(
                {
                    "conflict_index": conflict_index,
                    "side": side,
                    "department_path": department_path,
                    "name_a": name_a,
                    "name_b": name_b,
                    "position": position,
                    "source_frame": str(row["source_frame"]),
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    lock = threading.Lock()
    completed: List[Dict[str, Any]] = []

    def run(task: Mapping[str, Any]) -> Dict[str, Any]:
        image = _read_image(Path(str(task["source_frame"])))
        prompt = build_verification_prompt(
            str(task["name_a"]), str(task["name_b"]), str(task["position"])
        )
        response = client.analyze(image, prompt, max_tokens=512)
        return {
            **dict(task),
            "parsed": parse_verification_response(response.content),
            "model_response": response.raw,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(run, task): task for task in tasks}
        for completed_count, future in enumerate(as_completed(future_map), start=1):
            record = future.result()
            with lock:
                completed.append(record)
                with output_path.open("a", encoding="utf-8", newline="\n") as output_file:
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                "Verify conflict {}/{} {} side={}".format(
                    completed_count,
                    len(tasks),
                    record["name_a"] + "/" + record["name_b"],
                    record["side"],
                ),
                flush=True,
            )

    by_conflict: Dict[int, List[Mapping[str, Any]]] = {}
    for record in completed:
        by_conflict.setdefault(int(record["conflict_index"]), []).append(
            record["parsed"]
        )
    resolutions = []
    for conflict_index, conflict in enumerate(conflicts):
        if conflict.get("conflict_type") != "near_names_same_position":
            continue
        results = by_conflict.get(conflict_index, [])
        resolutions.append(
            {
                **dict(conflict),
                "resolution": resolve_visibility_results(results),
                "verification_result_count": len(results),
            }
        )
    return resolutions


def apply_name_resolutions(
    rows: Iterable[Mapping[str, Any]], resolutions: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    mappings: Dict[Tuple[str, str, str], str] = {}
    for resolution in resolutions:
        department_path = str(resolution["department_path"])
        position = str(resolution["position_a"])
        action = resolution.get("resolution")
        if action == "merge_b_into_a":
            mappings[(department_path, str(resolution["name_b"]), position)] = str(
                resolution["name_a"]
            )
        elif action == "merge_a_into_b":
            mappings[(department_path, str(resolution["name_a"]), position)] = str(
                resolution["name_b"]
            )

    updated = []
    for raw_row in rows:
        row = dict(raw_row)
        key = (
            str(row["department_path"]),
            str(row["name"]),
            str(row["position"]),
        )
        replacement = mappings.get(key)
        if replacement:
            row["name"] = replacement
        updated.append(row)
    return deduplicate_personnel_rows(updated)


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError("Unable to read verification frame: {}".format(path))
    return image
