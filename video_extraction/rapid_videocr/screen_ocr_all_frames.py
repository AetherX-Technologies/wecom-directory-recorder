# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import argparse
import html
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .ocr_client import PaddleOCRVLClient
from .screen_ocr import format_timestamp, normalize_ocr_line


LOCATION_TOKEN_RE = re.compile(r"<\|LOC_\d+\|>")
# Empty settings match nothing; private watermark text is not shipped.
WATERMARK_RE = re.compile(os.getenv("OCR_WATERMARK_PATTERN") or r"(?!)", re.IGNORECASE)
WATERMARK_DATE_RE = re.compile(os.getenv("OCR_WATERMARK_DATE_PATTERN") or r"(?!)")
HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
LEADING_UI_RE = re.compile(r"^[\s<←→Q□☐☑$\\(){}\[\]^*·•]+")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
SAFE_TEXT_RE = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fffA-Za-z0-9\s/@>：:，,。.!！？?、（）()\-+%#&]"
)


def extract_all_frames_text(
    video_path: Path,
    output_dir: Path,
    coverage_fps: float = 4.0,
    split_ratio: float = 0.268,
    left_change_threshold: float = 1.5,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError("Video does not exist: {}".format(video_path))
    if coverage_fps <= 0:
        raise ValueError("coverage_fps must be greater than zero")
    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be between zero and one")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_dir / "raw_ocr.jsonl"
    coverage_path = output_dir / "frame_coverage.jsonl"
    existing_records = _load_jsonl(raw_results_path)
    records_by_key = {
        (int(record["window_index"]), str(record["region"])): record
        for record in existing_records
        if "text" in record
    }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open video: {}".format(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or expected_frame_count <= 0:
        capture.release()
        raise RuntimeError("Video metadata is invalid")

    frames_per_window = max(1, round(fps / coverage_fps))
    expected_window_count = int(math.ceil(expected_frame_count / frames_per_window))
    ocr_client = client or PaddleOCRVLClient.from_env()
    coverage_file = coverage_path.open("w", encoding="utf-8", newline="\n")
    current_left_key: Optional[str] = None
    previous_left_signature: Optional[np.ndarray] = None
    decoded_frame_count = 0
    new_ocr_calls = 0
    window_index = 0

    try:
        while True:
            window_frames = []
            for _ in range(frames_per_window):
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = decoded_frame_count
                timestamp_seconds = frame_index / fps
                window_frames.append((frame_index, timestamp_seconds, frame))
                decoded_frame_count += 1
            if not window_frames:
                break

            split_x = max(
                1,
                min(
                    window_frames[0][2].shape[1] - 1,
                    round(window_frames[0][2].shape[1] * split_ratio),
                ),
            )
            main_choice = choose_sharpest_frame(window_frames, split_x, "main")
            main_record = _recognize_window_region(
                window_index,
                "main",
                main_choice,
                split_x,
                frames_dir,
                raw_results_path,
                records_by_key,
                ocr_client,
                expected_window_count,
            )
            if main_record.get("new_call"):
                new_ocr_calls += 1

            left_choice, left_signature = choose_changed_left_frame(
                window_frames,
                split_x,
                previous_left_signature,
                left_change_threshold,
            )
            if left_choice is not None:
                left_record = _recognize_window_region(
                    window_index,
                    "left",
                    left_choice,
                    split_x,
                    frames_dir,
                    raw_results_path,
                    records_by_key,
                    ocr_client,
                    expected_window_count,
                )
                if left_record.get("new_call"):
                    new_ocr_calls += 1
                current_left_key = _record_key(window_index, "left")
                previous_left_signature = left_signature

            main_key = _record_key(window_index, "main")
            for frame_index, timestamp_seconds, _ in window_frames:
                coverage_record = {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp_seconds, 6),
                    "timestamp": format_timestamp(timestamp_seconds),
                    "window_index": window_index,
                    "main_ocr_key": main_key,
                    "left_ocr_key": current_left_key,
                }
                coverage_file.write(
                    json.dumps(coverage_record, ensure_ascii=False) + "\n"
                )
            coverage_file.flush()
            window_index += 1
            if len(window_frames) < frames_per_window:
                break
    finally:
        coverage_file.close()
        capture.release()

    records = sorted(
        records_by_key.values(),
        key=lambda record: (
            int(record["window_index"]),
            0 if record["region"] == "left" else 1,
        ),
    )
    candidates, accepted = build_text_consensus(records)
    _write_candidate_report(output_dir / "text_candidates.tsv", candidates)
    (output_dir / "all_text.txt").write_text(
        "\n".join(accepted) + ("\n" if accepted else ""), encoding="utf-8"
    )
    _write_timeline(output_dir / "timeline.md", video_path, records)

    summary = {
        "video": str(video_path.resolve()),
        "source_fps": round(fps, 6),
        "expected_frame_count": expected_frame_count,
        "decoded_frame_count": decoded_frame_count,
        "coverage_fps": coverage_fps,
        "frames_per_window": frames_per_window,
        "window_count": window_index,
        "ocr_record_count": len(records),
        "new_ocr_calls": new_ocr_calls,
        "candidate_text_line_count": len(candidates),
        "accepted_text_line_count": len(accepted),
        "all_source_frames_mapped": decoded_frame_count == expected_frame_count,
        "watermark_handling": "text-layer filtering; PDF tool reported not_detected for rasterized frame",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def choose_sharpest_frame(
    frames: List[Tuple[int, float, np.ndarray]], split_x: int, region: str
) -> Tuple[int, float, np.ndarray]:
    if not frames:
        raise ValueError("frames cannot be empty")

    def score(item: Tuple[int, float, np.ndarray]) -> float:
        image = item[2]
        crop = image[:, :split_x] if region == "left" else image[:, split_x:]
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(grayscale, cv2.CV_64F).var())

    return max(frames, key=score)


def choose_changed_left_frame(
    frames: List[Tuple[int, float, np.ndarray]],
    split_x: int,
    previous_signature: Optional[np.ndarray],
    threshold: float,
) -> Tuple[Optional[Tuple[int, float, np.ndarray]], Optional[np.ndarray]]:
    candidates = []
    for item in frames:
        left = item[2][:, :split_x]
        signature = _signature(left)
        difference = (
            float("inf")
            if previous_signature is None
            else float(np.mean(cv2.absdiff(signature, previous_signature)))
        )
        if difference >= threshold:
            candidates.append((item, signature))
    if not candidates:
        return None, previous_signature
    chosen = choose_sharpest_frame([item for item, _ in candidates], split_x, "left")
    chosen_signature = next(
        signature for item, signature in candidates if item[0] == chosen[0]
    )
    return chosen, chosen_signature


def clean_watermark_text(line: str) -> str:
    cleaned = html.unescape(line)
    cleaned = HTML_BREAK_RE.sub(" ", cleaned)
    cleaned = LOCATION_TOKEN_RE.sub("", cleaned)
    cleaned = WATERMARK_RE.sub(" ", cleaned)
    cleaned = WATERMARK_DATE_RE.sub(" ", cleaned)
    cleaned = normalize_ocr_line(cleaned)
    return cleaned


def canonical_text(line: str) -> str:
    canonical = LEADING_UI_RE.sub("", line).strip()
    canonical = re.sub(r"\s*([/>：:,，。])\s*", r"\1", canonical)
    return canonical.casefold()


def build_text_consensus(
    records: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    counts: Counter[str] = Counter()
    variants: Dict[str, Counter[str]] = defaultdict(Counter)
    first_seen: Dict[str, str] = {}
    for record in records:
        timestamp = str(record.get("timestamp", ""))
        seen_in_record = set()
        for raw_line in str(record.get("text", "")).splitlines():
            line = clean_watermark_text(raw_line)
            key = canonical_text(line)
            if not line or not key or key in seen_in_record:
                continue
            seen_in_record.add(key)
            counts[key] += 1
            variants[key][line] += 1
            first_seen.setdefault(key, timestamp)

    candidates = []
    accepted = []
    for key, count in counts.most_common():
        text = variants[key].most_common(1)[0][0]
        accepted_flag = is_high_confidence_text(text, count)
        candidate = {
            "count": count,
            "first_seen": first_seen[key],
            "accepted": accepted_flag,
            "text": text,
        }
        candidates.append(candidate)
        if accepted_flag:
            accepted.append(text)
    return candidates, accepted


def is_high_confidence_text(text: str, count: int) -> bool:
    if len(text) > 100:
        return False
    cjk_count = len(CJK_RE.findall(text))
    unsafe_count = len(SAFE_TEXT_RE.sub("", text))
    if unsafe_count > max(2, round(len(text) * 0.15)):
        return False
    if count >= 2 and (cjk_count >= 1 or text.isascii()):
        return True
    if cjk_count >= 2 and cjk_count / max(len(text), 1) >= 0.65:
        return True
    return False


def _recognize_window_region(
    window_index: int,
    region: str,
    choice: Tuple[int, float, np.ndarray],
    split_x: int,
    frames_dir: Path,
    raw_results_path: Path,
    records_by_key: Dict[Tuple[int, str], Dict[str, Any]],
    client: Any,
    total_windows: int,
) -> Dict[str, Any]:
    key = (window_index, region)
    existing = records_by_key.get(key)
    if existing is not None:
        return dict(existing, new_call=False)

    frame_index, timestamp_seconds, frame = choice
    image = frame[:, :split_x] if region == "left" else frame[:, split_x:]
    if region == "left":
        image = cv2.resize(
            image,
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_LANCZOS4,
        )
    frame_name = "w{:04d}_f{:06d}_{}.jpg".format(
        window_index, frame_index, region
    )
    _write_jpeg(frames_dir / frame_name, image)
    started = time.perf_counter()
    text = client.recognize(image)
    elapsed_seconds = round(time.perf_counter() - started, 3)
    record = {
        "key": _record_key(window_index, region),
        "window_index": window_index,
        "representative_frame_index": frame_index,
        "timestamp_seconds": round(timestamp_seconds, 6),
        "timestamp": format_timestamp(timestamp_seconds),
        "region": region,
        "frame": (Path("frames") / frame_name).as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "text": text,
    }
    _append_jsonl(raw_results_path, record)
    records_by_key[key] = record
    print(
        "OCR window {}/{} frame={} {} chars={} elapsed={:.3f}s".format(
            window_index + 1,
            total_windows,
            frame_index,
            region,
            len(text),
            elapsed_seconds,
        ),
        flush=True,
    )
    return dict(record, new_call=True)


def _signature(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(grayscale, (128, 128), interpolation=cv2.INTER_AREA)


def _record_key(window_index: int, region: str) -> str:
    return "w{:04d}:{}".format(window_index, region)


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    encoded, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
    )
    if not encoded:
        raise RuntimeError("Unable to encode frame: {}".format(path))
    buffer.tofile(str(path))


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
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


def _write_candidate_report(path: Path, candidates: List[Dict[str, Any]]) -> None:
    lines = ["count\tfirst_seen\taccepted\ttext"]
    for candidate in candidates:
        text = str(candidate["text"]).replace("\t", " ").replace("\n", " ")
        lines.append(
            "{}\t{}\t{}\t{}".format(
                candidate["count"],
                candidate["first_seen"],
                str(candidate["accepted"]).lower(),
                text,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_timeline(
    path: Path, video_path: Path, records: List[Dict[str, Any]]
) -> None:
    lines = ["# All-frame screen OCR timeline", "", "Video: `{}`".format(video_path.name), ""]
    for record in records:
        lines.extend(
            [
                "## {} - {}".format(record["timestamp"], record["region"]),
                "",
                "Source frame: `{}`".format(record["representative_frame_index"]),
                "",
                "![{}]({})".format(record["timestamp"], record["frame"]),
                "",
                "```text",
                str(record["text"]).replace("```", "'''"),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map every video frame to OCR while retaining dynamic coverage."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coverage-fps", type=float, default=4.0)
    parser.add_argument("--split-ratio", type=float, default=0.268)
    parser.add_argument("--left-change-threshold", type=float, default=1.5)
    args = parser.parse_args()
    summary = extract_all_frames_text(
        args.video,
        args.output_dir,
        coverage_fps=args.coverage_fps,
        split_ratio=args.split_ratio,
        left_change_threshold=args.left_change_threshold,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
