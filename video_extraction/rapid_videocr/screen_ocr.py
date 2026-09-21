# -*- encoding: utf-8 -*-
import argparse
import html
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .ocr_client import PaddleOCRVLClient


LOCATION_TOKEN_RE = re.compile(r"<\|LOC_\d+\|>")


class RegionChangeDetector:
    def __init__(self, split_ratio: float = 0.268, threshold: float = 1.5):
        if not 0 < split_ratio < 1:
            raise ValueError("split_ratio must be between zero and one")
        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        self.split_ratio = split_ratio
        self.threshold = threshold
        self._previous: Dict[str, np.ndarray] = {}

    def changed_regions(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame is empty")

        split_x = max(1, min(frame.shape[1] - 1, round(frame.shape[1] * self.split_ratio)))
        regions = {
            "left": frame[:, :split_x],
            "main": frame[:, split_x:],
        }
        changed = {}
        for name, region in regions.items():
            signature = self._signature(region)
            previous = self._previous.get(name)
            score = (
                float("inf")
                if previous is None
                else float(np.mean(cv2.absdiff(signature, previous)))
            )
            if score >= self.threshold:
                changed[name] = self._prepare_region(name, region)
                self._previous[name] = signature
        return changed

    @staticmethod
    def _signature(region: np.ndarray) -> np.ndarray:
        grayscale = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        return cv2.resize(grayscale, (128, 128), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _prepare_region(name: str, region: np.ndarray) -> np.ndarray:
        if name == "left":
            return cv2.resize(
                region,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_LANCZOS4,
            )
        return region.copy()


def extract_screen_text(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 1.0,
    change_threshold: float = 1.5,
    split_ratio: float = 0.268,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    if not video_path.is_file():
        raise FileNotFoundError("Video does not exist: {}".format(video_path))
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_dir / "raw_ocr.jsonl"
    records = _load_records(raw_results_path)
    completed = {
        (round(float(record["timestamp_seconds"]), 3), record["region"])
        for record in records
        if "text" in record
    }

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Unable to open video: {}".format(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError("Video metadata is invalid")

    duration_seconds = frame_count / fps
    sample_count = int(math.ceil(duration_seconds / interval_seconds))
    detector = RegionChangeDetector(split_ratio, change_threshold)
    ocr_client = client or PaddleOCRVLClient.from_env()
    selected_frames = 0
    new_calls = 0

    try:
        for sample_index in range(sample_count):
            timestamp_seconds = round(sample_index * interval_seconds, 3)
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                continue

            changed_regions = detector.changed_regions(frame)
            if not changed_regions:
                continue
            selected_frames += 1

            for region_name, region_image in changed_regions.items():
                record_key = (timestamp_seconds, region_name)
                if record_key in completed:
                    continue

                frame_name = "{}_{}.jpg".format(
                    _filename_timestamp(timestamp_seconds), region_name
                )
                relative_frame_path = Path("frames") / frame_name
                _write_jpeg(frames_dir / frame_name, region_image)

                started = time.perf_counter()
                text = ocr_client.recognize(region_image)
                elapsed_seconds = round(time.perf_counter() - started, 3)
                record = {
                    "timestamp_seconds": timestamp_seconds,
                    "timestamp": format_timestamp(timestamp_seconds),
                    "region": region_name,
                    "frame": relative_frame_path.as_posix(),
                    "elapsed_seconds": elapsed_seconds,
                    "text": text,
                }
                _append_record(raw_results_path, record)
                records.append(record)
                completed.add(record_key)
                new_calls += 1
                print(
                    "OCR {}/{} {} {} chars={} elapsed={:.3f}s".format(
                        sample_index + 1,
                        sample_count,
                        format_timestamp(timestamp_seconds),
                        region_name,
                        len(text),
                        elapsed_seconds,
                    ),
                    flush=True,
                )
    finally:
        capture.release()

    records = sorted(
        records,
        key=lambda item: (
            float(item["timestamp_seconds"]),
            0 if item["region"] == "left" else 1,
        ),
    )
    unique_lines = unique_text_lines(record.get("text", "") for record in records)
    _write_timeline(output_dir / "timeline.md", video_path, records)
    (output_dir / "all_text.txt").write_text(
        "\n".join(unique_lines) + ("\n" if unique_lines else ""),
        encoding="utf-8",
    )
    summary = {
        "video": str(video_path.resolve()),
        "duration_seconds": round(duration_seconds, 3),
        "interval_seconds": interval_seconds,
        "change_threshold": change_threshold,
        "split_ratio": split_ratio,
        "sample_count": sample_count,
        "selected_frame_count": selected_frames,
        "ocr_record_count": len(records),
        "new_ocr_calls": new_calls,
        "unique_text_line_count": len(unique_lines),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def normalize_ocr_line(line: str) -> str:
    normalized = html.unescape(line)
    normalized = LOCATION_TOKEN_RE.sub("", normalized)
    normalized = normalized.replace("\u200b", "").strip()
    if normalized.startswith("```") or normalized == "```":
        return ""
    return " ".join(normalized.split())


def unique_text_lines(texts: Iterable[str]) -> List[str]:
    seen = set()
    lines = []
    for text in texts:
        for raw_line in text.splitlines():
            line = normalize_ocr_line(raw_line)
            key = line.casefold()
            if not line or key in seen:
                continue
            seen.add(key)
            lines.append(line)
    return lines


def format_timestamp(timestamp_seconds: float) -> str:
    milliseconds = round(timestamp_seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return "{:02d}:{:02d}:{:02d}.{:03d}".format(
        hours, minutes, seconds, milliseconds
    )


def _filename_timestamp(timestamp_seconds: float) -> str:
    return "{:010.3f}".format(timestamp_seconds)


def _write_jpeg(path: Path, image: np.ndarray) -> None:
    encoded, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]
    )
    if not encoded:
        raise RuntimeError("Unable to encode frame: {}".format(path))
    buffer.tofile(str(path))


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _append_record(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_file.flush()


def _write_timeline(
    path: Path, video_path: Path, records: List[Dict[str, Any]]
) -> None:
    lines = [
        "# Screen OCR timeline",
        "",
        "Video: `{}`".format(video_path.name),
        "",
    ]
    for record in records:
        lines.extend(
            [
                "## {} - {}".format(record["timestamp"], record["region"]),
                "",
                "![{}]({})".format(record["timestamp"], record["frame"]),
                "",
                "```text",
                str(record.get("text", "")).replace("```", "'''"),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract all visible screen text from a recording."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--change-threshold", type=float, default=1.5)
    parser.add_argument("--split-ratio", type=float, default=0.268)
    args = parser.parse_args()

    summary = extract_screen_text(
        args.video,
        args.output_dir,
        interval_seconds=args.interval,
        change_threshold=args.change_threshold,
        split_ratio=args.split_ratio,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
