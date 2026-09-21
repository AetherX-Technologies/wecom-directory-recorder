# -*- encoding: utf-8 -*-
import base64
import json
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


REQUIRED_MODEL = "Qwen3.6-35B-A3B"
PROHIBITED_HOST = "sub2api.gptclubapi.xyz"


class VisionClientError(RuntimeError):
    """Base error for the remote vision client."""


class VisionConfigError(VisionClientError):
    """Raised when the configured vision service is missing or prohibited."""


class VisionRequestError(VisionClientError):
    """Raised when the vision request cannot be completed."""


class VisionResponseError(VisionClientError):
    """Raised when the vision service returns an invalid response."""


@dataclass(frozen=True)
class VisionModelConfig:
    url: str
    model: str
    api_key: str
    timeout: float = 180.0
    max_retries: int = 2

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "VisionModelConfig":
        required_keys = ("VL_MODEL_URL", "VL_MODEL_NAME", "VL_MODEL_KEY")
        missing = [key for key in required_keys if not str(values.get(key, "")).strip()]
        if missing:
            raise VisionConfigError(
                "Missing vision configuration: " + ", ".join(sorted(missing))
            )

        url = str(values["VL_MODEL_URL"]).strip()
        model = str(values["VL_MODEL_NAME"]).strip()
        api_key = str(values["VL_MODEL_KEY"]).strip()
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise VisionConfigError("VL_MODEL_URL must be an absolute HTTP(S) URL")

        host = (parsed_url.hostname or "").lower()
        if host == PROHIBITED_HOST or host.endswith("." + PROHIBITED_HOST):
            raise VisionConfigError("The configured vision host is prohibited")
        if _is_gpt_model(model):
            raise VisionConfigError("External GPT models are prohibited")
        if model != REQUIRED_MODEL:
            raise VisionConfigError(
                "This extraction requires model {}".format(REQUIRED_MODEL)
            )

        timeout = _positive_float(values.get("VL_MODEL_TIMEOUT", 180.0), "timeout")
        max_retries = _non_negative_int(
            values.get("VL_MODEL_MAX_RETRIES", 2), "max retries"
        )
        return cls(
            url=url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "VisionModelConfig":
        values: Dict[str, Any] = {}
        dotenv_path = Path(env_path) if env_path is not None else _find_dotenv()
        if dotenv_path is not None:
            values.update(_read_dotenv(dotenv_path))
        values.update(os.environ)
        return cls.from_mapping(values)


@dataclass(frozen=True)
class VisionModelResponse:
    content: str
    raw: Dict[str, Any]


class QwenVisionClient:
    def __init__(
        self,
        config: VisionModelConfig,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.config = config
        self._opener = opener
        self._sleeper = sleeper

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> "QwenVisionClient":
        return cls(VisionModelConfig.from_env(env_path=env_path))

    def analyze(
        self,
        image: np.ndarray,
        prompt: str,
        max_tokens: int = 2048,
    ) -> VisionModelResponse:
        return self.analyze_images([image], prompt, max_tokens=max_tokens)

    def analyze_images(
        self,
        images: Sequence[np.ndarray],
        prompt: str,
        max_tokens: int = 4096,
    ) -> VisionModelResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise VisionRequestError("Vision prompt cannot be empty")
        if not images:
            raise VisionRequestError("At least one vision image is required")
        if max_tokens <= 0:
            raise VisionRequestError("max_tokens must be greater than zero")

        content = [
            {
                "type": "image_url",
                "image_url": {"url": _encode_image(image)},
            }
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = Request(
            self.config.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json; charset=utf-8",
                "Accept": "application/json",
                "User-Agent": "RapidVideOCR/QwenVision",
            },
            method="POST",
        )

        response_data = self._send_with_retry(request)
        try:
            decoded = json.loads(response_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise VisionResponseError("Vision service returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise VisionResponseError("Vision service response must be an object")
        return VisionModelResponse(content=_extract_content(decoded), raw=decoded)

    def _send_with_retry(self, request: Request) -> bytes:
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                with self._opener(request, timeout=self.config.timeout) as response:
                    return response.read()
            except HTTPError as exc:
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt + 1 >= attempts:
                    raise VisionRequestError(
                        "Vision service returned HTTP status {}".format(exc.code)
                    ) from exc
            except (URLError, TimeoutError, socket.timeout) as exc:
                if attempt + 1 >= attempts:
                    raise VisionRequestError("Vision service request failed") from exc
            self._sleeper(float(2**attempt))
        raise VisionRequestError("Vision service request failed")


def _extract_content(payload: Mapping[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionResponseError(
            "Vision response does not contain message content"
        ) from exc
    if not isinstance(content, str):
        raise VisionResponseError("Vision response message content is not text")
    return content.strip()


def _encode_image(image: np.ndarray) -> str:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise VisionRequestError("Vision input image is empty")
    encoded, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
    if not encoded:
        raise VisionRequestError("Vision input image could not be encoded")
    data = base64.b64encode(buffer.tobytes()).decode("ascii")
    return "data:image/jpeg;base64," + data


def _find_dotenv() -> Optional[Path]:
    seen = set()
    for start_dir in (Path.cwd(), Path(__file__).resolve().parent):
        for directory in (start_dir,) + tuple(start_dir.parents):
            candidate = directory / ".env"
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_file():
                return candidate
    return None


def _read_dotenv(path: Path) -> Dict[str, str]:
    values = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _is_gpt_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("chatgpt") or bool(
        re.search(r"(^|[/_.:-])gpt([/_.:-]|$)", normalized)
    )


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise VisionConfigError("Vision {} must be a number".format(label)) from exc
    if result <= 0:
        raise VisionConfigError("Vision {} must be greater than zero".format(label))
    return result


def _non_negative_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise VisionConfigError("Vision {} must be an integer".format(label)) from exc
    if result < 0:
        raise VisionConfigError("Vision {} cannot be negative".format(label))
    return result
