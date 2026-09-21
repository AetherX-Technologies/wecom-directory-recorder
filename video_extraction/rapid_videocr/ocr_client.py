# -*- encoding: utf-8 -*-
import base64
import json
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np


PROHIBITED_HOST = "sub2api.gptclubapi.xyz"
DEFAULT_PROMPT = "OCR:"


class OCRClientError(RuntimeError):
    """Base error for the remote OCR client."""


class OCRConfigError(OCRClientError):
    """Raised when the OCR model configuration is missing or prohibited."""


class OCRRequestError(OCRClientError):
    """Raised when the OCR service request cannot be completed."""


class OCRResponseError(OCRClientError):
    """Raised when the OCR service returns an invalid response."""


@dataclass(frozen=True)
class OCRModelConfig:
    url: str
    model: str
    api_key: str
    timeout: float = 120.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "OCRModelConfig":
        required_keys = ("OCR_MODEL_URL", "OCR_MODEL_NAME", "OCR_MODEL_KEY")
        missing = [key for key in required_keys if not str(values.get(key, "")).strip()]
        if missing:
            raise OCRConfigError(
                "Missing OCR configuration: " + ", ".join(sorted(missing))
            )

        url = str(values["OCR_MODEL_URL"]).strip()
        model = str(values["OCR_MODEL_NAME"]).strip()
        api_key = str(values["OCR_MODEL_KEY"]).strip()
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            raise OCRConfigError("OCR_MODEL_URL must be an absolute HTTP(S) URL")

        host = (parsed_url.hostname or "").lower()
        if host == PROHIBITED_HOST or host.endswith("." + PROHIBITED_HOST):
            raise OCRConfigError("The configured OCR host is prohibited")

        if _is_gpt_model(model):
            raise OCRConfigError("External GPT models are prohibited for OCR")

        timeout_value = values.get("OCR_MODEL_TIMEOUT", 120.0)
        try:
            timeout = float(timeout_value)
        except (TypeError, ValueError) as exc:
            raise OCRConfigError("OCR_MODEL_TIMEOUT must be a number") from exc
        if timeout <= 0:
            raise OCRConfigError("OCR_MODEL_TIMEOUT must be greater than zero")

        return cls(url=url, model=model, api_key=api_key, timeout=timeout)

    @classmethod
    def from_env(
        cls,
        env_path: Optional[Path] = None,
        overrides: Optional[Mapping[str, Any]] = None,
    ) -> "OCRModelConfig":
        values: Dict[str, Any] = {}
        dotenv_path = Path(env_path) if env_path is not None else _find_dotenv()
        if dotenv_path is not None:
            values.update(_read_dotenv(dotenv_path))

        # Process environment variables deliberately override file values.
        values.update(os.environ)
        if overrides:
            values.update(_normalize_overrides(overrides))
        return cls.from_mapping(values)


class PaddleOCRVLClient:
    def __init__(
        self,
        config: OCRModelConfig,
        opener: Callable[..., Any] = urlopen,
    ):
        self.config = config
        self._opener = opener

    @classmethod
    def from_env(
        cls, ocr_params: Optional[Mapping[str, Any]] = None
    ) -> "PaddleOCRVLClient":
        params = dict(ocr_params or {})
        env_path = params.pop("env_path", None)
        config = OCRModelConfig.from_env(env_path=env_path, overrides=params)
        return cls(config)

    def recognize(self, image: np.ndarray) -> str:
        image_url = _encode_image(image)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": DEFAULT_PROMPT},
                    ],
                }
            ],
            "stream": False,
        }
        request = Request(
            self.config.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.config.api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "RapidVideOCR/PaddleOCR-VL",
            },
            method="POST",
        )

        try:
            with self._opener(request, timeout=self.config.timeout) as response:
                response_data = response.read()
        except HTTPError as exc:
            raise OCRRequestError(
                "OCR service returned HTTP status {}".format(exc.code)
            ) from exc
        except (URLError, TimeoutError, socket.timeout) as exc:
            raise OCRRequestError("OCR service request failed") from exc

        try:
            decoded = json.loads(response_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OCRResponseError("OCR service returned invalid JSON") from exc

        return _extract_text(decoded)


def _is_gpt_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith("chatgpt") or bool(
        re.search(r"(^|[/_.:-])gpt([/_.:-]|$)", normalized)
    )


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


def _normalize_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
    key_map = {
        "url": "OCR_MODEL_URL",
        "model": "OCR_MODEL_NAME",
        "api_key": "OCR_MODEL_KEY",
        "timeout": "OCR_MODEL_TIMEOUT",
        "OCR_MODEL_URL": "OCR_MODEL_URL",
        "OCR_MODEL_NAME": "OCR_MODEL_NAME",
        "OCR_MODEL_KEY": "OCR_MODEL_KEY",
        "OCR_MODEL_TIMEOUT": "OCR_MODEL_TIMEOUT",
    }
    unknown = sorted(set(overrides) - set(key_map))
    if unknown:
        raise OCRConfigError(
            "Unsupported ocr_params for PaddleOCR-VL: " + ", ".join(unknown)
        )
    return {key_map[key]: value for key, value in overrides.items()}


def _encode_image(image: np.ndarray) -> str:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise OCRRequestError("OCR input image is empty")
    encoded, buffer = cv2.imencode(".png", image)
    if not encoded:
        raise OCRRequestError("OCR input image could not be encoded")
    base64_data = base64.b64encode(buffer.tobytes()).decode("ascii")
    return "data:image/png;base64," + base64_data


def _extract_text(payload: Any) -> str:
    try:
        choices = payload["choices"]
        message = choices[0]["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OCRResponseError("OCR response does not contain message content") from exc

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                text_parts.append(part["text"])
        content = "\n".join(text_parts)

    if not isinstance(content, str):
        raise OCRResponseError("OCR response message content is not text")

    text = _strip_code_fence(content.strip())
    try:
        structured = json.loads(text)
    except json.JSONDecodeError:
        return text

    if isinstance(structured, dict):
        for key in ("text", "content", "result", "output"):
            value = structured.get(key)
            if isinstance(value, str):
                return value.strip()
    return text


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].strip().startswith("```"):
        if lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text
