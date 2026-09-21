# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import json
import unittest

import numpy as np

from rapid_videocr.ocr_client import (
    OCRConfigError,
    OCRModelConfig,
    OCRResponseError,
    PaddleOCRVLClient,
)


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


class OCRModelConfigTest(unittest.TestCase):
    def test_builds_config_from_mapping(self):
        config = OCRModelConfig.from_mapping(
            {
                "OCR_MODEL_URL": "https://ocr.example.test/api/v1/chat/completions",
                "OCR_MODEL_NAME": "PaddleOCR-VL-1.6",
                "OCR_MODEL_KEY": "secret-value",
            }
        )

        self.assertEqual(config.model, "PaddleOCR-VL-1.6")
        self.assertEqual(config.timeout, 120.0)

    def test_rejects_external_gpt_model(self):
        with self.assertRaisesRegex(OCRConfigError, "GPT"):
            OCRModelConfig.from_mapping(
                {
                    "OCR_MODEL_URL": "https://ocr.example.test/chat/completions",
                    "OCR_MODEL_NAME": "gpt-5",
                    "OCR_MODEL_KEY": "secret-value",
                }
            )

    def test_rejects_prohibited_host(self):
        with self.assertRaisesRegex(OCRConfigError, "prohibited"):
            OCRModelConfig.from_mapping(
                {
                    "OCR_MODEL_URL": "https://sub2api.gptclubapi.xyz/v1/chat/completions",
                    "OCR_MODEL_NAME": "PaddleOCR-VL-1.6",
                    "OCR_MODEL_KEY": "secret-value",
                }
            )


class PaddleOCRVLClientTest(unittest.TestCase):
    def setUp(self):
        self.config = OCRModelConfig.from_mapping(
            {
                "OCR_MODEL_URL": "https://ocr.example.test/api/v1/chat/completions",
                "OCR_MODEL_NAME": "PaddleOCR-VL-1.6",
                "OCR_MODEL_KEY": "secret-value",
            }
        )

    def test_recognize_sends_image_and_returns_plain_text(self):
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHTTPResponse(
                {"choices": [{"message": {"content": "识别出的字幕"}}]}
            )

        client = PaddleOCRVLClient(self.config, opener=opener)
        result = client.recognize(np.zeros((8, 16, 3), dtype=np.uint8))

        self.assertEqual(result, "识别出的字幕")
        self.assertEqual(captured["url"], self.config.url)
        self.assertEqual(captured["authorization"], "Bearer secret-value")
        self.assertEqual(captured["timeout"], 120.0)
        self.assertEqual(captured["payload"]["model"], "PaddleOCR-VL-1.6")
        self.assertFalse(captured["payload"]["stream"])
        self.assertNotIn("temperature", captured["payload"])
        content = captured["payload"]["messages"][0]["content"]
        self.assertTrue(
            content[0]["image_url"]["url"].startswith("data:image/png;base64,")
        )
        self.assertEqual(content[1], {"type": "text", "text": "OCR:"})

    def test_recognize_unwraps_fenced_json_text(self):
        def opener(request, timeout):
            return FakeHTTPResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '```json\n{"text": "第一行\\n第二行"}\n```'
                            }
                        }
                    ]
                }
            )

        client = PaddleOCRVLClient(self.config, opener=opener)

        self.assertEqual(
            client.recognize(np.zeros((8, 16, 3), dtype=np.uint8)),
            "第一行\n第二行",
        )

    def test_malformed_response_does_not_leak_api_key(self):
        def opener(request, timeout):
            return FakeHTTPResponse({"choices": []})

        client = PaddleOCRVLClient(self.config, opener=opener)

        with self.assertRaises(OCRResponseError) as context:
            client.recognize(np.zeros((8, 16, 3), dtype=np.uint8))

        self.assertNotIn("secret-value", str(context.exception))


if __name__ == "__main__":
    unittest.main()
