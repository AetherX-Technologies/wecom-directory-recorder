# Public integration modification: synthetic fixtures or removal of private defaults; see THIRD_PARTY_NOTICES.md.
# -*- encoding: utf-8 -*-
import json
import unittest

import numpy as np

from rapid_videocr.vision_client import (
    QwenVisionClient,
    VisionConfigError,
    VisionModelConfig,
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


class VisionModelConfigTest(unittest.TestCase):
    def test_requires_the_selected_qwen_model(self):
        with self.assertRaisesRegex(VisionConfigError, "Qwen3.6"):
            VisionModelConfig.from_mapping(
                {
                    "VL_MODEL_URL": "https://vision.example.test/chat/completions",
                    "VL_MODEL_NAME": "DeepSeek-VL",
                    "VL_MODEL_KEY": "secret-value",
                }
            )

    def test_rejects_prohibited_host(self):
        with self.assertRaisesRegex(VisionConfigError, "prohibited"):
            VisionModelConfig.from_mapping(
                {
                    "VL_MODEL_URL": "https://sub2api.gptclubapi.xyz/v1",
                    "VL_MODEL_NAME": "Qwen3.6-35B-A3B",
                    "VL_MODEL_KEY": "secret-value",
                }
            )


class QwenVisionClientTest(unittest.TestCase):
    def test_sends_image_prompt_and_disables_thinking(self):
        captured = {}

        def opener(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeHTTPResponse(
                {
                    "choices": [
                        {"message": {"content": '{"people": []}'}}
                    ]
                }
            )

        config = VisionModelConfig.from_mapping(
            {
                "VL_MODEL_URL": "https://vision.example.test/chat/completions",
                "VL_MODEL_NAME": "Qwen3.6-35B-A3B",
                "VL_MODEL_KEY": "secret-value",
                "VL_MODEL_MAX_RETRIES": "0",
            }
        )
        response = QwenVisionClient(config, opener=opener).analyze(
            np.zeros((8, 16, 3), dtype=np.uint8), "extract"
        )

        self.assertEqual(response.content, '{"people": []}')
        self.assertEqual(captured["timeout"], 180.0)
        payload = captured["payload"]
        self.assertEqual(payload["model"], "Qwen3.6-35B-A3B")
        self.assertEqual(payload["temperature"], 0)
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
        content = payload["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(content[1]["text"], "extract")

    def test_sends_multiple_images_before_prompt(self):
        captured = {}

        def opener(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeHTTPResponse(
                {"choices": [{"message": {"content": '{"frames": []}'}}]}
            )

        config = VisionModelConfig.from_mapping(
            {
                "VL_MODEL_URL": "https://vision.example.test/chat/completions",
                "VL_MODEL_NAME": "Qwen3.6-35B-A3B",
                "VL_MODEL_KEY": "secret-value",
                "VL_MODEL_MAX_RETRIES": "0",
            }
        )
        client = QwenVisionClient(config, opener=opener)
        client.analyze_images(
            [
                np.zeros((8, 16, 3), dtype=np.uint8),
                np.zeros((8, 16, 3), dtype=np.uint8),
            ],
            "extract all",
        )

        content = captured["payload"]["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["image_url", "image_url", "text"])


if __name__ == "__main__":
    unittest.main()
