"""serve/test_model_card_sampling.py - a request that sends no sampling gets the model card's, not greedy (mock engine).

Qwen's "Best Practices": thinking mode temperature 1.0 / top_p 0.95 / top_k 20, non-thinking 0.7 / 0.8 / 20 with
presence_penalty 1.5.  The request's own fields win, then the Chat settings shared with apps, then the config's
`sampling` block, then the card; temperature 0 asked for is still greedy.

    python -m unittest serve.test_model_card_sampling -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, model_card_sampling, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Recording(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class ModelCardSampling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = Recording(cls.tok, "<think>\nhm\n</think>\n\nok", max_context=8192)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def setUp(self):
        self.engine.three_level = False
        self.svc.model_arch = "qwen4exp"
        self.svc.sampling_defaults = {}
        self.svc.shared = {}
        self.svc.config_effort = None

    def tearDown(self):
        self.svc.model_arch = None
        self.engine.three_level = False

    def keys(self, **body):
        body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, **body}
        self.engine.sampling = {}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
        return StrataEngine.sampling_keys(self.engine.sampling)

    def test_tables(self):
        self.assertEqual(model_card_sampling("qwen4exp", True), {"temperature": 1.0, "top_p": 0.95, "top_k": 20})
        self.assertEqual(model_card_sampling("qwen4exp", False),
                         {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5})
        self.assertEqual(model_card_sampling("qwen35moe", True)["presence_penalty"], 1.5)
        self.assertEqual(model_card_sampling(None, True), {})
        self.assertEqual(model_card_sampling("unknown", False), {})

    def test_thinking_default(self):
        k = self.keys()
        self.assertIn("temperature=1.0", k)
        self.assertIn("top_p=0.95", k)
        self.assertIn("top_k=20", k)
        self.assertNotIn("penalty_present", k)

    def test_non_thinking_default(self):
        k = self.keys(reasoning_effort="none")
        self.assertIn("temperature=0.7", k)
        self.assertIn("top_p=0.8", k)
        self.assertIn("penalty_present=1.5", k)

    def test_request_wins_and_zero_is_greedy(self):
        k = self.keys(temperature=0)
        self.assertNotIn("temperature", k)
        self.assertIn("top_k=20", k)
        k = self.keys(temperature=0.3, top_p=0.5)
        self.assertIn("temperature=0.3", k)
        self.assertIn("top_p=0.5", k)

    def test_config_block_and_shared_settings_win_over_the_card(self):
        self.svc.sampling_defaults = {"temperature": 0.6, "top_k": 40}
        k = self.keys()
        self.assertIn("temperature=0.6", k)
        self.assertIn("top_k=40", k)
        self.assertIn("top_p=0.95", k)
        self.svc.shared = {"temperature": 0.2}
        self.assertIn("temperature=0.2", self.keys())
        self.svc.sampling_defaults = {"temperature": 0}
        self.svc.shared = {}
        self.assertNotIn("temperature", self.keys())     # a config that asks for greedy gets it

    def test_no_family_no_defaults(self):
        self.svc.model_arch = None
        self.assertEqual(self.keys(), "")

    def test_three_level_defaults_stay_greedy_and_reject_sampling_before_streaming(self):
        self.engine.three_level = True
        for arch in ("qwen4exp", "qwen35moe"):
            self.svc.model_arch = arch
            for thinking in ("none", "high"):
                with self.subTest(arch=arch, thinking=thinking):
                    self.assertEqual(self.keys(reasoning_effort=thinking), "")
        for params in ({"temperature": 0.7}, {"presence_penalty": 1.5}, {"repeat_penalty": 1.1},
                       {"frequency_penalty": 0.2}, {"logprobs": True}, {"n": 2},
                       {"reasoning_budget_tokens": 4}, {"thinking_budget_tokens": 4}):
            with self.subTest(params=params), self.assertRaises(urllib.error.HTTPError) as caught:
                self.keys(stream=True, **params)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("three-level", caught.exception.read().decode())
            self.assertEqual(self.engine.sampling, {})
        self.svc.shared = {"temperature": 0.8}
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.keys()
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.keys(temperature=0), "")
        with self.assertRaisesRegex(ValueError, "three-level"):
            self.svc.set_shared({"temperature": 0.5})
        self.assertEqual(self.svc.set_shared({"temperature": 0}), {"temperature": 0})

    def test_three_level_rejects_tool_images_before_omitting_unreadable_images(self):
        self.engine.three_level = True
        messages = [{"role": "tool", "tool_call_id": "x", "content": [
            {"type": "image_url", "image_url": {"url": "https://invalid.example/image.png"}}]}]
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.keys(messages=messages, stream=True)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("three-level", caught.exception.read().decode())
        self.assertEqual(self.engine.sampling, {})

    def test_three_level_sampling_rejection_is_shared_by_all_http_apis(self):
        self.engine.three_level = True
        messages = [{"role": "user", "content": "hi"}]
        for path, body in (("/v1/chat/completions", {"messages": messages, "max_tokens": 8}),
                           ("/v1/messages", {"messages": messages, "max_tokens": 8}),
                           ("/v1/responses", {"input": "hi", "max_output_tokens": 8})):
            with self.subTest(path=path):
                req = urllib.request.Request(self.base + path,
                    data=json.dumps({**body, "model": "m", "temperature": 0.7, "stream": True}).encode(),
                    headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                self.assertEqual(caught.exception.code, 400)
                self.assertIn("three-level", caught.exception.read().decode())


if __name__ == "__main__":
    unittest.main()
