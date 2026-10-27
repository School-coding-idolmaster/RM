"""Offline tests for the LLM backends (no network)."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from robomorph.evolution import llm, loop


class LLMTests(unittest.TestCase):
    def setUp(self):
        llm.USAGE.update(calls=0, input_tokens=0, output_tokens=0)

    def test_anthropic_path_sends_effort_and_records_usage(self):
        reply = Mock(status_code=200)
        reply.json.return_value = {"content": [{"type": "thinking", "thinking": "..."},
                                               {"type": "text", "text": "Step 1: r0) H-T"}],
                                   "usage": {"input_tokens": 120, "output_tokens": 340}}
        env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test", "LLM_EFFORT": "high"}
        with patch.dict(os.environ, env), patch("httpx.post", return_value=reply) as post:
            text = loop.generate_grammar(None, "sys", "user", "claude-haiku-5-5")
        self.assertEqual(text, "Step 1: r0) H-T")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["output_config"], {"effort": "high"})
        self.assertEqual(body["system"], "sys")
        self.assertEqual(llm.USAGE, {"calls": 1, "input_tokens": 120, "output_tokens": 340})

    def test_anthropic_error_is_raised(self):
        reply = Mock(status_code=400, text="bad request")
        env = {"LLM_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "test"}
        with patch.dict(os.environ, env), patch("httpx.post", return_value=reply):
            with self.assertRaisesRegex(RuntimeError, "400"):
                loop.generate_grammar(None, "sys", "user", "claude-haiku-5-5")

    def test_openai_path_records_usage(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20))
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
            self.assertEqual(loop.generate_grammar(client, "s", "u", "deepseek-chat"), "ok")
        self.assertEqual(llm.USAGE["input_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
