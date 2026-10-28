"""LLM backends for design generation.

LLM_PROVIDER=openai (default): the original RoboMorph path, OpenAI-compatible chat completions
    (OpenAI, or DeepSeek via OPENAI_BASE_URL).
LLM_PROVIDER=anthropic: native Claude Messages API over httpx (no extra dependency), so that
    output_config.effort can be set; the OpenAI compatibility layer silently ignores it.
    Env: ANTHROPIC_API_KEY, LLM_EFFORT (low|medium|high|xhigh|max, default high),
    LLM_MAX_TOKENS (default 16000; thinking tokens count toward it).

Token usage of every call is accumulated in USAGE and written per generation by the loop.
"""
from __future__ import annotations

import os
import time

USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def provider() -> str:
    return os.environ.get("LLM_PROVIDER", "openai").strip().lower()


def _record(inp, out):
    USAGE["calls"] += 1
    USAGE["input_tokens"] += int(inp or 0)
    USAGE["output_tokens"] += int(out or 0)


def anthropic_chat(system_prompt: str, user_prompt: str, model_name: str, retries: int = 5) -> str:
    return anthropic_chat_full(system_prompt, user_prompt, model_name, retries)[0]


def anthropic_chat_full(system_prompt: str, user_prompt: str, model_name: str, retries: int = 5) -> tuple[str, dict]:
    """Same as anthropic_chat, also returns this call's usage (thread-safe per call; hyblock/search.py)."""
    import httpx
    body = {
        "model": model_name,
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "16000")),
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "output_config": {"effort": os.environ.get("LLM_EFFORT", "high")},
    }
    headers = {
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    for attempt in range(retries):
        r = httpx.post(ANTHROPIC_URL, json=body, headers=headers, timeout=600)
        if r.status_code in (429, 500, 502, 503, 529) and attempt < retries - 1:
            time.sleep(min(60, 5 * 2 ** attempt))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"Claude API error {r.status_code}: {r.text[:500]}")
        data = r.json()
        usage = data.get("usage", {})
        _record(usage.get("input_tokens"), usage.get("output_tokens"))
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        return text, {"input_tokens": int(usage.get("input_tokens") or 0),
                      "output_tokens": int(usage.get("output_tokens") or 0),
                      "model": data.get("model", model_name), "stop_reason": data.get("stop_reason")}
    raise RuntimeError("Claude API: retries exhausted")


def openai_chat(client, system_prompt: str, user_prompt: str, model_name: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    usage = getattr(response, "usage", None)
    if usage is not None:
        try:
            _record(getattr(usage, "prompt_tokens", 0), getattr(usage, "completion_tokens", 0))
        except (TypeError, ValueError):
            pass
    return response.choices[0].message.content
