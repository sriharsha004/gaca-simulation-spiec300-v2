"""
Thin OpenAI client wrapper: real API calls (matches the manuscript's stated
GPT-4o-mini backbone), retry/backoff, JSON-mode parsing, and a MOCK_MODE for
offline development/testing of the harness without burning API credits or
requiring a key. Every experiment script in this repo defaults to MOCK_MODE
when no API key is present so the *pipeline mechanics* (graph construction,
retrieval, rule checking, checkpointing, stats) can be verified honestly
before spending real API budget -- mock outputs are never written into the
final results files used for the paper (see run_conditions.py guard).
"""

from __future__ import annotations
import json
import os
import time
import random

MOCK_MODE = os.environ.get("OPENAI_API_KEY") is None

if not MOCK_MODE:
    from openai import OpenAI
    _client = OpenAI()
else:
    _client = None

MODEL = os.environ.get("GACA_MODEL", "gpt-4o-mini")
TEMPERATURE = 0.2
MAX_TOKENS = 500
MAX_RETRIES = 5


def _token_param_name(model: str) -> str:
    """Newer OpenAI models (o1/o3/o4 reasoning family, gpt-5.x) reject the
    legacy `max_tokens` chat-completions parameter and require
    `max_completion_tokens` instead. Older models (gpt-4o family and
    earlier) still require `max_tokens`. Detected by model name prefix so
    swapping GACA_MODEL doesn't silently break the call -- confirmed against
    a real API smoke test on gpt-5.4-mini/gpt-5.4-nano, which need
    max_completion_tokens."""
    m = model.lower()
    if m.startswith(("o1", "o3", "o4", "gpt-5")):
        return "max_completion_tokens"
    return "max_tokens"


def _mock_response(user_prompt: str) -> str:
    """Deterministic, clearly-fake stand-in used ONLY when no API key is set,
    so the rest of the pipeline can be exercised end to end. Real experiments
    MUST run with MOCK_MODE=False (i.e. a real OPENAI_API_KEY set)."""
    rng = random.Random(hash(user_prompt) % (2**32))
    return json.dumps({
        "gap_category_id": rng.randint(1, 24),
        "gap_category_name": "MOCK_PLACEHOLDER",
        "confidence": round(rng.uniform(0.4, 0.9), 2),
        "recommendation": "MOCK_PLACEHOLDER recommendation text for pipeline testing only.",
        "graph_path_explanation": "MOCK_PLACEHOLDER path.",
    })


def chat_json(system: str, user: str, *, model: str = MODEL, temperature: float = TEMPERATURE) -> dict:
    """Calls the chat model with JSON-object response format, retries on
    transient errors, and returns the parsed dict plus timing info attached
    under '_latency_ms'."""
    if MOCK_MODE:
        t0 = time.perf_counter()
        raw = _mock_response(user)
        latency_ms = (time.perf_counter() - t0) * 1000 + rng_jitter()
        out = json.loads(raw)
        out["_latency_ms"] = latency_ms
        out["_mock"] = True
        return out

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.perf_counter()
            system_msg = system
            user_msg = user
            if "json" not in f"{system_msg}\n{user_msg}".lower():
                system_msg = f"{system_msg}\nReturn valid json."
            kwargs = {
                "model": model,
                "temperature": temperature,
                _token_param_name(model): MAX_TOKENS,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            }
            resp = _client.chat.completions.create(**kwargs)
            latency_ms = (time.perf_counter() - t0) * 1000
            content = resp.choices[0].message.content
            out = json.loads(content)
            out["_latency_ms"] = latency_ms
            out["_mock"] = False
            out["_usage"] = resp.usage.model_dump() if resp.usage else {}
            return out
        except Exception as e:  # noqa: BLE001 - broad on purpose, we retry+log
            last_err = e
            sleep_s = min(2 ** attempt, 20) + random.random()
            time.sleep(sleep_s)
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries: {last_err}")


def rng_jitter() -> float:
    return random.uniform(1, 5)


if __name__ == "__main__":
    print("MOCK_MODE =", MOCK_MODE)
    out = chat_json("You are a test.", "Return {\"gap_category_id\": 1}")
    print(out)
