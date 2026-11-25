"""Minimal LLM clients for Gemini and Groq over plain HTTPS.

One `chat()` entry point for both providers with:
- retries with backoff on 429/5xx (honours Retry-After),
- a client-side request spacing per provider (free tiers are RPM-limited),
- an optional on-disk cache keyed by the full request (replays are free and identical),
- an optional JSONL audit log of every prompt and raw response,
- token accounting and a list-price cost estimate.

Plain httpx instead of vendor SDKs keeps the serverless bundle small.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

# Paid-tier list prices in USD per 1M tokens (input, output), used only to report
# what a run WOULD cost; free-tier runs cost $0. Checked Nov 2026; edit if they move.
PRICES = {
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}
MIN_INTERVAL = {"gemini": 4.2, "groq": 2.1}  # seconds between calls (15 RPM / ~30 RPM)


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def cost_usd(self) -> float:
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (self.prompt_tokens * pin + self.completion_tokens * pout) / 1e6


class LLMError(RuntimeError):
    pass


_last_call: dict[str, float] = {}
_lock = threading.Lock()


def _space(provider: str) -> None:
    with _lock:
        wait = _last_call.get(provider, 0) + MIN_INTERVAL.get(provider, 0) - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[provider] = time.monotonic()


def _key(name: str) -> str:
    v = os.environ.get(name)
    if not v and Path(".env").exists():  # local runs: read .env without exporting it
        for line in Path(".env").read_text(encoding="utf-8").splitlines():
            if line.startswith(f"{name}="):
                v = line.split("=", 1)[1].split("#", 1)[0].strip()
    if not v:
        raise LLMError(f"{name} is not set (copy .env.example to .env)")
    return v


def _gemini(model, system, user, temperature, max_tokens, json_mode, client):
    body = {
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if json_mode:
        body["generationConfig"]["responseMimeType"] = "application/json"
    if "3.5" in model or "3.1" in model:
        body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": "minimal"}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    r = client.post(url, json=body, headers={"x-goog-api-key": _key("GEMINI_API_KEY")})
    if r.status_code != 200:
        return r, None
    data = r.json()
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought"))
    u = data.get("usageMetadata", {})
    completion = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
    return r, (text, u.get("promptTokenCount", 0), completion, data)


def _groq(model, system, user, temperature, max_tokens, json_mode, client):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    r = client.post(
        "https://api.groq.com/openai/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {_key('GROQ_API_KEY')}"},
    )
    if r.status_code != 200:
        return r, None
    data = r.json()
    u = data.get("usage", {})
    return r, (data["choices"][0]["message"].get("content") or "", u.get("prompt_tokens", 0), u.get("completion_tokens", 0), data)


PROVIDERS = {"gemini": _gemini, "groq": _groq}


def chat(
    provider: str,
    model: str,
    user: str,
    system: str = "",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False,
    cache_dir: str | None = ".cache/llm",
    sample: int = 0,
    audit_path: str | None = None,
    tag: str = "",
    max_retries: int = 6,
) -> LLMResult:
    """Call a model. `sample` distinguishes deliberate repeats (test-retest) in the cache."""
    if provider not in PROVIDERS:
        raise LLMError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    req = dict(provider=provider, model=model, system=system, user=user, temperature=temperature,
               max_tokens=max_tokens, json_mode=json_mode, sample=sample)
    digest = hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest()
    cpath = Path(cache_dir) / provider / f"{digest}.json" if cache_dir else None

    if cpath and cpath.exists():
        res = LLMResult(**json.loads(cpath.read_text(encoding="utf-8")))
        res.cached = True
    else:
        delay = 5.0
        with httpx.Client(timeout=90) as client:
            for attempt in range(max_retries + 1):
                _space(provider)
                t0 = time.perf_counter()
                try:
                    r, out = PROVIDERS[provider](model, system, user, temperature, max_tokens, json_mode, client)
                except httpx.HTTPError as e:  # network blip
                    if attempt == max_retries:
                        raise LLMError(f"{provider} network error: {e}") from e
                    time.sleep(delay)
                    delay *= 2
                    continue
                ms = (time.perf_counter() - t0) * 1000
                if out is not None:
                    text, pt, ct, raw = out
                    res = LLMResult(text, provider, model, pt, ct, ms, False, raw)
                    break
                if r.status_code in (429, 500, 502, 503, 504) and attempt < max_retries:
                    wait = float(r.headers.get("retry-after", delay))
                    time.sleep(min(max(wait, delay), 90))
                    delay = min(delay * 2, 90)
                    continue
                raise LLMError(f"{provider} HTTP {r.status_code}: {r.text[:500]}")
        if cpath:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            cpath.write_text(json.dumps(asdict(res)), encoding="utf-8")

    if audit_path:
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "tag": tag, "request_hash": digest, **req, "response_text": res.text,
               "prompt_tokens": res.prompt_tokens, "completion_tokens": res.completion_tokens,
               "latency_ms": round(res.latency_ms, 1), "cached": res.cached, "cost_usd": round(res.cost_usd, 6)}
        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return res
