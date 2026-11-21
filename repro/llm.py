"""OpenRouter client: one call attempted up to three times, retries after 5 s and 20 s (App. A.1).
Anthropic models get a prompt-cache breakpoint on the last message, so each chat turn reads the earlier
conversation from cache (writes 1.25x, reads 0.1x input price; prefixes under the model's minimum, e.g.
4096 tokens on Opus 4.5 and 1024 on Sonnet 4.6, are not cached). Spend is tallied in USAGE."""

import logging
import threading
import time
from collections import Counter
from collections.abc import Callable
from functools import cache
from typing import Any

from openai import OpenAI

from repro.prompts import MAX_TOKENS
from repro.settings import openrouter_api_key

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1"
RETRY_WAITS_SECONDS = (5, 20)  # a call is attempted once more after each wait
DEFAULT_MAX_TOKENS = 32_000
REASONING_EFFORT = "medium"  # as in the paper
ANTHROPIC_PREFIX = "anthropic/"

Message = dict[str, Any]


@cache
def client() -> OpenAI:
    return OpenAI(base_url=OPENROUTER_URL, api_key=openrouter_api_key())


def retry[T](call: Callable[[], T]) -> T:
    """Run `call`, retrying after each of RETRY_WAITS_SECONDS if it raises."""
    for wait in RETRY_WAITS_SECONDS:
        try:
            return call()
        except Exception as error:  # any API or network failure is worth another try
            logger.warning("call failed, retrying wait_s=%s error=%r", wait, error)
            time.sleep(wait)
    return call()


class UsageTracker:
    """Thread-safe tally of requests, tokens and spend."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def __getitem__(self, key: str) -> float:
        return self._counts[key]

    def clear(self) -> None:
        self._counts.clear()

    def record(self, usage: Any) -> None:
        """Add one response's `usage` object (may be None)."""
        if not usage:
            return
        fields = usage.model_dump()
        prompt_details = fields.get("prompt_tokens_details") or {}
        with self._lock:
            self._counts.update(
                requests=1,
                input=fields.get("prompt_tokens") or 0,
                output=fields.get("completion_tokens") or 0,
                cache_read=prompt_details.get("cached_tokens") or 0,
                cache_write=prompt_details.get("cache_write_tokens") or 0,
            )
            self._counts["cost_usd"] += fields.get("cost") or 0

    def summary(self) -> str:
        u = self._counts
        return (
            f"{u['requests']} requests, {u['input']:,} input tokens "
            f"({u['cache_read']:,} cache read, {u['cache_write']:,} cache write), "
            f"{u['output']:,} output tokens, ${u['cost_usd']:.2f}"
        )


USAGE = UsageTracker()


def usage_summary() -> str:
    return USAGE.summary()


def _with_cache_breakpoint(messages: list[Message]) -> list[Message]:
    """Copy of messages whose last one carries cache_control (5-minute TTL). Each turn's breakpoint is a
    prefix of the next turn's, so the next request reads everything up to it from cache."""
    *head, last = messages
    content = (
        last["content"]
        if isinstance(last["content"], list)
        else [{"type": "text", "text": last["content"]}]
    )
    breakpoint_block = {**content[-1], "cache_control": {"type": "ephemeral"}}
    return [*head, {**last, "content": [*content[:-1], breakpoint_block]}]


def chat(
    model: str, messages: list[Message], reasoning: bool = True
) -> tuple[str, int]:
    """Returns (final text, output tokens incl. reasoning). Reasoning at 'medium', as in the paper."""
    if model.startswith(ANTHROPIC_PREFIX):
        messages = _with_cache_breakpoint(messages)
    reasoning_options = (
        {"effort": REASONING_EFFORT} if reasoning else {"enabled": False}
    )
    response = retry(
        lambda: client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=MAX_TOKENS.get(model, DEFAULT_MAX_TOKENS),
            extra_body={"reasoning": reasoning_options},
        )
    )
    USAGE.record(response.usage)
    text = (response.choices[0].message.content or "").split("</think>")[-1].strip()
    return text, response.usage.completion_tokens if response.usage else 0


def check_models(slugs: list[str]) -> None:
    """Exit unless every slug is served by OpenRouter."""
    available = {model.id for model in client().models.list().data}
    missing = [slug for slug in slugs if slug not in available]
    if missing:
        raise SystemExit(f"not on OpenRouter: {missing}")
