"""Jev client (TypeSafe System One API), the external classifier of App. B.1, pinned to version 1.13.0."""

from functools import cache
from typing import Any

import httpx

from repro.llm import retry
from repro.prompts import JEV_QUESTIONS
from repro.settings import jev_api_key

JEV_MODEL = "jev-1.13.0"
URL = "https://api.typesafe.ai/v1/systemone"
TIMEOUT_SECONDS = 60


@cache
def client() -> httpx.Client:
    return httpx.Client(
        headers={"Authorization": f"Bearer {jev_api_key()}"},
        timeout=TIMEOUT_SECONDS,
    )


def classify(feature: str, text: str) -> float:
    """Probability that the marked candidate in `text` shows `feature` ('denominal' or 'inanimate')."""
    payload = {
        "model": JEV_MODEL,
        "state": text,
        "questions": {
            feature: {"type": "noul", "instructions": JEV_QUESTIONS[feature]}
        },
    }

    def call() -> dict[str, Any]:
        response = client().post(URL, json=payload)
        response.raise_for_status()  # 429 / 529 -> retried
        return response.json()

    body = retry(call)
    if body["model"] != JEV_MODEL:
        raise RuntimeError(f"answered by {body['model']}, expected {JEV_MODEL}")
    return float(body["answers"][feature]["noul"])
