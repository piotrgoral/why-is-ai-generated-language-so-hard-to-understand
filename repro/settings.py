"""Environment-backed settings: the one place that lists the variables the pipeline needs."""

import os


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"environment variable {name} is not set")
    return value


def openrouter_api_key() -> str:
    return _required("OPENROUTER_API_KEY")


def jev_api_key() -> str:
    return _required("JEV_API_KEY")
