"""Jev's judgement of denominal-verb and inanimate-subject candidates, cached in data/jev.jsonl (App. B.1)."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repro import jev
from repro.analysis.syntax import Candidate
from repro.jsonl import append_jsonl, read_jsonl
from repro.paths import DATA

logger = logging.getLogger(__name__)

JUDGE_CACHE = DATA / "jev.jsonl"
ACCEPT_THRESHOLD = 0.5  # a candidate counts if Jev's probability exceeds this
DEFAULT_WORKERS = 16
CHUNK_SIZE = 500  # candidates judged and saved together


def judge_candidates(
    candidates: list[Candidate],
    workers: int = DEFAULT_WORKERS,
    chunk_size: int = CHUNK_SIZE,
    cache_path: Path = JUDGE_CACHE,
) -> dict[Candidate, float]:
    """Jev's probability that each candidate shows its feature; only uncached ones are sent."""
    cache = {(row["feature"], row["text"]): row["p"] for row in read_jsonl(cache_path)}
    todo = sorted(set(candidates) - cache.keys())
    with ThreadPoolExecutor(workers) as pool:
        for start in range(0, len(todo), chunk_size):
            chunk = todo[start : start + chunk_size]
            probabilities = pool.map(lambda c: jev.classify(*c), chunk)
            rows = [
                {"feature": feature, "text": text, "p": p}
                for (feature, text), p in zip(chunk, probabilities)
            ]
            append_jsonl(cache_path, rows)
            cache.update(((row["feature"], row["text"]), row["p"]) for row in rows)
            logger.info("jev judged=%d total=%d", start + len(chunk), len(todo))
    return cache


def accepted_count(
    feature: str, candidates: list, judged: dict[Candidate, float]
) -> int:
    """Number of `candidates` (feature, text pairs, possibly as lists) judged to show `feature`."""
    return sum(
        judged.get(tuple(c), 0) > ACCEPT_THRESHOLD
        for c in candidates
        if c[0] == feature
    )
