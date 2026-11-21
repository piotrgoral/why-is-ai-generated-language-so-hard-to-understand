"""Reference corpora for the n-gram features (App. B.2): ~500k random words each.

Chats are compared with OpenSubtitles, essays with Wikipedia. The samples are cached in data/."""

import itertools
import logging
import math
import random
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from repro.paths import DATA
from repro.text import clean, sentences

logger = logging.getLogger(__name__)

# n-gram reference per setting
REFERENCE_FOR_SETTING = {"chat": "opensubtitles", "essay": "wikipedia"}
DEFAULT_OPENSUBTITLES = DATA / "opensubtitles" / "en.txt.gz"
SAMPLE_SEED = 0

WIKIPEDIA_DUMP = "20231101.en"
WIKIPEDIA_SHUFFLE_BUFFER = 10_000
MIN_WIKIPEDIA_BLOCK_WORDS = 8
SUBTITLE_WORDS_PER_LINE = 3  # oversample n_words // 3 lines, trim afterwards


def sample_wikipedia(n_words: int) -> list[str]:
    """Sentences from shuffled Wikipedia articles until `n_words` words are collected."""
    from datasets import load_dataset

    dataset = load_dataset(
        "wikimedia/wikipedia", WIKIPEDIA_DUMP, split="train", streaming=True
    )
    sample: list[str] = []
    total_words = 0
    for article in dataset.shuffle(
        seed=SAMPLE_SEED, buffer_size=WIKIPEDIA_SHUFFLE_BUFFER
    ):
        blocks = [
            b
            for b in clean(article["text"])
            if len(b.split()) >= MIN_WIKIPEDIA_BLOCK_WORDS
        ]
        for sentence in sentences(blocks):
            sample.append(sentence)
            total_words += len(sentence.split())
        if total_words >= n_words:
            break
    return sample


def reservoir(items: Iterator[Any], k: int, rng: random.Random) -> list[Any]:
    """Uniform sample of k items from an iterator of unknown length (Li's Algorithm L)."""
    sample = list(itertools.islice(items, k))
    weight = math.exp(math.log(rng.random()) / k)
    while True:
        skip = math.floor(math.log(rng.random()) / math.log(1 - weight))
        item = next(itertools.islice(items, skip, None), None)
        if item is None:
            return sample
        sample[rng.randrange(k)] = item
        weight *= math.exp(math.log(rng.random()) / k)


def subtitle_line(line: str) -> str:
    """Drop music notes and bracketed sound descriptions; the dialogue dash goes with clean()'s list markers."""
    line = re.sub(r"[♪♫#]|\[[^\]]*\]|\([^)]*\)", " ", line)
    return line if re.search(r"[A-Za-z]", line) else ""


def sample_opensubtitles(
    path: str | Path, n_words: int, seed: int = SAMPLE_SEED
) -> list[str]:
    """Random lines of the OPUS OpenSubtitles 2018 English monolingual dump (one subtitle line per line)."""
    logger.info("sampling %s (one pass over the whole file, takes a few minutes)", path)
    rng = random.Random(seed)
    with subprocess.Popen(["gzip", "-dc", str(path)], stdout=subprocess.PIPE) as proc:
        # ~6 words/line: oversample, trim below
        lines = reservoir(iter(proc.stdout), n_words // SUBTITLE_WORDS_PER_LINE, rng)
        proc.kill()
    rng.shuffle(lines)
    sample: list[str] = []
    total_words = 0
    for line in lines:
        text = subtitle_line(line.decode("utf-8", "replace"))
        for sentence in sentences(clean(text)):
            sample.append(sentence)
            total_words += len(sentence.split())
        if total_words >= n_words:
            break
    else:
        logger.warning(
            "reference sample short words=%d wanted=%d", total_words, n_words
        )
    return sample


def reference_sentences(
    name: str, n_words: int, opensubtitles: str | Path
) -> list[str]:
    """The cached sample of reference corpus `name`, drawn on first use."""
    path = DATA / f"ref_{name}.txt"
    if not path.exists():
        sample = (
            sample_wikipedia(n_words)
            if name == "wikipedia"
            else sample_opensubtitles(opensubtitles, n_words)
        )
        DATA.mkdir(exist_ok=True)
        path.write_text("\n".join(sample))
    return path.read_text().splitlines()
