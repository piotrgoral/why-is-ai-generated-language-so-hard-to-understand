"""Default locations (relative to the working directory) of shared data and results."""

from pathlib import Path

DATA = Path("data")
RESULTS = Path("results")
RUNS = Path("runs")

RAW = DATA / "raw.jsonl"
SENTS = DATA / "sentences.jsonl"
