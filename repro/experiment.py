"""Experiment configs (experiments/*.toml): which models and settings to run, stage options, and where the
experiment's raw replies, sentences and results live. Without a config the stages use the default experiment:
all models in prompts.MODELS, data/raw.jsonl, data/sentences.jsonl and results/."""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repro.paths import RAW, RESULTS, RUNS, SENTS
from repro.prompts import MODELS, ModelSpec
from repro.records import SETTINGS, Setting

# keys accepted at the top level and in the [roundtrip] and [analyze] tables
TOP_LEVEL_KEYS = {
    "models",
    "settings",
    "persuade",
    "human_baseline",
    "limit",
    "max_output_tokens",
}
ROUNDTRIP_KEYS = {"translator"}
ANALYZE_KEYS = {"spacy", "judge"}


@dataclass
class Experiment:
    name: str = "default"
    models: list[str] = field(default_factory=lambda: [m.slug for m in MODELS])
    settings: list[Setting] = field(default_factory=lambda: list(SETTINGS))
    # PERSUADE 2.0 CSV: essay assignments (+ student essays if human_baseline)
    persuade: str | None = None
    human_baseline: bool = True
    limit: int | None = None  # max units per model and setting (smoke runs)
    # a chat ends after 12 replies or this many output tokens incl. reasoning (paper: 20k)
    max_output_tokens: int = 20_000
    # OpenRouter slug, or facebook/nllb-200-3.3B (local, as in the paper)
    translator: str = "tencent/hy-mt2-30b-a3b"
    spacy: str = "en_core_web_trf"
    judge: bool = True
    raw: Path = RAW
    sents: Path = SENTS
    results: Path = RESULTS

    def model_specs(self) -> list[ModelSpec]:
        """The configured models, in config order."""
        known = {m.slug: m for m in MODELS}
        missing = [slug for slug in self.models if slug not in known]
        if missing:
            raise SystemExit(f"not in prompts.MODELS: {missing}")
        return [known[slug] for slug in self.models]


def _reject_unknown_keys(
    path: str | Path, cfg: dict[str, Any], roundtrip: dict, analyze: dict
) -> None:
    unknown = (
        (set(cfg) - TOP_LEVEL_KEYS)
        | (set(roundtrip) - ROUNDTRIP_KEYS)
        | (set(analyze) - ANALYZE_KEYS)
    )
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")


def load(path: str | Path | None = None) -> Experiment:
    """The experiment described by the TOML at `path`; the default experiment without one."""
    if path is None:
        return Experiment()
    cfg = tomllib.loads(Path(path).read_text())
    roundtrip, analyze = cfg.pop("roundtrip", {}), cfg.pop("analyze", {})
    name = cfg.pop("name", Path(path).stem)
    run_dir = RUNS / name
    # raw: reuse another run's conversations (e.g. "runs/haiku-roundtrip/raw.jsonl" with a different
    # translator); collect then finds them done and adds nothing
    raw = Path(cfg.pop("raw", run_dir / "raw.jsonl"))
    _reject_unknown_keys(path, cfg, roundtrip, analyze)
    experiment = Experiment(
        name=name,
        **cfg,
        **roundtrip,
        **analyze,
        raw=raw,
        sents=run_dir / "sentences.jsonl",
        results=run_dir / "results",
    )
    experiment.model_specs()  # fail early on typos
    if bad := set(experiment.settings) - set(SETTINGS):
        raise SystemExit(f"{path}: unknown settings {sorted(bad)}")
    return experiment
