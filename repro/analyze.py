"""Measure the difficulty-linked features (Sec. 3.1, App. B) for original text, its round trip and student
essays; pool per model with bootstrap CIs and test original vs. round trip -> results/*.csv."""

import argparse
from pathlib import Path

import spacy

from repro import experiment
from repro.analysis.judge import judge_candidates
from repro.analysis.ngram import KneserNey, words
from repro.analysis.reference import (
    DEFAULT_OPENSUBTITLES,
    REFERENCE_FOR_SETTING,
    reference_sentences,
)
from repro.analysis.report import print_differences, write_results
from repro.analysis.syntax import featurize, reference_stats
from repro.analysis.tables import (
    ReferenceModels,
    add_count_columns,
    long_table,
    paired_differences,
    pooled_metrics,
    unit_counts,
)
from repro.cli import apply_overrides, configure_logging
from repro.experiment import Experiment
from repro.jsonl import read_jsonl

REFERENCE_WORDS = 500_000
REFERENCE_PARSE_WORDS = 100_000  # Wikipedia words parsed for the reference stats
# dependency-length weights and noun shares always come from Wikipedia (App. B.1, B.3)
STATS_REFERENCE = "wikipedia"


def build_reference_models(
    names: set[str], n_words: int, opensubtitles: str | Path
) -> ReferenceModels:
    """name -> (trigram model, attested bigrams) for each reference corpus."""
    models: ReferenceModels = {}
    for name in sorted(names):
        tokenized = [
            words(s) for s in reference_sentences(name, n_words, opensubtitles)
        ]
        bigrams = {
            pair for sentence in tokenized for pair in zip(sentence, sentence[1:])
        }
        models[name] = KneserNey(tokenized), bigrams
    return models


def run(
    exp: Experiment,
    ref_words: int = REFERENCE_WORDS,
    ref_parse_words: int = REFERENCE_PARSE_WORDS,
    opensubtitles: str | Path = DEFAULT_OPENSUBTITLES,
) -> None:
    """Compute counts for every sentence version, then the pooled metrics and paired tests."""
    rows = read_jsonl(exp.sents)
    if not rows:
        raise SystemExit(f"no {exp.sents}; run repro.roundtrip first")
    nlp = spacy.load(exp.spacy)
    reference_names = {REFERENCE_FOR_SETTING[r["setting"]] for r in rows} | {
        STATS_REFERENCE
    }
    reference_models = build_reference_models(reference_names, ref_words, opensubtitles)
    stats = reference_stats(
        nlp,
        reference_sentences(STATS_REFERENCE, ref_words, opensubtitles),
        ref_parse_words,
    )
    bin_weights = stats["bin_weights"]

    texts = long_table(rows)
    parse_features = featurize(texts.text.tolist(), nlp, stats["noun_share"])
    judged = (
        judge_candidates(
            [tuple(c) for f in parse_features.values() for c in f["cands"]]
        )
        if exp.judge
        else {}
    )
    units = unit_counts(
        add_count_columns(texts, reference_models, parse_features, judged)
    )
    pooled = pooled_metrics(units, bin_weights)
    differences = paired_differences(units, bin_weights)

    write_results(exp.results, units, pooled, differences)
    print_differences(exp.name, differences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="experiment TOML (default: data/sentences.jsonl -> results/)"
    )
    parser.add_argument(
        "--spacy", help="default en_core_web_trf; en_core_web_sm is much faster"
    )
    parser.add_argument("--ref-words", type=int, default=REFERENCE_WORDS)
    parser.add_argument(
        "--ref-parse-words",
        type=int,
        default=REFERENCE_PARSE_WORDS,
        help="Wikipedia words parsed for ref stats",
    )
    parser.add_argument(
        "--opensubtitles",
        default=DEFAULT_OPENSUBTITLES,
        help="OPUS OpenSubtitles v2018 mono/en.txt.gz (chat n-gram reference)",
    )
    parser.add_argument(
        "--no-judge", action="store_true", help="skip Jev (denominal/inanimate = 0)"
    )
    args = parser.parse_args()
    configure_logging()

    exp = experiment.load(args.config)
    apply_overrides(exp, args, "spacy")
    if args.no_judge:
        exp.judge = False
    run(exp, args.ref_words, args.ref_parse_words, args.opensubtitles)


if __name__ == "__main__":
    main()
