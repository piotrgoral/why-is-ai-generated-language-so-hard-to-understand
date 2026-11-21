"""Sentence-level and unit-level count tables, and the pooled and paired-difference results (App. B.4)."""

from collections.abc import Iterable

import numpy as np
import pandas as pd

from repro.analysis.judge import accepted_count
from repro.analysis.metrics import (
    COUNTS,
    DEP_LENGTH_ARC_COUNTS,
    DEP_LENGTH_SUM_COUNTS,
    JUDGED_COUNTS,
    NGRAM_COUNTS,
    SYNTAX_COUNTS,
    metrics,
)
from repro.analysis.ngram import KneserNey, ngram_features
from repro.analysis.reference import REFERENCE_FOR_SETTING
from repro.analysis.stats import bootstrap, paired_jackknife
from repro.jsonl import Record

TEXT_COLUMNS = ["setting", "model", "unit_id", "prompt", "source", "text"]
DEP_LENGTH_COLUMNS = {  # parse feature -> its count column per length bin
    "dl_sum": DEP_LENGTH_SUM_COUNTS,
    "dl_n": DEP_LENGTH_ARC_COUNTS,
}
UNIT_KEY = ["setting", "model", "source", "unit_id", "prompt"]


def long_table(sentence_rows: Iterable[Record]) -> pd.DataFrame:
    """One row per (sentence version): human text, or original and round trip of each pair."""
    rows = []
    for row in sentence_rows:
        base = (row["setting"], row["model"], row["unit_id"], row["prompt"])
        if row["model"] == "human":
            rows.append((*base, "human", row["orig"]))
        elif row["rt"]:
            rows += [(*base, "orig", row["orig"]), (*base, "rt", row["rt"])]
    return pd.DataFrame(rows, columns=TEXT_COLUMNS)


ReferenceModels = dict[str, tuple[KneserNey, set[tuple[str, str]]]]


def add_count_columns(
    texts: pd.DataFrame,
    reference_models: ReferenceModels,
    parse_features: dict[str, Record],
    judged: dict,
) -> pd.DataFrame:
    """Per-sentence counts, as columns in the order of COUNTS."""
    ngram_by_text = {
        (setting, text): ngram_features(
            text, *reference_models[REFERENCE_FOR_SETTING[setting]]
        )
        for setting, text in set(zip(texts.setting, texts.text))
    }
    columns: dict[str, list] = {}
    for name in NGRAM_COUNTS:
        columns[name] = [
            ngram_by_text[setting, text][name]
            for setting, text in zip(texts.setting, texts.text)
        ]
    for name in SYNTAX_COUNTS:
        columns[name] = [parse_features[text][name] for text in texts.text]
    for name in JUDGED_COUNTS:
        columns[name] = [
            accepted_count(name, parse_features[text]["cands"], judged)
            for text in texts.text
        ]
    for feature, names in DEP_LENGTH_COLUMNS.items():
        for length_bin, name in enumerate(names):
            columns[name] = [
                parse_features[text][feature]
                if parse_features[text]["dl_bin"] == length_bin
                else 0
                for text in texts.text
            ]
    return texts.assign(**{name: columns[name] for name in COUNTS})


def unit_counts(sentence_counts: pd.DataFrame) -> pd.DataFrame:
    """Counts summed per unit (a conversation, or an essay)."""
    return sentence_counts.groupby(UNIT_KEY)[list(COUNTS)].sum().reset_index()


def pooled_metrics(units: pd.DataFrame, bin_weights: list[float]) -> pd.DataFrame:
    """Each metric per (setting, model, source), pooled over units, with bootstrap CIs."""
    rows = []
    for (setting, model, source), group in units.groupby(
        ["setting", "model", "source"]
    ):
        strata = group.prompt.to_numpy() if setting == "essay" else np.zeros(len(group))
        lower, upper = bootstrap(group, bin_weights, strata)
        pooled = metrics(group[list(COUNTS)].sum().to_numpy(float), bin_weights)
        rows += [
            {
                "setting": setting,
                "model": model,
                "source": source,
                "metric": metric,
                "value": value,
                "lo": lower[metric],
                "hi": upper[metric],
                "units": len(group),
            }
            for metric, value in pooled.items()
        ]
    return pd.DataFrame(rows)


def paired_differences(units: pd.DataFrame, bin_weights: list[float]) -> pd.DataFrame:
    """Original minus round trip per (setting, model) and metric, with paired jackknife tests."""
    rows = []
    machine_units = units[units.source != "human"]
    for (setting, model), group in machine_units.groupby(["setting", "model"]):
        orig, rt = (
            group[group.source == source].set_index("unit_id")
            for source in ("orig", "rt")
        )
        difference, t, p = paired_jackknife(orig, rt, bin_weights)
        rows += [
            {
                "setting": setting,
                "model": model,
                "metric": metric,
                "orig_minus_rt": difference[metric],
                "t": t[metric],
                "p": p[metric],
            }
            for metric in difference.index
        ]
    return pd.DataFrame(rows)
