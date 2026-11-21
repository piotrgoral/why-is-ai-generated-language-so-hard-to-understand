import random

import numpy as np
import pandas as pd
import pytest

from repro.analysis import reference
from repro.analysis.metrics import COUNTS, metrics
from repro.analysis.ngram import KneserNey, ngram_features, words
from repro.analysis.stats import bootstrap, paired_jackknife
from repro.analysis.syntax import BINS

W = [1 / BINS] * BINS


def counts(**kw):
    return [kw.get(c, 0) for c in COUNTS]


def test_words():
    assert words("Don't STOP, it's 3 o'clock!") == [
        "don't",
        "stop",
        "it's",
        "3",
        "o'clock",
    ]


def test_reservoir_size_and_membership():
    out = reference.reservoir(iter(range(1000)), 10, random.Random(0))
    assert len(out) == 10 and set(out) <= set(range(1000))
    assert reference.reservoir(iter(range(3)), 10, random.Random(0)) == [0, 1, 2]


def test_subtitle_line():
    assert reference.subtitle_line("♪ la la ♪").split() == ["la", "la"]
    assert reference.subtitle_line("♪ ♪") == ""
    assert reference.subtitle_line("[door slams] Hello (softly)").split() == ["Hello"]
    assert reference.subtitle_line("12345") == ""


SENTS = [
    words(s)
    for s in ["the cat sat on the mat", "the dog sat on the log", "a cat saw the dog"]
] * 3


def test_kneser_ney_probabilities():
    kn = KneserNey(SENTS)
    assert all(kn.p((), w) > 0 for w in ("the", "cat", "zzz"))
    assert kn.p(("the",), "cat") > kn.p(("the",), "zzz")
    # distribution over seen vocabulary plus </s> stays below 1 and is substantial
    vocab = {w for s in SENTS for w in s} | {"</s>"}
    total = sum(kn.p(("the",), w) for w in vocab)
    assert 0.5 < total <= 1.0 + 1e-9


def test_surprisal_orders_and_skips_placeholders():
    kn = KneserNey(SENTS)
    common, _ = kn.surprisal(words("the cat sat on the mat"))
    rare, n = kn.surprisal(words("mat the on sat cat the"))
    assert rare > common and n == 7
    bits, n2 = kn.surprisal(
        ["the", "xyz", "sat"]
    )  # xyz is a placeholder: trigrams touching it are dropped
    assert n2 < 4


def test_kneser_ney_ignores_empty_sentences():
    assert KneserNey([[], *SENTS]).vocab == KneserNey(SENTS).vocab


def test_ngram_features():
    kn = KneserNey(SENTS)
    bigrams = {("the", "cat"), ("cat", "sat")}
    f = ngram_features("The cat sat Xyz", kn, bigrams)
    assert f["words"] == 3 and f["pairs"] == 2 and f["unattested"] == 0
    f = ngram_features("the dog", kn, bigrams)
    assert f["unattested"] == 1


def test_metrics_rates():
    m = metrics(
        counts(
            words=2000,
            pairs=100,
            unattested=25,
            bits=300,
            trigrams=100,
            orc_bare=4,
            orc_that=2,
            stacked=6,
            denominal=8,
            inanimate=10,
        ),
        W,
    )
    assert m["unattested_per_1k_pairs"] == 250
    assert m["trigram_surprisal_bits"] == 3
    assert m["orc_no_that_per_1k_words"] == 2 and m["orc_that_per_1k_words"] == 1
    assert m["stacked_nouns_per_1k_words"] == 3
    assert m["denominal_per_1k_words"] == 4 and m["inanimate_subj_per_1k_words"] == 5
    assert np.isnan(m["dep_length_std"])


def test_metrics_empty_is_nan():
    m = metrics(counts(), W)
    assert all(np.isnan(v) for v in m.values())


def test_metrics_dep_length_weighted_over_present_bins():
    c = counts(dl_sum0=10, dl_n0=5, dl_sum2=30, dl_n2=10)  # mean arc 2 and 3
    w = [0.5, 0, 0.5, 0, 0]
    assert metrics(c, w)["dep_length_std"] == pytest.approx(2.5)
    w = [0.25, 0, 0.75, 0, 0]
    assert metrics(c, w)["dep_length_std"] == pytest.approx(2.75)


def make_units(n, rng):
    return pd.DataFrame(
        [
            counts(
                words=int(rng.integers(50, 100)),
                pairs=40,
                unattested=int(rng.integers(0, 10)),
                bits=100,
                trigrams=40,
            )
            for _ in range(n)
        ],
        columns=COUNTS,
    )


def test_bootstrap_reproducible_and_brackets_point():
    units = make_units(30, np.random.default_rng(0))
    strata = np.zeros(30)
    lo, hi = bootstrap(units, W, strata, n_draws=200, seed=1)
    lo2, hi2 = bootstrap(units, W, strata, n_draws=200, seed=1)
    pd.testing.assert_series_equal(lo, lo2)
    point = metrics(units.to_numpy(float).sum(0), W)["unattested_per_1k_pairs"]
    assert lo["unattested_per_1k_pairs"] <= point <= hi["unattested_per_1k_pairs"]


def test_bootstrap_respects_strata():
    units = make_units(6, np.random.default_rng(0))
    units.loc[:2, "unattested"] = 0
    units.loc[3:, "unattested"] = 40
    lo, hi = bootstrap(units, W, np.array([0, 0, 0, 1, 1, 1]), n_draws=100)
    # each stratum is resampled within itself, so 3 low + 3 high units always: the interval is tight
    assert hi["unattested_per_1k_pairs"] - lo["unattested_per_1k_pairs"] < 1e-6


def test_paired_jackknife_detects_difference():
    rng = np.random.default_rng(0)
    orig, rt = make_units(20, rng), make_units(20, rng)
    orig["unattested"] += 10
    full, t, p = paired_jackknife(orig, rt, W)
    assert full["unattested_per_1k_pairs"] > 0
    assert p["unattested_per_1k_pairs"] < 0.01


def test_paired_jackknife_identical_has_no_test():
    orig = make_units(5, np.random.default_rng(0))
    full, t, p = paired_jackknife(orig, orig.copy(), W)
    assert full["unattested_per_1k_pairs"] == 0
    assert np.isnan(t["unattested_per_1k_pairs"])


def test_paired_jackknife_aligns_on_index():
    orig = make_units(5, np.random.default_rng(0)).set_axis(list("abcde"))
    rt = orig.loc[list("bcd")]
    full, _, _ = paired_jackknife(orig, rt, W)
    assert full["unattested_per_1k_pairs"] == 0
