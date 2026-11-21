import numpy as np
import pandas as pd
import pytest

from repro import figure2
from repro.analysis.metrics import COUNTS
from repro.analysis.syntax import BINS

W = [1 / BINS] * BINS


def unit(model, unit_id, source, words, stacked):
    counts = dict.fromkeys(COUNTS, 0)
    counts.update(words=words, stacked=stacked)
    return {
        "setting": "chat",
        "model": model,
        "source": source,
        "unit_id": unit_id,
        "prompt": "p",
        **counts,
    }


@pytest.mark.filterwarnings("ignore:Mean of empty slice")  # pairs that are all NaN here
def test_score_is_change_over_sd_of_originals_across_models():
    # stacked nouns per 1k words: model a 10 -> 4 (rt), model b 20 -> 20; SD of originals = std(10, 20)
    units = pd.DataFrame(
        [
            unit("a", "u1", "orig", 1000, 10),
            unit("a", "u1", "rt", 1000, 4),
            unit("b", "u1", "orig", 1000, 20),
            unit("b", "u1", "rt", 1000, 20),
            unit("b", "u2", "rt", 1000, 99),  # no original for u2: dropped
            unit("human", "h1", "human", 1000, 50),  # not a model: ignored
        ]
    )
    df, sd = figure2.figure2(units, W, n_draws=5)
    assert sd.loc["chat", "stacked_nouns_per_1k_words"] == pytest.approx(
        np.std([10, 20], ddof=1)
    )
    assert sd.loc["chat", "n_models"] == 2
    chat = df[df.setting == "chat"].set_index(["model", "group"]).value
    expected = 6 / np.std([10, 20], ddof=1)
    # dependency length is NaN (no arcs), so the pair is stacked nouns alone; other pairs have SD 0 -> NaN
    assert chat["a", "dependencies"] == pytest.approx(expected)
    assert chat["a", figure2.ALL_PAIRS] == pytest.approx(expected)
    assert chat["b", figure2.ALL_PAIRS] == pytest.approx(0)
    overall = df[df.setting == "overall"].set_index("model").value
    assert overall["a"] == pytest.approx(expected)
