import pytest

from repro.analysis import tables
from repro.analysis.metrics import COUNTS
from repro.analysis.syntax import BINS

W = [1 / BINS] * BINS


def sentence(model, unit, orig, rt, setting="chat"):
    return {
        "setting": setting,
        "model": model,
        "unit_id": unit,
        "prompt": "p",
        "orig": orig,
        "rt": rt,
    }


def test_long_table_pairs_machine_text_and_keeps_human_text():
    rows = [
        sentence("m", "u1", "Hello.", "Hi."),
        sentence("m", "u1", "Dropped.", None),  # failed round trip: both versions out
        sentence("human", "E1", "I agree.", None, setting="essay"),
    ]
    table = tables.long_table(rows)
    assert list(zip(table.model, table.source, table.text)) == [
        ("m", "orig", "Hello."),
        ("m", "rt", "Hi."),
        ("human", "human", "I agree."),
    ]


def test_unit_counts_sums_sentences_of_a_unit():
    texts = tables.long_table(
        [sentence("m", "u1", "a.", "b."), sentence("m", "u1", "c.", "d.")]
    )
    texts = texts.assign(**{name: 1 for name in COUNTS})
    units = tables.unit_counts(texts)
    assert sorted(units.source) == ["orig", "rt"]
    assert (units.words == 2).all()


def test_paired_differences_is_empty_without_machine_units():
    texts = tables.long_table([sentence("human", "E1", "x.", None, setting="essay")])
    units = tables.unit_counts(texts.assign(**{name: 1 for name in COUNTS}))
    assert tables.paired_differences(units, W).empty


@pytest.mark.parametrize("setting", ["chat", "essay"])
def test_paired_differences_reports_every_metric(setting):
    rows = [sentence("m", f"u{i}", "a.", "b.", setting=setting) for i in range(3)]
    texts = tables.long_table(rows)
    texts = texts.assign(**{name: 1 for name in COUNTS})
    differences = tables.paired_differences(tables.unit_counts(texts), W)
    assert set(differences.model) == {"m"} and len(differences) == 8
