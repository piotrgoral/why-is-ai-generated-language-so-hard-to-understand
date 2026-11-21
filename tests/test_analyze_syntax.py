import json

import spacy
from spacy.tokens import Doc

from repro.analysis import judge, syntax

VOCAB = spacy.blank("en").vocab


def doc(words, heads, deps, pos=None, tags=None, lemmas=None):
    return Doc(
        VOCAB,
        words=words,
        heads=heads,
        deps=deps,
        pos=pos or ["X"] * len(words),
        tags=tags or ["X"] * len(words),
        lemmas=lemmas or words,
    )


def spaced(words, **kw):
    d = doc(words, **kw)
    return Doc(
        VOCAB,
        words=words,
        spaces=[True] * (len(words) - 1) + [False],
        heads=[t.head.i for t in d],
        deps=[t.dep_ for t in d],
        pos=[t.pos_ for t in d],
        tags=[t.tag_ for t in d],
        lemmas=[t.lemma_ for t in d],
    )


def test_dep_length():
    d = doc(
        ["She", "eats", "fish", "."], [1, 1, 1, 1], ["nsubj", "ROOT", "dobj", "punct"]
    )
    assert syntax.dep_length(d) == (0, 2, 2)  # punctuation arc excluded


def test_dep_length_long_arc_and_bins():
    d = doc(["a", "b", "c", "d"], [3, 3, 3, 3], ["x", "x", "x", "ROOT"])
    assert syntax.dep_length(d) == (0, 3 + 2 + 1, 3)
    n = 40
    long = doc(["w"] * n, [0] * n, ["ROOT"] + ["dep"] * (n - 1))
    assert syntax.dep_length(long) == (None, 0, 0)  # over 30 words: not binned
    n = 28
    d28 = doc(["w"] * n, [0] * n, ["ROOT"] + ["dep"] * (n - 1))
    assert syntax.dep_length(d28)[0] == 4  # last bin


def test_object_relatives_overt_and_bare():
    # the book that she read
    d = doc(
        ["the", "book", "that", "she", "read"],
        [1, 1, 4, 4, 1],
        ["det", "ROOT", "dobj", "nsubj", "relcl"],
        tags=["DT", "NN", "WDT", "PRP", "VBD"],
    )
    assert syntax.object_relatives(d) == (0, 1)
    # the book she read
    d = doc(
        ["the", "book", "she", "read"],
        [1, 3, 3, 3],
        ["det", "ROOT", "nsubj", "relcl"],
        tags=["DT", "NN", "PRP", "VBD"],
    )
    d[1].dep_ = "ROOT"
    d[3].head = d[1]
    assert syntax.object_relatives(d) == (1, 0)


def test_object_relatives_subject_relative_ignored():
    # the man who left: relativizer is the subject
    d = doc(
        ["the", "man", "who", "left"],
        [1, 1, 3, 1],
        ["det", "ROOT", "nsubj", "relcl"],
        tags=["DT", "NN", "WP", "VBD"],
    )
    assert syntax.object_relatives(d) == (0, 0)


def test_stacked_nouns():
    # "the apple pie recipe": apple, pie are compounds of recipe
    d = doc(
        ["the", "apple", "pie", "recipe"],
        [3, 3, 3, 3],
        ["det", "compound", "compound", "ROOT"],
        pos=["DET", "NOUN", "NOUN", "NOUN"],
    )
    assert syntax.stacked_nouns(d) == 1
    d = doc(["pie", "recipe"], [1, 1], ["compound", "ROOT"], pos=["NOUN", "NOUN"])
    assert syntax.stacked_nouns(d) == 0


def test_mark():
    d = spaced(
        ["The", "rule", "applies", "."],
        heads=[1, 2, 2, 2],
        deps=["det", "nsubj", "ROOT", "punct"],
    )
    assert syntax.mark(d, d[2], d[1]) == "The {{rule}} [[applies]] ."
    assert "{{" not in syntax.mark(d, d[2])


def test_candidates_inanimate_and_denominal():
    d = spaced(
        ["The", "rule", "governs", "it"],
        heads=[1, 2, 2, 2],
        deps=["det", "nsubj", "ROOT", "dobj"],
        pos=["DET", "NOUN", "VERB", "PRON"],
        lemmas=["the", "rule", "govern", "it"],
    )
    out = syntax.candidates(d, {"govern": 0.9})
    assert ("inanimate", "The {{rule}} [[governs]] it") in out
    assert ("denominal", "The rule [[governs]] it") in out
    assert [k for k, _ in syntax.candidates(d, {})] == ["inanimate"]


def test_candidates_skips_animate_and_passive():
    d = spaced(
        ["She", "runs"], heads=[1, 1], deps=["nsubj", "ROOT"], pos=["PRON", "VERB"]
    )
    assert syntax.candidates(d, {}) == []
    d = spaced(
        ["The", "rule", "was", "applied"],
        heads=[1, 3, 3, 3],
        deps=["det", "nsubjpass", "auxpass", "ROOT"],
        pos=["DET", "NOUN", "AUX", "VERB"],
    )
    assert syntax.candidates(d, {}) == []


def test_candidates_placeholder_subject_is_inanimate():
    d = spaced(
        ["Xyz", "returns", "a", "value"],
        heads=[1, 1, 3, 1],
        deps=["nsubj", "ROOT", "det", "dobj"],
        pos=["PROPN", "VERB", "DET", "NOUN"],
    )
    assert [k for k, _ in syntax.candidates(d, {})] == ["inanimate"]


class FakeNlp:
    meta = {"name": "fake"}

    def __init__(self):
        self.parsed = []

    def pipe(self, texts, batch_size=64):
        for t in texts:
            self.parsed.append(t)
            ws = t.split()
            yield doc(ws, [0] * len(ws), ["ROOT"] + ["dep"] * (len(ws) - 1))


def test_featurize_caches(tmp_path):
    nlp = FakeNlp()
    out = syntax.featurize(["a b c", "d e", "a b c"], nlp, {})
    assert set(out) == {"a b c", "d e"} and sorted(nlp.parsed) == ["a b c", "d e"]
    assert out["a b c"]["dl_n"] == 2
    nlp2 = FakeNlp()
    out2 = syntax.featurize(["a b c", "f g"], nlp2, {})
    assert nlp2.parsed == ["f g"] and out2["a b c"] == out["a b c"]
    assert len(syntax.FEATURE_CACHE.read_text().splitlines()) == 3


def test_run_jev_caches_and_calls_classifier(monkeypatch):
    calls = []
    monkeypatch.setattr(judge.jev, "classify", lambda f, t: calls.append((f, t)) or 0.9)
    cands = [("denominal", "a [[b]]"), ("inanimate", "c {{d}} [[e]]")]
    out = judge.judge_candidates(cands, workers=2, chunk_size=1)
    assert out == {c: 0.9 for c in cands} and len(calls) == 2
    calls.clear()
    assert (
        judge.judge_candidates(cands + [("denominal", "new")], workers=1)[
            ("denominal", "new")
        ]
        == 0.9
    )
    assert calls == [("denominal", "new")]
    assert (
        len([json.loads(line) for line in judge.JUDGE_CACHE.read_text().splitlines()])
        == 3
    )
