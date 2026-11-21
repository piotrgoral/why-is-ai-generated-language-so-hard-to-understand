from repro import jev
from repro.analysis import judge
from repro.jsonl import read_jsonl


def test_only_uncached_candidates_are_sent_and_results_are_cached(
    monkeypatch, tmp_path
):
    cache = tmp_path / "jev.jsonl"
    sent = []

    def classify(feature, text):
        sent.append((feature, text))
        return 0.9

    monkeypatch.setattr(jev, "classify", classify)
    first = [("denominal", "a [[b]]"), ("inanimate", "c [[d]]")]
    result = judge.judge_candidates(first, workers=1, cache_path=cache)
    assert result == {c: 0.9 for c in first} and len(sent) == 2

    sent.clear()
    again = judge.judge_candidates(
        [*first, ("denominal", "new [[x]]")], workers=1, cache_path=cache
    )
    assert sent == [("denominal", "new [[x]]")] and len(again) == 3
    assert len(read_jsonl(cache)) == 3


def test_accepted_count_applies_threshold_and_feature():
    judged = {("denominal", "a"): 0.6, ("denominal", "b"): 0.5, ("inanimate", "c"): 1}
    cands = [["denominal", "a"], ["denominal", "b"], ["inanimate", "c"]]
    assert judge.accepted_count("denominal", cands, judged) == 1  # 0.5 is not > 0.5
    assert judge.accepted_count("inanimate", cands, judged) == 1
    assert judge.accepted_count("denominal", [["denominal", "unknown"]], judged) == 0
