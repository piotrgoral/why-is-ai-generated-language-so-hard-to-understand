from repro import llm, roundtrip
from repro.experiment import Experiment
from repro.jsonl import append_jsonl, read_jsonl
from tests.fakes import FakeClient


def test_unit_sentences():
    rows = [
        {
            "setting": "chat",
            "model": "m",
            "unit_id": "u",
            "prompt": "p",
            "turn": 2,
            "text": "Hello there. How are you?\n- A list item.",
        }
    ]
    out = roundtrip.unit_sentences(rows, essay=False)
    assert [o["orig"] for o in out] == ["Hello there.", "How are you?", "A list item."]
    assert [o["sent_id"] for o in out] == ["2.0", "2.1", "2.2"]
    assert out[0]["model"] == "m" and out[0]["unit_id"] == "u"


def test_translator_dispatch(monkeypatch):
    monkeypatch.setattr(roundtrip, "NLLBTranslator", lambda n: ("nllb", n))
    monkeypatch.setattr(roundtrip, "ModalNLLBTranslator", lambda n: ("modal", n))
    assert roundtrip.translator("facebook/nllb-200-3.3B") == (
        "nllb",
        "facebook/nllb-200-3.3B",
    )
    assert roundtrip.translator("modal:facebook/nllb-200-3.3B") == (
        "modal",
        "facebook/nllb-200-3.3B",
    )
    assert isinstance(roundtrip.translator("tencent/x"), roundtrip.OpenRouterTranslator)


def test_openrouter_round_trip(monkeypatch):
    def fn(kw):
        prompt = kw["messages"][0]["content"]
        assert kw["temperature"] == 0
        return " こんにちは " if "Japanese" in prompt else " Hello again "

    monkeypatch.setattr(llm, "client", lambda: FakeClient(fn))
    ja, en = roundtrip.OpenRouterTranslator("m", workers=2).round_trip(["Hello", ""])
    assert ja == ["こんにちは", ""] and en == ["Hello again", ""]


def test_openrouter_untranslated_pivot_gives_no_round_trip(monkeypatch):
    monkeypatch.setattr(llm, "client", lambda: FakeClient(lambda kw: "still English"))
    ja, en = roundtrip.OpenRouterTranslator("m").round_trip(["Hello"])
    assert ja == [""] and en == [""]


class FakeTranslator:
    remote = False

    def __init__(self, rts):
        self.rts = rts

    def round_trip(self, texts):
        return ["ja"] * len(texts), [self.rts.get(t, t) for t in texts]


def test_run_filters_ratio_and_resumes(monkeypatch, tmp_path):
    e = Experiment(
        raw=tmp_path / "raw.jsonl",
        sents=tmp_path / "sents.jsonl",
        human_baseline=False,
        translator="fake",
    )
    append_jsonl(
        e.raw,
        [
            {
                "setting": "chat",
                "model": "m",
                "unit_id": "u1",
                "prompt": "p",
                "turn": 0,
                "text": "One two three four. Five six.",
            },
            {
                "setting": "chat",
                "model": "m",
                "unit_id": "u2",
                "prompt": "p",
                "turn": 0,
                "text": "Alpha beta.",
            },
        ],
    )
    tr = FakeTranslator(
        {
            "One two three four.": "One.",  # ratio 0.25: rejected
            "Five six.": "Five six seven.",  # ratio 1.5: kept
            "Alpha beta.": "a b c d e f.",
        }
    )  # ratio 3: rejected
    monkeypatch.setattr(roundtrip, "translator", lambda name: tr)
    roundtrip.run(e, units_per_flush=1)
    rows = {r["orig"]: r for r in read_jsonl(e.sents)}
    assert rows["One two three four."]["rt"] is None
    assert rows["Five six."]["rt"] == "Five six seven."
    assert rows["Alpha beta."]["rt"] is None
    assert (
        rows["Five six."]["pivot"] == "ja" and rows["Five six."]["translator"] == "fake"
    )
    n = len(read_jsonl(e.sents))
    roundtrip.run(e)  # everything done: no translator needed
    assert len(read_jsonl(e.sents)) == n


def test_run_adds_human_baseline(monkeypatch, tmp_path):
    csv = tmp_path / "p.csv"
    csv.write_text(
        "essay_id_comp,prompt_name,full_text,grade_level,ell_status,holistic_essay_score\n"
        'E1,Summer projects,"I think summer projects are a very good idea for students everywhere. They help.",10,No,5\n'
        'E2,Summer projects,"Low scored essay text goes here and is long enough to count.",10,No,3\n'
    )
    e = Experiment(
        raw=tmp_path / "raw.jsonl", sents=tmp_path / "sents.jsonl", persuade=str(csv)
    )
    roundtrip.run(e)
    rows = read_jsonl(e.sents)
    assert (
        rows
        and {r["model"] for r in rows} == {"human"}
        and {r["unit_id"] for r in rows} == {"E1"}
    )
    assert all(r["rt"] is None for r in rows)
    n = len(rows)
    roundtrip.run(e)  # not added twice
    assert len(read_jsonl(e.sents)) == n
