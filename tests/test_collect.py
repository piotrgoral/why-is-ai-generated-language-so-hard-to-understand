import pytest

from repro import collect, llm
from repro.experiment import Experiment
from repro.prompts import MODELS, SIM_USER_FALLBACK, SIM_USER_MODEL, TOPICS
from tests.fakes import FakeChat

MODEL = MODELS[0]


@pytest.fixture
def fake(monkeypatch):
    f = FakeChat()
    monkeypatch.setattr(llm, "chat", f)
    monkeypatch.setattr(llm, "check_models", lambda slugs: None)
    return f


def test_simulate_user_swaps_roles_and_returns_text(fake):
    fake.replies = ["next question"]
    convo = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert collect.simulate_user("SYS", convo) == "next question"
    model, msgs, reasoning = fake.calls[0]
    assert model == SIM_USER_MODEL and reasoning is False
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert [m["content"] for m in msgs[2:]] == ["hi", "hello"]
    assert len(fake.calls) == 1


def test_simulate_user_falls_back_after_two_bad_replies(fake):
    fake.replies = ["", "x" * 2001]
    assert collect.simulate_user("S", []) == SIM_USER_FALLBACK
    assert len(fake.calls) == 2


def test_run_chat_stops_at_token_budget(fake):
    rows = collect.run_chat(MODEL, TOPICS[0], max_output_tokens=15)
    assert [r["turn"] for r in rows] == [
        0,
        1,
    ]  # 10 tokens/reply: budget exceeded after the second
    assert (
        rows[0]["setting"] == "chat"
        and rows[0]["unit_id"] == TOPICS[0].topic_id
        and rows[0]["model"] == MODEL.slug
    )


def test_run_chat_stops_at_max_replies(fake):
    rows = collect.run_chat(MODEL, TOPICS[0], max_output_tokens=10**9)
    assert len(rows) == collect.MAX_REPLIES


def test_run_essay(fake):
    (row,) = collect.run_essay(MODEL, "Summer projects", "Write.", 3)
    assert (
        row["unit_id"] == "Summer projects#3"
        and row["setting"] == "essay"
        and row["text"] == "reply"
    )


def exp(tmp_path, **kw):
    return Experiment(models=[MODEL.slug], raw=tmp_path / "raw.jsonl", **kw)


def test_run_chat_setting_resumes_and_limits(fake, tmp_path):
    e = exp(tmp_path, limit=2, max_output_tokens=1)
    collect.run(e, "chat", workers=1)
    rows = collect.read_jsonl(e.raw)
    assert len(rows) == 2 and {r["setting"] for r in rows} == {"chat"}
    collect.run(e, "chat", workers=1)  # limit counts units still to do: two more
    assert len({r["unit_id"] for r in collect.read_jsonl(e.raw)}) == 4


def test_run_skips_done_units(fake, tmp_path):
    e = exp(tmp_path, max_output_tokens=1)
    collect.append_jsonl(
        e.raw,
        [
            {"setting": "chat", "model": MODEL.slug, "unit_id": t.topic_id}
            for t in TOPICS
        ],
    )
    collect.run(e, "chat", workers=1)
    assert fake.calls == []


def test_run_drops_failed_unit(monkeypatch, tmp_path, caplog):
    def boom(*a, **k):
        raise RuntimeError("api down")

    monkeypatch.setattr(llm, "chat", boom)
    monkeypatch.setattr(llm, "check_models", lambda s: None)
    e = exp(tmp_path, limit=1)
    collect.run(e, "chat", workers=1)
    assert collect.read_jsonl(e.raw) == []
    failure = next(r for r in caplog.records if r.levelname == "ERROR")
    assert (
        "unit_id=" in failure.getMessage() and failure.exc_info
    )  # context + traceback


def test_run_essay_setting(fake, tmp_path):
    csv = tmp_path / "p.csv"
    csv.write_text(
        "prompt_name,assignment,grade_level,ell_status,holistic_essay_score\n"
        "Summer projects,Write about summer,10,No,5\n"
    )
    e = exp(tmp_path, persuade=str(csv), limit=2)
    collect.run(e, "essay", workers=1)
    rows = collect.read_jsonl(e.raw)
    assert sorted(r["unit_id"] for r in rows) == [
        "Summer projects#0",
        "Summer projects#1",
    ]


def test_run_essay_requires_persuade(fake, tmp_path):
    with pytest.raises(SystemExit):
        collect.run(exp(tmp_path), "essay")
