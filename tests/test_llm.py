from collections import Counter
from types import SimpleNamespace

import pytest

from repro import jev, llm
from tests.fakes import FakeClient, FakeResp, Usage


@pytest.fixture(autouse=True)
def reset_usage():
    llm.USAGE.clear()
    yield
    llm.USAGE.clear()


def test_cache_breakpoint_string_content():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    out = llm._with_cache_breakpoint(msgs)
    assert out[-1]["content"] == [
        {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
    ]
    assert msgs[-1]["content"] == "hi" and out[0] is msgs[0]  # input untouched


def test_cache_breakpoint_list_content():
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }
    ]
    c = llm._with_cache_breakpoint(msgs)[0]["content"]
    assert "cache_control" not in c[0] and c[1]["cache_control"] == {
        "type": "ephemeral"
    }


def test_tally_and_summary():
    llm.USAGE.record(None)
    llm.USAGE.record(
        Usage(
            prompt_tokens=1000,
            completion_tokens=50,
            cost=0.5,
            prompt_tokens_details={"cached_tokens": 400, "cache_write_tokens": 100},
        )
    )
    llm.USAGE.record(Usage(prompt_tokens=10, completion_tokens=None))
    assert (
        llm.USAGE["requests"] == 2
        and llm.USAGE["input"] == 1010
        and llm.USAGE["output"] == 50
    )
    assert llm.USAGE["cache_read"] == 400 and llm.USAGE["cache_write"] == 100
    s = llm.usage_summary()
    assert "2 requests" in s and "1,010 input" in s and "$0.50" in s


def test_retry_waits_then_succeeds(monkeypatch):
    sleeps, n = [], Counter()
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    def flaky():
        n["c"] += 1
        if n["c"] < 3:
            raise ValueError("boom")
        return "ok"

    assert llm.retry(flaky) == "ok" and sleeps == [5, 20]


def test_retry_reraises_after_three_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm.time, "sleep", sleeps.append)

    def bad():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        llm.retry(bad)
    assert sleeps == [5, 20]


def test_chat_strips_think_and_reports_tokens(monkeypatch):
    fc = FakeClient(
        "reasoning</think>  Answer ", Usage(prompt_tokens=5, completion_tokens=7)
    )
    monkeypatch.setattr(llm, "client", lambda: fc)
    text, used = llm.chat("openai/x", [{"role": "user", "content": "q"}])
    assert (text, used) == ("Answer", 7)
    assert fc.calls[0]["extra_body"] == {"reasoning": {"effort": "medium"}}
    assert (
        fc.calls[0]["messages"][-1]["content"] == "q"
    )  # no cache breakpoint for non-Anthropic


def test_chat_anthropic_breakpoint_and_no_reasoning(monkeypatch):
    fc = FakeClient(None, None)
    monkeypatch.setattr(llm, "client", lambda: fc)
    text, used = llm.chat(
        "anthropic/claude", [{"role": "user", "content": "q"}], reasoning=False
    )
    assert (text, used) == ("", 0)
    assert fc.calls[0]["extra_body"] == {"reasoning": {"enabled": False}}
    assert fc.calls[0]["messages"][-1]["content"][0]["cache_control"]


def test_check_models(monkeypatch):
    models = SimpleNamespace(
        list=lambda: SimpleNamespace(data=[SimpleNamespace(id="a")])
    )
    monkeypatch.setattr(llm, "client", lambda: SimpleNamespace(models=models))
    llm.check_models(["a"])
    with pytest.raises(SystemExit):
        llm.check_models(["a", "b"])


def test_jev_classify(monkeypatch):
    sent = {}

    def post(url, json):
        sent.update(url=url, json=json)
        return FakeResp(
            {"model": jev.JEV_MODEL, "answers": {"denominal": {"noul": "0.75"}}}
        )

    monkeypatch.setattr(jev, "client", lambda: SimpleNamespace(post=post))
    assert jev.classify("denominal", "x [[y]]") == 0.75
    assert (
        sent["json"]["state"] == "x [[y]]" and "denominal" in sent["json"]["questions"]
    )


def test_jev_model_mismatch(monkeypatch):
    def post(url, json):
        return FakeResp({"model": "jev-9", "answers": {"denominal": {"noul": 1}}})

    monkeypatch.setattr(jev, "client", lambda: SimpleNamespace(post=post))
    with pytest.raises(RuntimeError, match="jev-9"):
        jev.classify("denominal", "t")


def test_jev_retries_http_errors(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    n = Counter()

    def post(url, json):
        n["c"] += 1
        if n["c"] == 1:
            raise RuntimeError("429")
        return FakeResp(
            {"model": jev.JEV_MODEL, "answers": {"inanimate": {"noul": 0.1}}}
        )

    monkeypatch.setattr(jev, "client", lambda: SimpleNamespace(post=post))
    assert jev.classify("inanimate", "t") == 0.1 and n["c"] == 2


def test_missing_api_key_names_the_variable(monkeypatch):
    llm.client.cache_clear()
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(SystemExit, match="OPENROUTER_API_KEY"):
        llm.client()
    llm.client.cache_clear()


def test_retry_logs_each_failure(monkeypatch, caplog):
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(ValueError):
        llm.retry(lambda: (_ for _ in ()).throw(ValueError("boom")))
    assert [r.levelname for r in caplog.records] == ["WARNING", "WARNING"]
    assert "boom" in caplog.records[0].getMessage()


def test_chat_sends_exactly_one_request(monkeypatch):
    fc = FakeClient("x", Usage(prompt_tokens=1, completion_tokens=2))
    monkeypatch.setattr(llm, "client", lambda: fc)
    llm.chat("openai/x", [{"role": "user", "content": "q"}])
    assert len(fc.calls) == 1 and llm.USAGE["requests"] == 1
