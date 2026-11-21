import sys

from repro import analyze, collect, roundtrip
from repro import run as run_mod
from repro.prompts import MODELS


def cfg(tmp_path):
    p = tmp_path / "e.toml"
    p.write_text(f'models = ["{MODELS[0].slug}"]\nsettings = ["chat", "essay"]\n')
    return str(p)


def patch_stages(monkeypatch):
    calls = []
    monkeypatch.setattr(
        collect,
        "run",
        lambda exp, setting, workers: calls.append(("collect", setting, workers)),
    )
    monkeypatch.setattr(
        roundtrip, "run", lambda exp: calls.append(("roundtrip", exp.translator))
    )
    monkeypatch.setattr(
        analyze,
        "run",
        lambda exp: calls.append(("analyze", exp.judge, exp.spacy, exp.limit)),
    )
    return calls


def test_all_stages_in_order(monkeypatch, tmp_path):
    calls = patch_stages(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            cfg(tmp_path),
            "--no-judge",
            "--translator",
            "T",
            "--spacy",
            "sm",
            "--limit",
            "2",
        ],
    )
    run_mod.main()
    assert calls == [
        ("collect", "chat", 32),
        ("collect", "essay", 32),
        ("roundtrip", "T"),
        ("analyze", False, "sm", 2),
    ]


def test_stage_selection(monkeypatch, tmp_path):
    calls = patch_stages(monkeypatch)
    monkeypatch.setattr(
        sys, "argv", ["run", cfg(tmp_path), "--stages", "roundtrip", "--workers", "4"]
    )
    run_mod.main()
    assert [c[0] for c in calls] == ["roundtrip"]
