import argparse
import logging

from repro import cli
from repro.experiment import Experiment


def test_apply_overrides_copies_only_given_options():
    exp = Experiment(limit=5)
    args = argparse.Namespace(limit=None, translator="T", spacy=None)
    cli.apply_overrides(exp, args, "limit", "translator", "spacy")
    assert exp.limit == 5 and exp.translator == "T" and exp.spacy == "en_core_web_trf"


def test_configure_logging_installs_a_root_handler(monkeypatch):
    monkeypatch.setattr(logging.root, "handlers", [])
    cli.configure_logging()
    assert logging.root.handlers and logging.root.level == logging.INFO
