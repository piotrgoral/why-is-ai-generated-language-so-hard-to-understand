import pytest

from repro import experiment
from repro.prompts import MODELS

SLUG = MODELS[0].slug


def write(tmp_path, body, name="exp.toml"):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_default():
    e = experiment.load()
    assert e.name == "default" and e.models == [m.slug for m in MODELS]
    assert e.model_specs() == list(MODELS)


def test_load_derives_paths_and_sections(tmp_path):
    p = write(
        tmp_path,
        f'models = ["{SLUG}"]\nlimit = 3\n[roundtrip]\ntranslator = "t"\n[analyze]\njudge = false\n',
    )
    e = experiment.load(p)
    assert e.name == "exp" and e.limit == 3 and e.translator == "t" and e.judge is False
    assert (
        str(e.raw) == "runs/exp/raw.jsonl"
        and str(e.sents) == "runs/exp/sentences.jsonl"
    )
    assert str(e.results) == "runs/exp/results"
    assert e.model_specs() == [MODELS[0]]


def test_load_name_and_raw_override(tmp_path):
    p = write(
        tmp_path, f'name = "mine"\nraw = "runs/other/raw.jsonl"\nmodels = ["{SLUG}"]\n'
    )
    e = experiment.load(p)
    assert (
        e.name == "mine"
        and str(e.raw) == "runs/other/raw.jsonl"
        and str(e.sents) == "runs/mine/sentences.jsonl"
    )


@pytest.mark.parametrize(
    "body",
    ["bogus = 1\n", '[roundtrip]\nspacy = "x"\n', '[analyze]\ntranslator = "x"\n'],
)
def test_unknown_keys(tmp_path, body):
    with pytest.raises(SystemExit, match="unknown keys"):
        experiment.load(write(tmp_path, body))


def test_unknown_model(tmp_path):
    with pytest.raises(SystemExit, match="not in prompts.MODELS"):
        experiment.load(write(tmp_path, 'models = ["nope/nope"]\n'))


def test_unknown_setting(tmp_path):
    with pytest.raises(SystemExit, match="unknown settings"):
        experiment.load(write(tmp_path, f'models = ["{SLUG}"]\nsettings = ["poem"]\n'))
