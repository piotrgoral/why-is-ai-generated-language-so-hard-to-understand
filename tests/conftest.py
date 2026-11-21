import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Dummy keys, and relative data/runs/results paths resolve inside tmp_path."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("JEV_API_KEY", "test")
    monkeypatch.chdir(tmp_path)
