import pytest

from repro import settings


def test_keys_come_from_the_environment():
    assert settings.openrouter_api_key() == "test"
    assert settings.jev_api_key() == "test"


@pytest.mark.parametrize("value", [None, ""])
def test_missing_or_empty_key_is_a_clear_error(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("JEV_API_KEY")
    else:
        monkeypatch.setenv("JEV_API_KEY", value)
    with pytest.raises(SystemExit, match="JEV_API_KEY"):
        settings.jev_api_key()
