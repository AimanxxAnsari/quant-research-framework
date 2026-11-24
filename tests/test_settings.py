import pytest

from quant_research.data.settings import secret


def test_missing_secret_error_is_clear(monkeypatch):
    secret.cache_clear()
    monkeypatch.delenv("QRF_TEST_KEY", raising=False)
    with pytest.raises(RuntimeError, match="QRF_TEST_KEY is not set"):
        secret("QRF_TEST_KEY")


def test_config_files_load(settings):
    inst = settings.instruments()
    assert "DGS10" in inst["fred"]
    assert set(settings.sources()) >= {"fred", "treasury", "french", "tiingo"}
