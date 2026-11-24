import stat

import pytest

from quant_research.data.http import redact_url
from quant_research.data.raw_store import RawStore


def test_save_is_read_only_and_recorded(tmp_path):
    store = RawStore(tmp_path)
    snap = store.save(
        "fred", "DGS10", b"abc", url="https://x/y?series_id=DGS10&api_key=SECRET", ext="json"
    )
    path = tmp_path / snap.path
    assert path.read_bytes() == b"abc"
    assert not path.stat().st_mode & stat.S_IWUSR  # read-only
    assert "SECRET" not in store.manifest.read_text()
    assert store.latest("fred", "DGS10") == snap


def test_identical_payload_not_stored_twice(tmp_path):
    store = RawStore(tmp_path)
    first = store.save("fred", "DGS10", b"abc", url="u", ext="json")
    again = store.save("fred", "DGS10", b"abc", url="u", ext="json")
    assert again == first
    assert [e.status for e in store.events()] == ["stored", "unchanged"]
    assert len(list((tmp_path / "fred" / "DGS10").iterdir())) == 1


def test_changed_payload_creates_new_snapshot(tmp_path):
    store = RawStore(tmp_path)
    store.save("fred", "DGS10", b"v1", url="u", ext="json")
    second = store.save("fred", "DGS10", b"v2", url="u", ext="json")
    assert store.latest("fred", "DGS10") == second
    assert len(list((tmp_path / "fred" / "DGS10").iterdir())) == 2


def test_tampered_snapshot_is_detected(tmp_path):
    store = RawStore(tmp_path)
    snap = store.save("fred", "DGS10", b"abc", url="u", ext="json")
    p = tmp_path / snap.path
    p.chmod(0o644)
    p.write_bytes(b"abd")
    with pytest.raises(ValueError, match="checksum"):
        store.read(snap)


def test_redact_url():
    url = "https://api.example.com/p?series_id=X&api_key=abc123&token=t0k"
    red = redact_url(url)
    assert "abc123" not in red and "t0k" not in red and "series_id=X" in red
