import pandas as pd
import pytest

from quant_research.monitor import _changes
from quant_research.reporting import inject, md_table


def test_inject_replaces_only_marked_block():
    readme = "intro\n<!-- X:START -->\nold\n<!-- X:END -->\noutro\n"
    new = inject(readme, "X", "NEW")
    assert new.startswith("intro\n") and new.endswith("outro\n")
    assert "old" not in new and "NEW" in new
    assert inject(new, "X", "AGAIN").count("<!-- X:START -->") == 1
    with pytest.raises(ValueError):
        inject("no markers", "X", "y")


def test_md_table_per_column_formats():
    df = pd.DataFrame({"a": [0.12345], "b": [2.0]}, index=pd.Index(["r"], name="idx"))
    out = md_table(df, {"a": "{:.1%}", "b": "{:.0f}"})
    assert out.splitlines()[0] == "| idx | a | b |"
    assert out.splitlines()[2] == "| r | 12.3% | 2 |"


def test_monitor_changes_bp_and_percent():
    idx = pd.bdate_range("2025-12-01", periods=80)
    s = pd.Series(range(100, 180), index=idx, dtype=float)
    pct = _changes(s, bp=False)
    assert pct["1W"] == pytest.approx((179 / 174 - 1) * 100)
    assert pct["YTD"] == pytest.approx((179 / s[s.index < "2026-01-01"].iloc[-1] - 1) * 100)
    y = pd.Series([4.00] * 70 + [4.25] * 10, index=idx)
    assert _changes(y, bp=True)["1M"] == pytest.approx(25.0)


def test_md_table_escapes_pipes_in_labels():
    import pandas as pd

    from quant_research.reporting import md_table

    df = pd.DataFrame({"x": [1.0], "note": ["a|b"]}, index=pd.Index(["SPY|MA"], name="trial"))
    row = md_table(df).splitlines()[-1]
    assert row == "| SPY\\|MA | 1.000 | a\\|b |"
