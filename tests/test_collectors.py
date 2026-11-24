import pandas as pd
from conftest import fred_payload, french_payload, tiingo_payload, treasury_payload

from quant_research.data.collectors import fred, french, tiingo, treasury


def test_fred_parse_drops_missing_dots():
    df = fred.parse(fred_payload({"2024-01-01": ".", "2024-01-02": "3.95"}), "DGS10")
    assert df.to_dict("records") == [
        {"series_id": "DGS10", "date": pd.Timestamp("2024-01-02"), "value": 3.95}
    ]


def test_treasury_tenor_codes():
    assert treasury.tenor_code("1 Mo") == "1M"
    assert treasury.tenor_code("1.5 Month") == "1.5M"
    assert treasury.tenor_code("10 Yr") == "10Y"
    assert treasury.tenor_code("Date") is None


def test_treasury_parse_long_format():
    df = treasury.parse(treasury_payload(), "nominal_2024")
    assert set(df["series_id"]) == {
        "UST_PAR_1M",
        "UST_PAR_1.5M",
        "UST_PAR_2M",
        "UST_PAR_10Y",
        "UST_PAR_30Y",
    }
    ten = df[df["series_id"] == "UST_PAR_10Y"].sort_values("date")
    assert ten["value"].tolist() == [3.95, 4.10]
    assert len(df) == 9  # one blank 1.5M value dropped
    assert (
        treasury.parse(treasury_payload(), "real_2024")["series_id"]
        .str.startswith("UST_REAL_")
        .all()
    )


def test_french_parse_skips_preamble_and_footer():
    df = french.parse(french_payload(), "F-F_Research_Data_Factors_daily")
    assert set(df["series_id"]) == {"FF_MKT_RF", "FF_SMB", "FF_HML", "FF_RF"}
    assert len(df) == 8
    mkt = df[df["series_id"] == "FF_MKT_RF"]
    assert mkt["value"].tolist() == [0.09, 0.45]
    assert mkt["date"].iloc[0] == pd.Timestamp("1926-07-01")


def test_tiingo_parse_columns():
    df = tiingo.parse(tiingo_payload(), "SPY")
    assert list(df.columns[:3]) == ["ticker", "date", "open"]
    assert df["adj_close"].tolist() == [100.0, 101.0]
    assert df["date"].iloc[0] == pd.Timestamp("2024-01-02")
