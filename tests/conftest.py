"""Synthetic payloads in each source's real format (no vendor data is committed)."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from quant_research.data.settings import Settings

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(root=REPO, data_dir=tmp_path / "data", config_dir=REPO / "config")


def fred_payload(values: dict[str, str]) -> bytes:
    obs = [
        {"realtime_start": "2026-01-01", "realtime_end": "2026-01-01", "date": d, "value": v}
        for d, v in values.items()
    ]
    return json.dumps({"observations": obs}).encode()


def treasury_payload(ten_year: str = "4.10") -> bytes:
    return (
        'Date,"1 Mo","1.5 Month","2 Mo","10 Yr","30 Yr"\n'
        f"01/03/2024,5.55,5.54,5.50,{ten_year},4.20\n"
        "01/02/2024,5.55,,5.51,3.95,4.08\n"
    ).encode()


def french_payload() -> bytes:
    text = (
        "This file was created by CMPT_ME_BEME_RETS_DAILY using the 202609 CRSP database.\n"
        "The Tbill return is the simple daily rate.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19260701,    0.09,   -0.25,   -0.27,    0.01\n"
        "19260702,    0.45,   -0.33,   -0.06,    0.01\n"
        "\n"
        "Copyright 2026 Kenneth R. French\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("F-F_Research_Data_Factors_daily.CSV", text)
    return buf.getvalue()


def tiingo_payload(closes=(100.0, 101.0)) -> bytes:
    rows = []
    for i, c in enumerate(closes):
        rows.append(
            {
                "date": f"2024-01-0{i + 2}T00:00:00.000Z",
                "open": c,
                "high": c + 1,
                "low": c - 1,
                "close": c,
                "volume": 1000,
                "adjOpen": c,
                "adjHigh": c + 1,
                "adjLow": c - 1,
                "adjClose": c,
                "adjVolume": 1000,
                "divCash": 0.0,
                "splitFactor": 1.0,
            }
        )
    return json.dumps(rows).encode()
