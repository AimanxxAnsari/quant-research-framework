"""Distribution shape: moments, normality test, tail frequencies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def describe(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float | int]:
    r = returns.dropna().to_numpy()
    jb = stats.jarque_bera(r)
    sd = r.std(ddof=1)
    z = (r - r.mean()) / sd
    return {
        "n": int(r.size),
        "mean_ann": float(r.mean() * periods_per_year),
        "vol_ann": float(sd * np.sqrt(periods_per_year)),
        "skew": float(stats.skew(r, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(r, fisher=True, bias=False)),
        "jarque_bera": float(jb.statistic),
        "jb_pvalue": float(jb.pvalue),
        # Share of |z| > 4 vs ~6.3 per 100,000 under a normal distribution.
        "freq_beyond_4sd": float(np.mean(np.abs(z) > 4)),
        "normal_freq_beyond_4sd": float(2 * stats.norm.sf(4)),
        "worst": float(r.min()),
        "best": float(r.max()),
    }
