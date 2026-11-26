"""Small typed wrappers for numpy functions applied to pandas objects.

numpy's type stubs say ``np.log(series)`` returns an ndarray, although at runtime it returns a
Series with the same index. These helpers keep the pandas type, so the strict type check can
follow Series and DataFrame through the maths without casts in every formula.
"""

from __future__ import annotations

from typing import cast, overload

import numpy as np
import pandas as pd


@overload
def log(x: pd.Series) -> pd.Series: ...
@overload
def log(x: pd.DataFrame) -> pd.DataFrame: ...
def log(x: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Natural log, element by element, keeping the index and columns."""
    return cast(pd.Series | pd.DataFrame, np.log(x))


def sqrt[F: (pd.Series, pd.DataFrame)](x: F) -> F:
    """Square root, element by element, keeping the index and columns."""
    return cast(F, np.sqrt(x))
