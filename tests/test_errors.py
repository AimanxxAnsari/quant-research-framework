"""The error family: specific errors that still behave like the built-ins they replace."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_research import errors
from quant_research.stats.returns import cagr
from quant_research.validation.overfitting import expected_max_sharpe


def test_input_errors_are_value_errors():
    # Code that catches ValueError must keep working after the switch to specific errors.
    for cls in (
        errors.EmptySeriesError,
        errors.MisalignedDataError,
        errors.MissingValuesError,
        errors.InvalidParameterError,
        errors.ConfigError,
    ):
        assert issubclass(cls, errors.ResearchInputError)
        assert issubclass(cls, ValueError)
        assert issubclass(cls, errors.QuantResearchError)
    assert issubclass(errors.DataNotFoundError, KeyError)
    assert issubclass(errors.DataIntegrityError, ValueError)


def test_data_not_found_message_is_not_quoted():
    assert str(errors.DataNotFoundError("ticker not in warehouse: XYZ")) == (
        "ticker not in warehouse: XYZ"
    )


def test_cagr_of_nothing_fails_loudly():
    with pytest.raises(errors.EmptySeriesError):
        cagr(pd.Series([np.nan, np.nan]))


def test_invalid_trial_count_is_a_parameter_error():
    with pytest.raises(errors.InvalidParameterError):
        expected_max_sharpe(0)
