"""The framework's exceptions: one small family, so callers can catch exactly what they mean.

Every class also subclasses the built-in exception it replaces (ValueError, KeyError), so code
that catches the built-in keeps working. The rule is to fail loudly at the boundary: a wrong
number returned quietly is worse than an exception.

    QuantResearchError
    |-- ResearchInputError (ValueError)    bad input passed to a public function
    |   |-- EmptySeriesError               no data where some is required
    |   |-- MisalignedDataError            dates, shapes or columns that do not line up
    |   |-- MissingValuesError             NaNs where none are allowed
    |   |-- InvalidParameterError          a parameter outside its valid range
    |   `-- ConfigError                    configuration that is incomplete or inconsistent
    |-- DataNotFoundError (KeyError)       a series or ticker that is not in the warehouse
    |-- DataIntegrityError (ValueError)    stored data that is corrupt or cannot be parsed
    |-- LookAheadError                     a strategy whose past decisions change when future
    |                                      data is removed (it peeks at the future)
    `-- AccountWipedOutError               a backtest whose net asset value reached zero
"""

from __future__ import annotations


class QuantResearchError(Exception):
    """Base class for every error raised on purpose by the framework."""


class ResearchInputError(QuantResearchError, ValueError):
    """Bad input passed to a public function."""


class EmptySeriesError(ResearchInputError):
    """A series or table is empty where data is required."""


class MisalignedDataError(ResearchInputError):
    """Inputs whose dates, shapes or columns do not line up."""


class MissingValuesError(ResearchInputError):
    """NaNs where the calculation does not allow them."""


class InvalidParameterError(ResearchInputError):
    """A parameter outside its valid range, for example a negative volatility."""


class ConfigError(ResearchInputError):
    """Configuration that is incomplete or inconsistent."""


class DataNotFoundError(QuantResearchError, KeyError):
    """A requested series or ticker is not in the warehouse."""

    def __str__(self) -> str:
        # KeyError wraps its message in quotes; show the message as written instead.
        return str(self.args[0]) if self.args else ""


class DataIntegrityError(QuantResearchError, ValueError):
    """Stored data is corrupt or cannot be parsed."""


class LookAheadError(QuantResearchError):
    """A strategy's decision at some date depends on data from after that date."""


class AccountWipedOutError(QuantResearchError):
    """A backtest's net asset value fell to zero or below, so it cannot continue."""
