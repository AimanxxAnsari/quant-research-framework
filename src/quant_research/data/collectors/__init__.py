"""Collectors: download raw payloads into the RawStore; parse them into tidy frames.

Each collector exposes ``fetch(store, settings, ...) -> list[Snapshot]`` (network) and
``parse(payload, dataset) -> DataFrame`` (pure, tested offline with fixtures).
"""

from quant_research.data.collectors import french, fred, tiingo, treasury

COLLECTORS = {"fred": fred, "treasury": treasury, "french": french, "tiingo": tiingo}
