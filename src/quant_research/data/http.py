"""HTTP with timeouts, retries and backoff, and no secrets in logs.

Retries transient failures (connection errors, 429, 5xx) with exponential backoff.
Does not retry other 4xx errors: those are permission or request problems that retrying
cannot fix.
"""

from __future__ import annotations

import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from quant_research.errors import QuantResearchError

USER_AGENT = "quant-research-framework/0.1 (educational research; python-requests)"
SECRET_PARAMS = {"api_key", "apikey", "token", "key", "access_token"}
RETRY_STATUS = {429, 500, 502, 503, 504}


def redact_url(url: str) -> str:
    """Replace secret query-parameter values with REDACTED."""
    parts = urlsplit(url)
    query = [
        (k, "REDACTED" if k.lower() in SECRET_PARAMS else v) for k, v in parse_qsl(parts.query)
    ]
    return urlunsplit(parts._replace(query=urlencode(query)))


class FetchError(QuantResearchError, RuntimeError):
    """A download failed after all retries."""


def get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 2.0,
    session: requests.Session | None = None,
) -> requests.Response:
    s = session or requests.Session()
    h = {"User-Agent": USER_AGENT, **(headers or {})}
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = s.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code in RETRY_STATUS and attempt < retries:
                time.sleep(backoff**attempt)
                continue
            if r.status_code >= 400:
                raise FetchError(f"HTTP {r.status_code} for {redact_url(r.url)}")
            return r
        except requests.RequestException as e:  # network-level problem: retry
            last = e
            if attempt < retries:
                time.sleep(backoff**attempt)
    raise FetchError(f"giving up on {redact_url(url)}: {type(last).__name__}")
