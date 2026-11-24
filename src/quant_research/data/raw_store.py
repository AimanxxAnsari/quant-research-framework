"""Immutable raw archive.

Every download is stored exactly as received, with its retrieval time and SHA-256, in

    data/raw/<source>/<dataset>/<retrieved_at>_<sha12>.<ext>

and recorded in ``data/raw/manifest.jsonl``. Files are never overwritten or edited. If a
new download is byte-identical to the latest stored snapshot it is not stored again; the
manifest records an ``unchanged`` event instead. Clean tables are always rebuilt from the
latest snapshot of each dataset, so a rebuild is fully reproducible from the raw layer.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from quant_research.data.http import redact_url


@dataclass(frozen=True)
class Snapshot:
    source: str
    dataset: str
    path: str  # relative to the raw root
    sha256: str
    n_bytes: int
    retrieved_at: str  # ISO-8601 UTC
    url: str  # secrets redacted
    status: str = "stored"  # stored | unchanged


class RawStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.manifest = self.root / "manifest.jsonl"

    # ------------------------------------------------------------------ write
    def save(
        self,
        source: str,
        dataset: str,
        payload: bytes,
        *,
        url: str,
        ext: str,
        retrieved_at: datetime | None = None,
    ) -> Snapshot:
        sha = hashlib.sha256(payload).hexdigest()
        ts = (retrieved_at or datetime.now(UTC)).astimezone(UTC)
        latest = self.latest(source, dataset)
        if latest is not None and latest.sha256 == sha:
            event = Snapshot(
                source, dataset, latest.path, sha, len(payload),
                ts.isoformat(), redact_url(url), "unchanged",
            )
            self._append(event)
            return latest

        rel = Path(source) / dataset / f"{ts.strftime('%Y%m%dT%H%M%S%fZ')}_{sha[:12]}.{ext}"
        dest = self.root / rel
        if dest.exists():
            raise FileExistsError(f"refusing to overwrite raw snapshot {rel}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_bytes(payload)
        os.replace(tmp, dest)  # atomic: no half-written snapshots
        dest.chmod(0o444)  # read-only: raw data is never edited
        snap = Snapshot(
            source, dataset, rel.as_posix(), sha, len(payload), ts.isoformat(), redact_url(url)
        )
        self._append(snap)
        return snap

    def _append(self, snap: Snapshot) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.manifest.open("a") as f:
            f.write(json.dumps(asdict(snap)) + "\n")

    # ------------------------------------------------------------------ read
    def events(self) -> list[Snapshot]:
        if not self.manifest.exists():
            return []
        return [Snapshot(**json.loads(line)) for line in self.manifest.read_text().splitlines()]

    def latest(self, source: str, dataset: str) -> Snapshot | None:
        stored = [
            e for e in self.events()
            if e.source == source and e.dataset == dataset and e.status == "stored"
        ]
        return stored[-1] if stored else None

    def latest_all(self, source: str) -> list[Snapshot]:
        """Latest stored snapshot of every dataset of a source, sorted by dataset."""
        datasets = sorted({e.dataset for e in self.events() if e.source == source})
        return [s for d in datasets if (s := self.latest(source, d)) is not None]

    def read(self, snap: Snapshot) -> bytes:
        data = (self.root / snap.path).read_bytes()
        if hashlib.sha256(data).hexdigest() != snap.sha256:
            raise ValueError(f"checksum mismatch for raw snapshot {snap.path}")
        return data
