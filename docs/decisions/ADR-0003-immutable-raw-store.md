# ADR-0003: Immutable raw store with content hashing

- **Date:** 2026-11-24
- **Status:** accepted

## Context
Sources revise data and change formats. To trust any result we must know exactly which bytes it
came from, and be able to rebuild clean tables without re-downloading.

## Options considered
1. **Overwrite a single file per dataset.** Simple; loses history and lineage.
2. **Store every download.** Full history; wastes space when nothing changed (most days).
3. **Store a download only when its SHA-256 differs from the latest snapshot; log every fetch.**

## Decision
Option 3. Path `raw/<source>/<dataset>/<UTC time>_<sha12>.<ext>`; files written atomically and
made read-only; `manifest.jsonl` records every fetch as `stored` or `unchanged` with the redacted
URL, size, hash and time. Reads verify the checksum. Clean tables use the latest stored snapshot
per dataset and carry `retrieved_at` and `raw_sha256` on every row.

## Consequences
- Rebuilding from the same raw layer is deterministic (tested).
- Every clean row traces back to the exact file it came from.
- Tampering or corruption of a raw file is detected on read.
- Revisions are kept: older snapshots can be compared to study data revisions.
