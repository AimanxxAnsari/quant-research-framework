# Interview pack: data foundation

## 30-second pitch
"Before any strategy work I built the data layer the way a trading firm would: every download is
stored immutably with a hash, clean tables are rebuilt deterministically from those raw files with
lineage on every row, and quality checks run on every refresh. It covers about 45 rate, FX, vol,
commodity and macro series back to 1947 and 33 ETFs back to 2000. On the first run the checks
caught a real extreme, OVX at 226 on the day oil went negative, and I had to decide whether it was
an error or a real event."

## Likely questions
1. **Why keep raw data separately from clean data?** Reproducibility and lineage: you can rebuild
   and audit, and you can study vendor revisions.
2. **Why hash downloads?** To detect change (store only when different), detect corruption, and
   tie every clean row to exact bytes.
3. **What is point-in-time data and why does it matter?** Using the value as known at that date,
   not today's revised value; otherwise backtests use information that did not exist (look-ahead).
4. **Adjusted vs unadjusted prices?** Adjusted for returns (splits, dividends); unadjusted for
   anything tied to the traded price level (orders, stops, option strikes).
5. **FX quote conventions?** DEXUSEU is USD per EUR, DEXJPUS is JPY per USD; mixing them flips
   signs of returns and correlations.
6. **What makes a good range check?** Flags impossible values, not unusual ones; a real extreme
   should never be "cleaned" away.
7. **How do you keep secrets out of a public repo?** `.env` ignored, headers instead of query
   strings where possible, redaction in logs and manifests, pre-commit scans.
8. **How do you handle a source failing?** Retry transient errors with backoff; do not retry
   permission errors; report per source; other sources still refresh; coverage check fails.

## Must be able to rebuild from scratch
- [ ] Content-addressed snapshot store with a manifest.
- [ ] Retry-with-backoff HTTP wrapper.
- [ ] Wide-to-long reshape of a yield-curve table.
- [ ] A freshness check.

## Numbers to remember
45 series; ~408k observations; 33 ETFs; ~197k price rows; FRED from 1947, Treasury from 1990,
ETFs from 2000.
