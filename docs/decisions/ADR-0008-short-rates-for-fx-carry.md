# ADR-0008: Short rates for FX carry

- **Date:** 2026-11-26
- **Status:** accepted

## Context
The Pillar 4 backtester will hold FX spot positions. A long position in a foreign currency
earns that currency's short rate and pays the dollar rate, so the engine needs a daily short
rate for every currency in the FX universe (USD, EUR, GBP, JPY, CHF, AUD, CAD, NOK, NZD, SEK).
The budget is zero, so the source has to be free, and FRED (already used) was checked first.

What FRED has, checked on 2026-11-26:

| Currency | Daily overnight rate on FRED | Monthly fallback (OECD) |
|---|---|---|
| USD | Fed funds `DFF` (1954 on), SOFR (2018 on) | not needed |
| EUR | EONIA `EONIARATE` (1999 to 2021), ESTR (Oct 2019 on), ECB policy rates | not needed |
| GBP | SONIA `IUDSOIA` (1997 on) | not needed |
| JPY, CHF, AUD, CAD, NOK | none found | call-money rate `IRSTCI01xxM156N`, current |
| NZD, SEK | none found | call money discontinued (NZD 2024, SEK 2020); 3-month interbank `IR3TIB01xxM156N` is current |

## Decision
- **Daily where FRED has an overnight rate:** USD fed funds; EUR EONIA until 30 Sep 2019, then
  ESTR; GBP SONIA.
- **Monthly OECD averages for the other six currencies**: call money where it is still
  published, 3-month interbank for NZD and SEK. A monthly value applies to every day of its
  month and is carried at most one more month while the next is unpublished.
- Splices and day counts live in `short_rates` in `config/instruments.yaml`;
  `quant_research.data.rates` builds one calendar-day series per currency and raises
  `MissingValuesError` on any gap longer than the fill limit (7 days daily, 62 days monthly).
  `qrf check` reports one PASS / FAIL row per currency.
- **Day counts** follow each market's convention: ACT/360 for USD, EUR, CHF, NOK, SEK and
  ACT/365 for GBP, JPY, AUD, CAD, NZD.
- **Accounting, not signals:** these are the rates earned over each day, which is right for
  carry accrual. A strategy that *trades on* rate differentials must lag them first, because a
  monthly average is published weeks after the month starts.

## Rejected
- **SOFR for USD:** it starts in 2018; fed funds covers the whole ETF history with one
  definition.
- **ECB policy rates for EUR** (main refinancing rate before Oct 2008, deposit rate after): the
  first design, before EONIA turned up on FRED. Measured against EONIA, the deposit rate was
  13.6 bp too low on average from Oct 2008 to Sep 2019, with a worst monthly average of 62 bp
  (June 2011) and a worst day of 122 bp. FRED's main refinancing series is also blank from June
  2000 to October 2008 (variable-rate tenders), which the gap check caught on the first run.
- **OECD monthly for every currency:** simpler and uniform, but throws away daily data where it
  exists.
- **3-month rates everywhere:** a term rate is not what an overnight FX position earns; only used
  where no overnight series is current.
- **Scraping central-bank websites for daily rates:** ten different formats and terms of use
  for little gain at daily-bar frequency.

## Consequences
- The EUR splice has a step of about 10 bp on 1 Oct 2019 (EONIA was redefined as ESTR + 8.5 bp
  from that date).
- For JPY, CHF, AUD, CAD, NOK, NZD and SEK, rate changes inside a month appear as a smoothed
  monthly average, so carry around a policy move is approximate. NZD and SEK use a 3-month
  rate, which embeds expected policy moves.
- Two short rates exceeded the old 25% sanity bound for rates (NZD 26.5% in 1985, NOK call
  money 28.5% in 1982); the bound was widened to 40% (ADR-0004 rule).
