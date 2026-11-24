# Market-data basics for research

## Observation time, publication time, retrieval time
- **Observation time:** the date a value refers to (CPI for March).
- **Publication time:** when the market could first know it (mid-April).
- **Retrieval time:** when we downloaded it.
A backtest may only use a value after its publication time. FRED's ALFRED service keeps
historical vintages for exactly this reason.

## Par yields vs constant-maturity yields
Treasury "par yield curve" values are read off a fitted curve at fixed tenors; they are not the
yield of any specific traded bond. FRED's DGS series are the same constant-maturity data.

## Real yields and breakevens
10Y breakeven inflation ~= 10Y nominal yield - 10Y TIPS real yield (DGS10 - DFII10 ~= T10YIE).

## FX quote conventions
EUR/USD 1.08 means 1 EUR = 1.08 USD (FRED DEXUSEU). USD/JPY 150 means 1 USD = 150 JPY (DEXJPUS).
To express everything as USD per unit of foreign currency, invert the second kind.
H.10 rates are noon New York buying rates, published weekly with a lag; ECB reference rates are
set around 14:10 CET. Never mix fixings from different times in one calculation.

## Adjusted prices
Splits and dividends create jumps in raw prices that are not returns. Adjusted series rescale
history so that returns are correct; raw prices remain the right input for anything tied to
traded levels.

## Volatility indices
VIX, OVX and GVZ are 30-day implied volatilities in annualised percentage points, derived from
option prices. They can spike far above normal in crises (OVX above 300 in April 2020).
