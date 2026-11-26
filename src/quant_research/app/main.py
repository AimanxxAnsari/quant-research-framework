"""Streamlit app: the engine lab and the Pillar 4 study explorer.

Run with ``qrf app`` (or ``uv run streamlit run src/quant_research/app/main.py``) after
``uv sync --group app``. The demo market needs no data or keys; "my data" uses the local
warehouse built by ``qrf build``. Nothing here changes the pre-registered study.
"""

from __future__ import annotations

from typing import cast

import pandas as pd
import streamlit as st

from quant_research.app import charts, lab
from quant_research.app.study import filter_trials, load_study
from quant_research.backtest.config import FILL_RULES, load_config
from quant_research.backtest.market import Market
from quant_research.data.settings import load_settings
from quant_research.errors import QuantResearchError
from quant_research.studies.pillar4_grid import CURRENCIES, ETFS
from quant_research.studies.pillar4_trials import HEADLINE, asset_market

DEMO = "demo (synthetic)"
REAL = "my data"
PAGES = ("Engine lab", "Study explorer")
PERCENT_ROWS = {
    "total return", "CAGR", "volatility", "max drawdown", "hit rate", "exposure",
    "cost drag per year",
}  # fmt: skip


@st.cache_data(show_spinner=False)
def _demo_market(asset: str) -> Market:
    return lab.single_asset(lab.demo_market(), asset)


@st.cache_data(show_spinner=False)
def _real_market(asset: str, start: str | None) -> Market:
    return lab.single_asset(asset_market(load_settings(), asset, end=None), asset, start)


@st.cache_data(show_spinner=False)
def _run(
    source: str,
    asset: str,
    family: str,
    params: tuple[tuple[str, float], ...],
    allow_short: bool,
    fill: str,
    cost_multiplier: float,
    engine_choice: str,
    start: str | None = None,
) -> lab.LabRun:
    market = _demo_market(asset) if source == DEMO else _real_market(asset, start)
    rule = lab.build_rule(family, asset, allow_short, dict(params))
    config = lab.lab_config(load_config(load_settings()), fill, cost_multiplier)
    return lab.run_lab(market, rule, asset, config, lab.ENGINE_CHOICES[engine_choice])


def _has_local_data() -> bool:
    return (load_settings().clean_dir / "prices.parquet").exists()


def _format(table: pd.DataFrame) -> pd.DataFrame:
    out = table.astype(object)
    for row in table.index:
        fmt = "{:.2%}" if row in PERCENT_ROWS else "{:.3f}"
        out.loc[row] = [fmt.format(v) for v in table.loc[row]]
    return out


def _sidebar_inputs() -> dict[str, object]:
    sb = st.sidebar
    sources = [DEMO, REAL] if _has_local_data() else [DEMO]
    source = sb.radio("Market", sources, key="source")
    if source == DEMO:
        market = lab.demo_market()
        assets, classes = market.assets, dict(market.asset_class)
    else:
        assets = [*ETFS, *CURRENCIES]
        classes = {a: ("fx" if a in CURRENCIES else "etf") for a in assets}
    asset = sb.selectbox("Asset", assets, key="asset")
    asset_class = classes[asset]
    family = sb.selectbox("Rule family", lab.families_for(asset_class), key="family")
    params = {}
    for name, default in lab.rule_parameters(family).items():
        step: int | float = 1 if isinstance(default, int) else 0.01
        params[name] = sb.number_input(name, value=default, step=step, key=f"p_{family}_{name}")
    # one widget per class, so a currency's "short" setting never carries over to an ETF
    allow_short = asset_class == "fx" and sb.checkbox(
        "Allow shorts",
        value=True,
        key="short_fx",
        help="As in the study: currencies long/short, ETFs long/flat.",
    )
    start = None
    if source == REAL:
        # the loop engine re-runs the rule on each bar's history, so time grows with length^2
        first = sb.number_input(
            "First year",
            1980,
            2026,
            2000,
            key="first_year",
            help="Shorter histories make the loop engine much faster.",
        )
        start = f"{int(first)}-01-01"
    fill = sb.radio("Fill", FILL_RULES, key="fill", horizontal=True)
    mult = sb.slider("Cost multiplier", 0.0, 3.0, 1.0, 0.5, key="costs")
    return {
        "source": source, "asset": asset, "family": family,
        "params": tuple((k, float(v)) for k, v in params.items()),
        "allow_short": bool(allow_short), "fill": fill, "cost_multiplier": float(mult),
        "start": start,
    }  # fmt: skip


def _one_engine(run: lab.LabRun, engine: str, t: charts.Theme) -> None:
    r = run.runs[engine]
    st.caption(f"{engine} engine: {r.seconds:.3f} s for the rule")
    st.dataframe(_format(run.metrics_table(engine)), width="stretch")
    st.plotly_chart(charts.equity(run, engine, t), width="stretch", key=f"eq_{engine}")
    st.plotly_chart(charts.drawdown(run, engine, t), width="stretch", key=f"dd_{engine}")
    st.plotly_chart(charts.position(run, engine, t), width="stretch", key=f"pos_{engine}")


def _comparison(run: lab.LabRun, t: charts.Theme) -> None:
    st.subheader("Engine comparison")
    gap = run.max_gap()
    w, lo = run.runs["weights"].seconds, run.runs["loop"].seconds
    c1, c2, c3 = st.columns(3)
    c1.metric("Largest NAV gap", f"{gap:.1e}")
    c2.metric("Loop / weights run time", f"{lo / w:.0f}x" if w > 0 else "n/a")
    n = run.lookahead_checks
    c3.metric("Look-ahead check", f"passed on {n} dates" if n else "switched off in config")
    st.markdown(
        "The weights engine works in return space; the loop engine keeps units and cash bar by "
        "bar and cannot read the future. Matching numbers mean the fast engine's algebra is "
        "right for this rule. The look-ahead check re-runs the rule on data cut off at sampled "
        "dates and would have stopped the run if any past decision changed."
    )
    st.dataframe(run.comparison().style.format("{:.3g}"), width="stretch")
    st.plotly_chart(charts.nav_gap(run, t), width="stretch", key="gap")
    st.plotly_chart(charts.timings(run, t), width="stretch", key="time")


def engine_lab(t: charts.Theme) -> None:
    st.title("Engine lab")
    st.info(
        "In-sample exploration only. Rules and parameters picked here are not part of the "
        "pre-registered Pillar 4 study, and a good-looking result here is exactly the kind of "
        "search the study's tests correct for."
    )
    inputs = _sidebar_inputs()
    choice = st.radio("Engine", tuple(lab.ENGINE_CHOICES), horizontal=True, key="engine")
    try:
        with st.spinner("Running (the loop engine takes a few seconds on long histories)"):
            run = _run(engine_choice=choice, **inputs)  # type: ignore[arg-type]
    except QuantResearchError as e:
        st.error(f"{type(e).__name__}: {e}")
        return
    bench = "buy-and-hold of the same ETF" if run.asset_class == "etf" else "cash"
    st.markdown(f"**{run.rule}**, benchmark: {bench}")
    if choice == "both":
        left, right = st.columns(2)
        with left:
            _one_engine(run, "weights", t)
        with right:
            _one_engine(run, "loop", t)
        _comparison(run, t)
    else:
        _one_engine(run, choice, t)


def study_explorer(t: charts.Theme) -> None:
    st.title("Pillar 4 study explorer")
    try:
        res = load_study(load_settings().root)
    except QuantResearchError as e:
        st.error(str(e))
        return
    m = res.manifest
    st.caption(
        f"Read-only view of the committed results: commit {str(m['git_commit'])[:7]}, data "
        f"through {m['data_through']}, development period to {res.summary['period_end']}."
    )
    h = res.summary["h1_h2"]
    c = st.columns(4)
    c[0].metric("Trials", f"{res.summary['trials']:,}")
    c[1].metric("SPA p-value, ETFs", f"{h['etf']['spa_p']:.3f}")
    c[2].metric("SPA p-value, FX", f"{h['fx']['spa_p']:.3f}")
    survivors = len(h["bhy_survivors"]) + len(h["all"]["stepm_survivors"])
    c[3].metric("Survivors (BHY, StepM)", survivors)
    st.plotly_chart(charts.sharpe_histogram(res.trials, HEADLINE, t), width="stretch")

    st.subheader("Trials")
    f1, f2, f3 = st.columns(3)
    classes = f1.multiselect("Class", sorted(res.trials["class"].unique()), key="f_class")
    families = f2.multiselect("Family", sorted(res.trials["family"].unique()), key="f_family")
    assets = f3.multiselect("Asset", sorted(res.trials["asset"].unique()), key="f_asset")
    shown = filter_trials(res.trials, classes, families, assets)
    cols = ["asset", "family", "params", HEADLINE, "next_close x1", "next_open x3", "p BHY"]
    st.dataframe(shown[cols].sort_values(HEADLINE, ascending=False), width="stretch")
    st.caption(
        f"{len(shown):,} of {len(res.trials):,} trials. Sharpe columns are annualised "
        "active Sharpe ratios under each fill and cost variant."
    )

    st.subheader("Per asset")
    st.dataframe(res.per_asset, width="stretch")

    st.subheader("Holdout")
    if res.holdout_summary is None or res.holdout_top is None:
        st.write("The holdout has not been opened.")
        return
    hs = res.holdout_summary
    st.caption(f"{hs['holdout_start']} to {hs['holdout_end']}, opened once. {hs['selection']}.")
    st.plotly_chart(charts.holdout_scatter(res.holdout_top, t), width="stretch")
    st.dataframe(res.holdout_top, width="stretch")
    if res.holdout_walk_forward is not None:
        st.dataframe(res.holdout_walk_forward, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Quant research lab", layout="wide")
    t = charts.theme_for(cast(str | None, st.context.theme.type))
    page = st.sidebar.radio("Page", PAGES, key="page")
    if page == PAGES[0]:
        engine_lab(t)
    else:
        study_explorer(t)


main()
