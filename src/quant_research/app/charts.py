"""Plotly figures for the app, coloured with the repo palette from ``plotting.py``.

Every function takes the theme (light or dark) so the charts match the README figures and
the reader's Streamlit theme.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from quant_research.app.lab import LabRun
from quant_research.plotting import DARK, LIGHT, Theme


def theme_for(kind: str | None) -> Theme:
    return DARK if kind == "dark" else LIGHT


def _layout(fig: go.Figure, t: Theme, title: str, y_title: str, height: int = 320) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "font": {"size": 15, "color": t.ink}},
        paper_bgcolor=t.surface,
        plot_bgcolor=t.surface,
        font={"color": t.ink2, "family": "Helvetica Neue, Helvetica, Arial, sans-serif"},
        height=height,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "yanchor": "bottom"},
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor=t.baseline, tickfont={"color": t.muted})
    fig.update_yaxes(title=y_title, gridcolor=t.grid, zeroline=False, tickfont={"color": t.muted})
    return fig


def _line(x: pd.Index, y: pd.Series, name: str, colour: str, dash: str = "solid") -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name, mode="lines", line={"color": colour, "width": 2, "dash": dash}
    )


def _log_ticks(lo: float, hi: float) -> list[float]:
    """1, 2, 5 steps of each power of ten that fall inside [lo, hi]."""
    out = []
    for p in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1):
        out += [m * 10.0**p for m in (1, 2, 5) if lo <= m * 10.0**p <= hi]
    return [float(f"{v:.6g}") for v in out]


def equity(run: LabRun, engine: str, t: Theme) -> go.Figure:
    """Growth of 1 for the rule and its benchmark, on a log scale."""
    r = run.runs[engine]
    nav, bench = r.result.nav, r.benchmark.nav
    fig = go.Figure(
        [
            _line(nav.index, nav / nav.iloc[0], "rule", t.series[0]),
            _line(bench.index, bench / bench.iloc[0], "benchmark", t.series[1]),
        ]
    )
    ticks = _log_ticks(
        min(nav.min() / nav.iloc[0], bench.min() / bench.iloc[0]),
        max(nav.max() / nav.iloc[0], bench.max() / bench.iloc[0]),
    )
    fig.update_yaxes(type="log", tickvals=ticks, ticktext=[f"{v:g}" for v in ticks])
    return _layout(fig, t, f"Growth of 1, {engine} engine", "growth of 1 (log)")


def drawdown(run: LabRun, engine: str, t: Theme) -> go.Figure:
    r = run.runs[engine]
    traces = []
    for label, nav, colour in (
        ("rule", r.result.nav, t.series[0]),
        ("benchmark", r.benchmark.nav, t.series[1]),
    ):
        traces.append(_line(nav.index, nav / nav.cummax() - 1.0, label, colour))
    fig = go.Figure(traces)
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, t, f"Drawdown, {engine} engine", "below previous peak", height=260)


def position(run: LabRun, engine: str, t: Theme) -> go.Figure:
    w = run.runs[engine].result.weights[run.asset]
    fig = go.Figure(
        [
            go.Scatter(
                x=w.index,
                y=w,
                name="weight",
                mode="lines",
                line={"color": t.series[2], "width": 1.2, "shape": "hv"},
            )
        ]
    )
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, t, f"Position in {run.asset}, {engine} engine", "weight of NAV", height=220)


def nav_gap(run: LabRun, t: Theme) -> go.Figure:
    gap = run.nav_gap()
    fig = go.Figure([_line(gap.index, gap, "loop / weights - 1", t.series[6])])
    fig.update_yaxes(exponentformat="e")
    return _layout(fig, t, "NAV gap between the engines", "relative difference", height=260)


def timings(run: LabRun, t: Theme) -> go.Figure:
    names = list(run.runs)
    secs = [run.runs[n].seconds for n in names]
    fig = go.Figure(
        [
            go.Bar(
                x=secs,
                y=names,
                orientation="h",
                marker={"color": t.series[0]},
                text=[f"{s:.3f} s" for s in secs],
                textposition="outside",
            )
        ]
    )
    # room on the right for the longest bar's label
    fig.update_xaxes(title="seconds", gridcolor=t.grid, range=[0, max(secs) * 1.3])
    fig.update_yaxes(title=None, gridcolor=t.surface)
    fig = _layout(fig, t, "Run time of the rule", "", height=200)
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig


def sharpe_histogram(trials: pd.DataFrame, column: str, t: Theme) -> go.Figure:
    """Distribution of the active Sharpe ratio across trials, one colour per asset class."""
    fig = go.Figure()
    for i, (cls, group) in enumerate(trials.groupby("class")):
        fig.add_trace(
            go.Histogram(
                x=group[column],
                name=str(cls).upper(),
                nbinsx=40,
                marker={"color": t.series[i]},
                opacity=0.75,
            )
        )
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=0, line={"color": t.baseline, "width": 1})
    fig = _layout(fig, t, "Development active Sharpe of every trial", "trials")
    fig.update_xaxes(title="annualised active Sharpe")
    fig.update_layout(hovermode="closest")
    return fig


def holdout_scatter(top: pd.DataFrame, t: Theme) -> go.Figure:
    fig = go.Figure(
        [
            go.Scatter(
                x=top["development Sharpe"],
                y=top["holdout Sharpe"],
                mode="markers",
                text=top.index,
                marker={"color": t.series[0], "size": 9},
                hovertemplate="%{text}<br>development %{x:.2f}<br>holdout %{y:.2f}<extra></extra>",
            )
        ]
    )
    fig.add_hline(y=0, line={"color": t.baseline, "width": 1})
    fig = _layout(fig, t, "Forwarded trials: development vs holdout", "holdout active Sharpe")
    fig.update_xaxes(title="development active Sharpe")
    fig.update_layout(hovermode="closest", showlegend=False)
    return fig
