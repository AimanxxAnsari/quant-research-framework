"""``qrf`` command line.

qrf fetch   [--source fred treasury french tiingo] [--tickers SPY QQQ]
qrf build                       # raw snapshots -> clean Parquet + DuckDB
qrf check                       # data-quality checks; exit 1 on any FAIL
qrf refresh                     # fetch + build + check
"""

from __future__ import annotations

import argparse
import sys

from quant_research.data.build import build
from quant_research.data.collectors import COLLECTORS
from quant_research.data.http import FetchError
from quant_research.data.quality import health_markdown, run_checks
from quant_research.data.raw_store import RawStore
from quant_research.data.settings import load_settings


def cmd_fetch(args, settings) -> int:
    store = RawStore(settings.raw_dir)
    failures = 0
    for name in args.source:
        kwargs = {"tickers": args.tickers} if name == "tiingo" and args.tickers else {}
        before = len(store.events())
        try:
            snaps = COLLECTORS[name].fetch(store, settings, **kwargs)
        except (FetchError, RuntimeError) as e:
            print(f"[{name}] FAILED: {e}")
            failures += 1
            continue
        new = sum(e.status == "stored" for e in store.events()[before:])
        print(
            f"[{name}] {len(snaps)} datasets fetched: {new} new/changed, "
            f"{len(snaps) - new} unchanged"
        )
    return 1 if failures else 0


def cmd_build(args, settings) -> int:
    counts = build(settings)
    print("built:", ", ".join(f"{k} {v:,} rows" for k, v in counts.items()))
    return 0


def cmd_check(args, settings) -> int:
    results = run_checks(settings)
    report = health_markdown(results, settings)
    (settings.data_dir / "health.md").write_text(report)
    shown = results[results["status"] != "PASS"]
    print(shown.to_string(index=False) if len(shown) else "all checks PASS")
    counts = results["status"].value_counts().to_dict()
    print(
        f"PASS {counts.get('PASS', 0)} | WARN {counts.get('WARN', 0)} | "
        f"FAIL {counts.get('FAIL', 0)}  (full report: {settings.data_dir / 'health.md'})"
    )
    return 1 if counts.get("FAIL", 0) else 0


def cmd_refresh(args, settings) -> int:
    code = cmd_fetch(args, settings)
    cmd_build(args, settings)
    return max(code, cmd_check(args, settings))


def cmd_monitor(args, settings) -> int:
    from quant_research.monitor import write_snapshot

    print(write_snapshot(settings))
    return 0


def cmd_study(args, settings) -> int:
    from quant_research.studies import pillar2, pillar5

    if args.name == "pillar4":
        from quant_research.studies import pillar4, pillar4_report

        pillar4.run_development(settings, recompute=not args.reuse)
        pillar4_report.run(settings)
    else:
        {"pillar2": pillar2.run, "pillar5": pillar5.run}[args.name](settings)
    print(f"{args.name}: results written to README.md and reports/{args.name}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qrf", description="quant-research-framework data tools")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in [("fetch", cmd_fetch), ("refresh", cmd_refresh)]:
        sp = sub.add_parser(name)
        sp.add_argument("--source", nargs="+", choices=list(COLLECTORS), default=list(COLLECTORS))
        sp.add_argument("--tickers", nargs="+", help="Tiingo tickers (default: all configured)")
        sp.set_defaults(func=fn)
    sub.add_parser("build").set_defaults(func=cmd_build)
    sub.add_parser("check").set_defaults(func=cmd_check)
    sub.add_parser("monitor").set_defaults(func=cmd_monitor)
    st = sub.add_parser("study")
    st.add_argument("name", choices=["pillar2", "pillar4", "pillar5"])
    st.add_argument("--reuse", action="store_true", help="pillar4: reuse cached trial returns")
    st.set_defaults(func=cmd_study)
    args = p.parse_args(argv)
    return args.func(args, load_settings())


if __name__ == "__main__":
    sys.exit(main())
