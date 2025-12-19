from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd


def _latest_sheet_for_date(date: dt.date) -> Path:
    yyyymmdd = date.strftime("%Y%m%d")
    d = Path("reports/trade_sheets/rule_a_maker_orders") / yyyymmdd
    candidates = sorted(d.glob(f"rule_a_maker_orders_{yyyymmdd}_*.csv"))
    if not candidates:
        raise SystemExit(f"[error] no maker order sheets found under {d}")
    return candidates[-1]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print copy/paste commands for placing Rule A maker orders (does not trade; prints only)."
    )
    p.add_argument("--date", default="", help="YYYY-MM-DD; used to auto-pick latest maker sheet for that date.")
    p.add_argument("--sheet", default="", help="Explicit maker sheet CSV path.")
    p.add_argument(
        "--live",
        action="store_true",
        help="Print LIVE commands (include --confirm). Default prints dry-run commands only.",
    )
    p.add_argument(
        "--env-list",
        default="config/env.list",
        help="Env list path for private creds (forwarded to kalshi_place_limit_order.py).",
    )
    p.add_argument(
        "--require-maker",
        action="store_true",
        help="Include --require-maker on each command (guards against crossing spread).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.sheet:
        sheet_path = Path(str(args.sheet))
    else:
        if not str(args.date).strip():
            raise SystemExit("[error] pass either --sheet or --date")
        date = dt.date.fromisoformat(str(args.date).strip())
        sheet_path = _latest_sheet_for_date(date)

    df = pd.read_csv(sheet_path)
    if df.empty:
        print(f"[ok] sheet is empty: {sheet_path}")
        return

    required = ["market_ticker_away", "contracts", "maker_limit_price_cents"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] sheet missing columns: {missing}")

    df["maker_limit_price_cents"] = pd.to_numeric(df["maker_limit_price_cents"], errors="coerce")
    df = df[df["maker_limit_price_cents"].notna()].copy()
    if df.empty:
        print(f"[warn] no rows have maker_limit_price_cents; nothing to print (sheet={sheet_path})")
        return

    print(f"# Source sheet: {sheet_path}")
    print("# NOTE: commands below may place REAL trades if you use --live / --confirm.")
    print("")

    base = "PYTHONPATH=. python chimera_v2c/tools/kalshi_place_limit_order.py"
    for row in df.to_dict(orient="records"):
        ticker = str(row.get("market_ticker_away") or "").strip()
        count = int(row.get("contracts") or 0)
        price = int(float(row.get("maker_limit_price_cents")))
        if not ticker or count <= 0:
            continue

        bits = [
            base,
            f"--ticker {ticker}",
            "--action buy",
            "--side yes",
            f"--count {count}",
            f"--price-cents {price}",
            f"--env-list {args.env_list}",
        ]
        if args.require_maker:
            bits.append("--require-maker")
        if args.live:
            bits.append("--confirm")
        print(" ".join(bits))


if __name__ == "__main__":
    main()

