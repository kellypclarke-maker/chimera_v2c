from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MakerPriceDecision:
    limit_price_cents: int | None
    reason: str


def _maker_limit_price_cents(
    bid_cents: int | float | None,
    ask_cents: int | float | None,
    improve_cents: int,
) -> MakerPriceDecision:
    if bid_cents is None or (isinstance(bid_cents, float) and pd.isna(bid_cents)):
        bid = 0
    else:
        bid = int(bid_cents)

    if ask_cents is None or (isinstance(ask_cents, float) and pd.isna(ask_cents)):
        return MakerPriceDecision(limit_price_cents=None, reason="missing_ask")

    ask = int(ask_cents)
    if ask <= 0:
        return MakerPriceDecision(limit_price_cents=None, reason="invalid_ask")

    max_maker = ask - 1
    if max_maker <= 0:
        return MakerPriceDecision(limit_price_cents=None, reason="ask_too_low")

    proposed = bid + max(0, int(improve_cents))
    limit = min(max_maker, proposed)
    if limit <= 0:
        return MakerPriceDecision(limit_price_cents=None, reason="limit_too_low")

    if limit <= bid:
        return MakerPriceDecision(limit_price_cents=limit, reason="maker_at_bid")
    return MakerPriceDecision(limit_price_cents=limit, reason="maker_inside_spread")


def _read_latest_rule_a_plan_csvs(plan_dir: Path, leagues: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    for league in leagues:
        candidates = sorted(plan_dir.glob(f"rule_a_votes_plan_{league}_*.csv"))
        if not candidates:
            continue
        out.append(candidates[-1])
    return out


def build_maker_order_sheet(
    plan_paths: list[Path],
    maker_improve_cents: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for plan_path in plan_paths:
        plan = pd.read_csv(plan_path)
        if plan.empty:
            continue

        required = [
            "date",
            "league",
            "matchup",
            "market_ticker_away",
            "contracts_planned",
            "away_yes_bid_cents",
            "away_yes_ask_cents",
        ]
        missing = [c for c in required if c not in plan.columns]
        if missing:
            raise ValueError(f"plan CSV missing required columns: {missing} ({plan_path})")

        for record in plan.to_dict(orient="records"):
            contracts = int(record.get("contracts_planned") or 0)
            if contracts <= 0:
                continue

            start_time_utc = str(record.get("start_time_utc") or "").strip()
            start_time_pt = ""
            if start_time_utc:
                try:
                    s = start_time_utc[:-1] + "+00:00" if start_time_utc.endswith("Z") else start_time_utc
                    dtu = dt.datetime.fromisoformat(s)
                    if dtu.tzinfo is None:
                        dtu = dtu.replace(tzinfo=dt.timezone.utc)
                    start_time_pt = dtu.astimezone(ZoneInfo("America/Los_Angeles")).isoformat()
                except Exception:
                    start_time_pt = ""

            decision = _maker_limit_price_cents(
                record.get("away_yes_bid_cents"),
                record.get("away_yes_ask_cents"),
                improve_cents=maker_improve_cents,
            )

            rows.append(
                {
                    "date": record.get("date"),
                    "league": record.get("league"),
                    "matchup": record.get("matchup"),
                    "start_time_utc": record.get("start_time_utc"),
                    "start_time_pt": start_time_pt,
                    "minutes_to_start": record.get("minutes_to_start"),
                    "market_ticker_away": record.get("market_ticker_away"),
                    "action": "buy",
                    "side": "yes",
                    "contracts": contracts,
                    "maker_limit_price_cents": decision.limit_price_cents,
                    "maker_price_reason": decision.reason,
                    "away_yes_bid_cents": record.get("away_yes_bid_cents"),
                    "away_yes_ask_cents": record.get("away_yes_ask_cents"),
                    "price_away_planned": record.get("price_away_planned"),
                    "slippage_cents": record.get("slippage_cents"),
                    "mid_home": record.get("mid_home"),
                    "mid_bucket": record.get("mid_bucket"),
                    "weak_bucket": record.get("weak_bucket"),
                    "votes": record.get("votes"),
                    "weighted_votes": record.get("weighted_votes"),
                    "vote_models": record.get("vote_models"),
                    "edge_net_vote_sum": record.get("edge_net_vote_sum"),
                    "size_mode": record.get("size_mode"),
                    "source_plan_csv": str(plan_path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Keep "unpriceable" rows (so the operator sees them), but sort them to the bottom.
    df["maker_limit_price_cents"] = pd.to_numeric(df["maker_limit_price_cents"], errors="coerce")
    df = df.sort_values(
        by=["start_time_utc", "league", "maker_limit_price_cents"],
        ascending=[True, True, False],
        na_position="last",
    )
    return df.reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a maker-only order sheet from Rule A votes plan CSVs (no trading; output CSV only)."
    )
    parser.add_argument("--date", required=True, help="Ledger date (YYYY-MM-DD). Used to locate plan directory.")
    parser.add_argument(
        "--leagues",
        default="nba,nhl",
        help="Comma-separated leagues to include (default: nba,nhl).",
    )
    parser.add_argument(
        "--plan-dir",
        default=None,
        help="Optional explicit plan directory; default is reports/execution_logs/rule_a_votes/YYYYMMDD.",
    )
    parser.add_argument(
        "--plan-csv",
        action="append",
        default=[],
        help="Explicit plan CSV path (repeatable). If provided, overrides --plan-dir/--leagues discovery.",
    )
    parser.add_argument(
        "--maker-improve-cents",
        type=int,
        default=0,
        help="Improve bid by this many cents (clamped to ask-1). 0 means 'at bid'.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path; default writes under reports/trade_sheets/.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    date = dt.date.fromisoformat(args.date)
    yyyymmdd = date.strftime("%Y%m%d")

    leagues = [s.strip().lower() for s in str(args.leagues).split(",") if s.strip()]

    if args.plan_csv:
        plan_paths = [Path(p) for p in args.plan_csv]
    else:
        plan_dir = Path(args.plan_dir) if args.plan_dir else Path("reports/execution_logs/rule_a_votes") / yyyymmdd
        plan_paths = _read_latest_rule_a_plan_csvs(plan_dir=plan_dir, leagues=leagues)

    sheet = build_maker_order_sheet(plan_paths=plan_paths, maker_improve_cents=int(args.maker_improve_cents))

    out_path = Path(args.out) if args.out else None
    if out_path is None:
        out_dir = Path("reports/trade_sheets") / "rule_a_maker_orders" / yyyymmdd
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        out_path = out_dir / f"rule_a_maker_orders_{yyyymmdd}_{ts}.csv"

    sheet.to_csv(out_path, index=False)
    print(f"[ok] wrote {len(sheet)} rows -> {out_path}")


if __name__ == "__main__":
    main()
