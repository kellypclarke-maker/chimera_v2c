#!/usr/bin/env python
"""
Validation suite for Rule A using Kalshi bid/ask snapshots at fixed T-minus anchors (read-only).

Implements checks/analyses after a hand-audit passes:
  2) Out-of-sample (OOS) split summary + daily curve
  3) Anchor sensitivity (T-60/T-30/T-15/T-5, etc.)
  4) Mid bucket breakdown
  5) Fill-rate stress test (expected $ PnL/$ risked scaling)

This tool does NOT modify any ledgers.

Inputs:
  - Daily ledgers under reports/daily_ledgers/ (for model probs + outcomes)
  - One or more bid/ask snapshot CSVs produced by export_kalshi_bidask_tminus.py

Usage:
  # Ensure snapshots exist (optional):
  PYTHONPATH=. python chimera_v2c/tools/validate_rule_a_bidask_suite.py \
    --start-date 2025-11-19 --end-date 2025-12-15 \
    --anchors 60 30 15 5 --export-missing

Outputs (under reports/thesis_summaries/):
  - ruleA_bidask_suite_oos_split_YYYYMMDD_YYYYMMDD.csv
  - ruleA_bidask_suite_daily_curve_YYYYMMDD_YYYYMMDD.csv
  - ruleA_bidask_suite_anchor_sensitivity_YYYYMMDD_YYYYMMDD.csv
  - ruleA_bidask_suite_mid_buckets_YYYYMMDD_YYYYMMDD.csv
  - ruleA_bidask_suite_fill_rate_stress_YYYYMMDD_YYYYMMDD.csv
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from chimera_v2c.lib import team_mapper
from chimera_v2c.src.kalshi_fees import taker_fee_dollars
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games


MODELS: List[str] = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]


@dataclass(frozen=True)
class TeamSnap:
    market_ticker: str
    yes_team: str
    bid: float
    ask: float
    mid: float
    spread: float
    target_iso: str


def _clamp_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _parse_matchup(*, league: str, matchup: str) -> Optional[Tuple[str, str]]:
    if "@" not in matchup:
        return None
    away_raw, home_raw = matchup.split("@", 1)
    away = team_mapper.normalize_team_code(away_raw.strip(), league)
    home = team_mapper.normalize_team_code(home_raw.strip(), league)
    if not away or not home:
        return None
    return away, home


def _load_snapshot(path: Path) -> Dict[Tuple[str, str, str, str], TeamSnap]:
    df = pd.read_csv(path)
    df = df[df.get("ok", 0) == 1].copy()
    out: Dict[Tuple[str, str, str, str], TeamSnap] = {}
    for _, r in df.iterrows():
        date = str(r.get("date", "")).strip()
        league = str(r.get("league", "")).strip().lower()
        matchup = str(r.get("matchup", "")).strip().upper()
        yes_team = str(r.get("yes_team", "")).strip().upper()
        if not date or not league or not matchup or not yes_team:
            continue
        try:
            bid = float(r["yes_bid_cents"]) / 100.0
            ask = float(r["yes_ask_cents"]) / 100.0
            mid = float(r["mid"])
            spread = float(r["spread"])
        except Exception:
            continue
        out[(date, league, matchup, yes_team)] = TeamSnap(
            market_ticker=str(r.get("market_ticker", "")).strip(),
            yes_team=yes_team,
            bid=bid,
            ask=ask,
            mid=mid,
            spread=spread,
            target_iso=str(r.get("target_iso", "")).strip(),
        )
    return out


def _away_pnl_per_contract(*, price_away: float, home_win: float) -> float:
    if home_win == 0.0:
        return 1.0 - float(price_away)
    return -float(price_away)


def _iter_dates(start: str, end: str) -> List[str]:
    ds = sorted({d.strftime("%Y-%m-%d") for d in pd.date_range(start=start, end=end, freq="D").to_pydatetime()})
    return ds


def _mid_bucket(mid_home: float) -> str:
    m = float(mid_home)
    edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]
    labels = ["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.80", "0.80-1.00"]
    for i in range(len(labels)):
        if edges[i] <= m < edges[i + 1]:
            return labels[i]
    return "other"


def _ensure_snapshot(
    *,
    start_date: str,
    end_date: str,
    minutes: int,
    out_path: Path,
    export_missing: bool,
) -> None:
    if out_path.exists() or not export_missing:
        return
    cmd = [
        "python",
        "chimera_v2c/tools/export_kalshi_bidask_tminus.py",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--minutes-before-start",
        str(int(minutes)),
        "--out",
        str(out_path),
    ]
    env = dict(**{k: v for k, v in dict(**__import__("os").environ).items()})
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def build_game_table(
    games: Iterable[GameRow],
    *,
    snapshot: Dict[Tuple[str, str, str, str], TeamSnap],
    slippage_cents: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    slip = float(slippage_cents) / 100.0
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        parsed = _parse_matchup(league=g.league, matchup=g.matchup)
        if parsed is None:
            continue
        away, home = parsed
        date = g.date.strftime("%Y-%m-%d")
        matchup = f"{away}@{home}"
        home_snap = snapshot.get((date, g.league, matchup, home))
        away_snap = snapshot.get((date, g.league, matchup, away))
        if home_snap is None or away_snap is None:
            continue
        mid_home = float(home_snap.mid)
        if mid_home <= 0.5:
            continue
        price_away = _clamp_price(float(away_snap.ask) + slip)
        pnl_unit = _away_pnl_per_contract(price_away=price_away, home_win=float(g.home_win))

        votes_by_model: Dict[str, int] = {}
        votes = 0
        for m in MODELS:
            p = g.probs.get(m)
            if p is None:
                votes_by_model[m] = 0
                continue
            trig = 1 if float(p) < mid_home else 0
            votes_by_model[m] = trig
            votes += trig

        units = 1 + votes
        gross = units * pnl_unit
        fees = taker_fee_dollars(contracts=int(units), price=float(price_away))
        net = gross - fees
        risked = units * float(price_away)
        roi = 0.0 if risked <= 0 else net / risked

        rows.append(
            {
                "date": date,
                "league": g.league,
                "matchup": matchup,
                "mid_home": round(mid_home, 6),
                "away_ask": round(float(away_snap.ask), 6),
                "exec_price_away": round(float(price_away), 6),
                "slippage_cents": int(slippage_cents),
                "units": int(units),
                "votes": int(votes),
                "pnl_per_unit": round(float(pnl_unit), 6),
                "gross_pnl": round(float(gross), 6),
                "fees": round(float(fees), 6),
                "net_pnl": round(float(net), 6),
                "risked": round(float(risked), 6),
                "roi_net": round(float(roi), 6),
                "target_iso": home_snap.target_iso,
                **{f"vote_{k}": int(v) for k, v in votes_by_model.items()},
            }
        )
    return pd.DataFrame(rows)


def _agg(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    g = df.groupby(list(keys), dropna=False, as_index=False).agg(
        bets=("matchup", "count"),
        units=("units", "sum"),
        risked=("risked", "sum"),
        fees=("fees", "sum"),
        net_pnl=("net_pnl", "sum"),
    )
    g["roi_net"] = g.apply(lambda r: 0.0 if r["risked"] <= 0 else float(r["net_pnl"] / r["risked"]), axis=1)
    return g


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Rule A bid/ask validation suite (read-only).")
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD inclusive.")
    ap.add_argument("--end-date", required=True, help="YYYY-MM-DD inclusive.")
    ap.add_argument("--anchors", nargs="+", type=int, default=[60, 30, 15, 5], help="T-minus anchors in minutes.")
    ap.add_argument("--slippage-cents", nargs="+", type=int, default=[0, 1, 2], help="Extra slippage to add on top of away ask.")
    ap.add_argument("--export-missing", action="store_true", help="Export missing snapshot CSVs for requested anchors.")
    ap.add_argument("--split-date", help="YYYY-MM-DD split date for OOS; defaults to midpoint by calendar days.")
    ap.add_argument("--fill-rates", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0], help="Fill rates for stress test.")
    ap.add_argument("--out-dir", default="reports/thesis_summaries", help="Output directory (default: reports/thesis_summaries).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    start = args.start_date
    end = args.end_date
    days = _iter_dates(start, end)
    split_date = args.split_date or days[len(days) // 2]

    games = load_games(daily_dir=LEDGER_DIR, start_date=start, end_date=end, models=MODELS)
    if not games:
        raise SystemExit("[error] no games loaded from daily ledgers for the given window")

    anchor_paths: Dict[int, Path] = {}
    for a in args.anchors:
        p = Path(f"reports/market_snapshots/kalshi_bidask_tminus_{start.replace('-', '')}_{end.replace('-', '')}_m{int(a)}.csv")
        anchor_paths[int(a)] = p
        _ensure_snapshot(start_date=start, end_date=end, minutes=int(a), out_path=p, export_missing=bool(args.export_missing))
        if not p.exists():
            raise SystemExit(f"[error] missing snapshot CSV for anchor m{a}: {p} (pass --export-missing)")

    # Build per-game tables and aggregate outputs.
    suite_rows: List[pd.DataFrame] = []
    for a, snap_path in sorted(anchor_paths.items(), key=lambda kv: kv[0], reverse=True):
        snap = _load_snapshot(snap_path)
        for s in args.slippage_cents:
            tbl = build_game_table(games, snapshot=snap, slippage_cents=int(s))
            if tbl.empty:
                continue
            tbl.insert(0, "anchor_minutes", int(a))
            tbl.insert(1, "window_start", start)
            tbl.insert(2, "window_end", end)
            suite_rows.append(tbl)

    if not suite_rows:
        raise SystemExit("[error] no qualifying games found across requested anchors/slippage")

    all_games = pd.concat(suite_rows, ignore_index=True)

    # 2) OOS split: (anchor=30 by default when present; else first anchor)
    primary_anchor = 30 if 30 in anchor_paths else sorted(anchor_paths.keys())[0]
    primary = all_games[(all_games["anchor_minutes"] == primary_anchor) & (all_games["slippage_cents"] == 1)].copy()
    if primary.empty:
        primary = all_games[(all_games["anchor_minutes"] == primary_anchor)].copy()

    oos = primary.copy()
    oos["segment"] = oos["date"].apply(lambda d: "oos" if str(d) >= split_date else "in_sample")
    oos_sum = _agg(oos, ["window_start", "window_end", "anchor_minutes", "slippage_cents", "segment"])
    oos_sum.insert(0, "split_date", split_date)
    _write_csv(out_dir / f"ruleA_bidask_suite_oos_split_{start.replace('-', '')}_{end.replace('-', '')}.csv", oos_sum)

    daily = oos.groupby(["date"], as_index=False).agg(net_pnl=("net_pnl", "sum"), risked=("risked", "sum"))
    daily = daily.sort_values("date")
    daily["cum_net_pnl"] = daily["net_pnl"].cumsum()
    daily["cum_risked"] = daily["risked"].cumsum()
    daily["cum_roi_net"] = daily.apply(lambda r: 0.0 if r["cum_risked"] <= 0 else float(r["cum_net_pnl"] / r["cum_risked"]), axis=1)
    daily.insert(0, "window_start", start)
    daily.insert(1, "window_end", end)
    daily.insert(2, "split_date", split_date)
    daily.insert(3, "anchor_minutes", primary_anchor)
    daily.insert(4, "slippage_cents", int(primary["slippage_cents"].iloc[0]))
    _write_csv(out_dir / f"ruleA_bidask_suite_daily_curve_{start.replace('-', '')}_{end.replace('-', '')}.csv", daily)

    # 3) Anchor sensitivity: aggregate overall + by league
    sens_src = pd.concat([all_games, all_games.assign(league="overall")], ignore_index=True)
    sens = _agg(sens_src, ["window_start", "window_end", "anchor_minutes", "slippage_cents", "league"])
    _write_csv(out_dir / f"ruleA_bidask_suite_anchor_sensitivity_{start.replace('-', '')}_{end.replace('-', '')}.csv", sens)

    # 4) Mid buckets: overall + by league
    buckets = pd.concat([all_games, all_games.assign(league="overall")], ignore_index=True)
    buckets["mid_bucket"] = buckets["mid_home"].apply(_mid_bucket)
    bsum = _agg(buckets, ["window_start", "window_end", "anchor_minutes", "slippage_cents", "league", "mid_bucket"])
    _write_csv(out_dir / f"ruleA_bidask_suite_mid_buckets_{start.replace('-', '')}_{end.replace('-', '')}.csv", bsum)

    # 5) Fill-rate stress: expected scaling (no selective fills modeled)
    fr_rows: List[Dict[str, object]] = []
    base = _agg(sens_src, ["window_start", "window_end", "anchor_minutes", "slippage_cents", "league"])
    for _, r in base.iterrows():
        for fr in args.fill_rates:
            fr = float(fr)
            if fr < 0 or fr > 1:
                continue
            fr_rows.append(
                {
                    "window_start": r["window_start"],
                    "window_end": r["window_end"],
                    "anchor_minutes": int(r["anchor_minutes"]),
                    "slippage_cents": int(r["slippage_cents"]),
                    "league": r["league"],
                    "fill_rate": fr,
                    "expected_bets": float(r["bets"]) * fr,
                    "expected_units": float(r["units"]) * fr,
                    "expected_risked": float(r["risked"]) * fr,
                    "expected_fees": float(r["fees"]) * fr,
                    "expected_net_pnl": float(r["net_pnl"]) * fr,
                    "roi_net": float(r["roi_net"]),
                }
            )
    fr_df = pd.DataFrame(fr_rows)
    _write_csv(out_dir / f"ruleA_bidask_suite_fill_rate_stress_{start.replace('-', '')}_{end.replace('-', '')}.csv", fr_df)

    print(f"[ok] wrote suite outputs under {out_dir}")


if __name__ == "__main__":
    main()
