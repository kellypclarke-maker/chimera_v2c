#!/usr/bin/env python
"""
Build a compact evaluation scoreboard from daily ledgers (read-only on ledgers).

Outputs (default under reports/thesis_summaries/):
  - scoreboard_summary_<league>_<start>_<end>.csv
  - scoreboard_daily_<league>_<start>_<end>.csv
  - scoreboard_reliability_<league>_<start>_<end>.csv
  - scoreboard_<league>_<start>_<end>.md

This tool is intended to answer:
  - "Are we improving?" (daily Brier/EV trends)
  - "Which models/buckets are calibrated?" (reliability)
  - "Which models have positive realized EV vs Kalshi mid?" (EV vs mid)

Usage:
  PYTHONPATH=. python chimera_v2c/tools/build_scoreboard.py --league nhl --days 14
  PYTHONPATH=. python chimera_v2c/tools/build_scoreboard.py --league nhl --start-date YYYY-MM-DD --end-date YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.ledger_analysis import BrierStats, EvStats, GameRow, compute_brier, compute_ev_vs_kalshi, load_games
from chimera_v2c.src.scoreboard import BucketStats, compute_accuracy, compute_reliability_by_bucket, sanitize_games


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt", "market_proxy", "moneypuck"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build evaluation scoreboard artifacts from daily ledgers.")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl). Default: all leagues.")
    ap.add_argument(
        "--days",
        type=int,
        default=30,
        help="Include only the most recent N ledger days by filename date (default: 30). "
        "Use 0 or a negative value to include all days (ignored when start/end date is set).",
    )
    ap.add_argument("--start-date", help="Optional start date (YYYY-MM-DD). Overrides --days.")
    ap.add_argument("--end-date", help="Optional end date (YYYY-MM-DD). Overrides --days.")
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Probability columns to include (default: v2c gemini grok gpt market_proxy moneypuck).",
    )
    ap.add_argument("--bucket-width", type=float, default=0.1, help="Bucket width for reliability stats (default: 0.1).")
    ap.add_argument("--min-n", type=int, default=10, help="Min games per (model,bucket) row (default: 10).")
    ap.add_argument("--out-dir", default="reports/thesis_summaries", help="Output directory (default: reports/thesis_summaries).")
    return ap.parse_args()


def _window_tokens(games: Sequence[GameRow]) -> Tuple[str, str]:
    dates = sorted({g.date.strftime("%Y%m%d") for g in games})
    if not dates:
        return "unknown", "unknown"
    return dates[0], dates[-1]


def _fmt(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return ""
    return f"{x:.{digits}f}"


def _count_outcomes(games: Iterable[GameRow]) -> Dict[str, int]:
    out = {"resolved": 0, "push": 0, "unresolved": 0}
    for g in games:
        y = g.home_win
        if y is None:
            out["unresolved"] += 1
        elif y == 0.5:
            out["push"] += 1
        else:
            out["resolved"] += 1
    return out


def _coverage(games: Iterable[GameRow], models: Sequence[str]) -> Dict[str, int]:
    cov = {m: 0 for m in models}
    for g in games:
        for m in models:
            if m in g.probs:
                cov[m] += 1
    return cov


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _write_md(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _summary_rows_for_subset(
    *,
    league: str,
    games: List[GameRow],
    models: List[str],
    summary_models: List[str],
    window_start: str,
    window_end: str,
) -> List[Dict[str, object]]:
    cov = _coverage(games, summary_models)
    outcome_counts = _count_outcomes(games)
    brier = compute_brier(games, models=summary_models)
    acc = compute_accuracy(games, models=summary_models)
    ev = compute_ev_vs_kalshi(games, models=models)

    rows: List[Dict[str, object]] = []
    for m in summary_models:
        br: Optional[BrierStats] = brier.get(m)
        ac = acc.get(m)
        evs: Optional[EvStats] = ev.get(m) if m in ev else None
        rows.append(
            {
                "league": league,
                "window_start": window_start,
                "window_end": window_end,
                "games_total": len(games),
                "games_resolved": outcome_counts["resolved"],
                "games_push": outcome_counts["push"],
                "games_unresolved": outcome_counts["unresolved"],
                "model": m,
                "coverage_n": cov.get(m, 0),
                "brier": _fmt(br.mean_brier if br else None, 6),
                "brier_n": "" if br is None else br.n,
                "acc": _fmt(ac.acc if ac else None, 6),
                "acc_n": "" if ac is None else ac.n,
                "ev_bets": "" if evs is None else evs.bets,
                "ev_total_pnl": "" if evs is None else _fmt(evs.total_pnl, 6),
                "ev_avg_pnl": "" if evs is None else _fmt(evs.avg_pnl, 6),
            }
        )
    return rows


def _daily_rows_for_subset(
    *,
    league: str,
    games: List[GameRow],
    models: List[str],
    summary_models: List[str],
) -> List[Dict[str, object]]:
    # Group by ledger file date (already stored in GameRow.date).
    by_date: Dict[str, List[GameRow]] = defaultdict(list)
    for g in games:
        by_date[g.date.strftime("%Y-%m-%d")].append(g)

    out: List[Dict[str, object]] = []
    for date_iso in sorted(by_date.keys()):
        subset = by_date[date_iso]
        outcome_counts = _count_outcomes(subset)
        brier = compute_brier(subset, models=summary_models)
        acc = compute_accuracy(subset, models=summary_models)
        ev = compute_ev_vs_kalshi(subset, models=models)
        cov = _coverage(subset, summary_models)

        for m in summary_models:
            br = brier.get(m)
            ac = acc.get(m)
            evs = ev.get(m) if m in ev else None
            out.append(
                {
                    "date": date_iso,
                    "league": league,
                    "games_total": len(subset),
                    "games_resolved": outcome_counts["resolved"],
                    "games_push": outcome_counts["push"],
                    "games_unresolved": outcome_counts["unresolved"],
                    "model": m,
                    "coverage_n": cov.get(m, 0),
                    "brier": _fmt(br.mean_brier if br else None, 6),
                    "brier_n": "" if br is None else br.n,
                    "acc": _fmt(ac.acc if ac else None, 6),
                    "acc_n": "" if ac is None else ac.n,
                    "ev_bets": "" if evs is None else evs.bets,
                    "ev_total_pnl": "" if evs is None else _fmt(evs.total_pnl, 6),
                    "ev_avg_pnl": "" if evs is None else _fmt(evs.avg_pnl, 6),
                }
            )
    return out


def _reliability_rows_for_subset(
    *,
    league: str,
    games: List[GameRow],
    summary_models: List[str],
    bucket_width: float,
    min_n: int,
) -> List[Dict[str, object]]:
    stats = compute_reliability_by_bucket(games, models=summary_models, bucket_width=bucket_width)
    out: List[Dict[str, object]] = []
    for m in summary_models:
        for bucket in sorted(stats.get(m, {}).keys()):
            s: BucketStats = stats[m][bucket]
            if s.n < min_n:
                continue
            out.append(
                {
                    "league": league,
                    "model": m,
                    "bucket": bucket,
                    "n": s.n,
                    "avg_p": _fmt(s.avg_p, 6),
                    "actual_rate": _fmt(s.actual_rate, 6),
                    "brier": _fmt(s.brier, 6),
                    "bets": s.bets,
                    "total_pnl": _fmt(s.pnl_sum, 6),
                    "avg_pnl": _fmt(s.avg_pnl, 6),
                }
            )
    return out


def main() -> None:
    args = _parse_args()
    league_filter = args.league.lower() if args.league else None

    if args.start_date or args.end_date:
        days = None
    else:
        days = args.days
        if days <= 0:
            days = None

    # Always include kalshi_mid as a baseline model in the scoreboard outputs.
    models = list(dict.fromkeys([m.strip() for m in args.models if m.strip() and m.strip() != "kalshi_mid"]))
    summary_models = models + ["kalshi_mid"]

    games = load_games(
        days=days,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=summary_models,
    )
    if not games:
        raise SystemExit("[error] no games loaded from daily ledgers for the requested filters/window")

    games = sanitize_games(games, models=summary_models)

    window_start, window_end = _window_tokens(games)
    league_token = league_filter or "all"

    out_dir = Path(args.out_dir)
    summary_path = out_dir / f"scoreboard_summary_{league_token}_{window_start}_{window_end}.csv"
    daily_path = out_dir / f"scoreboard_daily_{league_token}_{window_start}_{window_end}.csv"
    reliability_path = out_dir / f"scoreboard_reliability_{league_token}_{window_start}_{window_end}.csv"
    md_path = out_dir / f"scoreboard_{league_token}_{window_start}_{window_end}.md"

    leagues = sorted({g.league for g in games})
    if league_filter is not None:
        leagues = [league_filter]

    summary_rows: List[Dict[str, object]] = []
    daily_rows: List[Dict[str, object]] = []
    reliability_rows: List[Dict[str, object]] = []

    for lg in leagues:
        subset = [g for g in games if g.league == lg]
        if not subset:
            continue
        summary_rows.extend(
            _summary_rows_for_subset(
                league=lg,
                games=subset,
                models=models,
                summary_models=summary_models,
                window_start=window_start,
                window_end=window_end,
            )
        )
        daily_rows.extend(_daily_rows_for_subset(league=lg, games=subset, models=models, summary_models=summary_models))
        reliability_rows.extend(
            _reliability_rows_for_subset(
                league=lg,
                games=subset,
                summary_models=summary_models,
                bucket_width=float(args.bucket_width),
                min_n=int(args.min_n),
            )
        )

    # Overall rollup when no league filter was requested.
    if league_filter is None and len(leagues) > 1:
        summary_rows.extend(
            _summary_rows_for_subset(
                league="overall",
                games=games,
                models=models,
                summary_models=summary_models,
                window_start=window_start,
                window_end=window_end,
            )
        )
        daily_rows.extend(_daily_rows_for_subset(league="overall", games=games, models=models, summary_models=summary_models))
        reliability_rows.extend(
            _reliability_rows_for_subset(
                league="overall",
                games=games,
                summary_models=summary_models,
                bucket_width=float(args.bucket_width),
                min_n=int(args.min_n),
            )
        )

    summary_fields = [
        "league",
        "window_start",
        "window_end",
        "games_total",
        "games_resolved",
        "games_push",
        "games_unresolved",
        "model",
        "coverage_n",
        "brier",
        "brier_n",
        "acc",
        "acc_n",
        "ev_bets",
        "ev_total_pnl",
        "ev_avg_pnl",
    ]
    _write_csv(summary_path, summary_fields, summary_rows)

    daily_fields = [
        "date",
        "league",
        "games_total",
        "games_resolved",
        "games_push",
        "games_unresolved",
        "model",
        "coverage_n",
        "brier",
        "brier_n",
        "acc",
        "acc_n",
        "ev_bets",
        "ev_total_pnl",
        "ev_avg_pnl",
    ]
    _write_csv(daily_path, daily_fields, daily_rows)

    reliability_fields = [
        "league",
        "model",
        "bucket",
        "n",
        "avg_p",
        "actual_rate",
        "brier",
        "bets",
        "total_pnl",
        "avg_pnl",
    ]
    _write_csv(reliability_path, reliability_fields, reliability_rows)

    md_lines: List[str] = []
    md_lines.append(f"# Scoreboard ({league_token})")
    md_lines.append("")
    md_lines.append(f"- Window: `{window_start}` → `{window_end}` (by ledger filename date)")
    md_lines.append(f"- Models: `{', '.join(summary_models)}`")
    md_lines.append(f"- Summary CSV: `{summary_path}`")
    md_lines.append(f"- Daily CSV: `{daily_path}`")
    md_lines.append(f"- Reliability CSV: `{reliability_path}`")
    md_lines.append("")
    md_lines.append("## Summary (CSV Preview)")
    md_lines.append("")
    md_lines.append("| league | model | brier | brier_n | acc | ev_total_pnl | ev_bets |")
    md_lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in summary_rows:
        md_lines.append(
            f"| {r['league']} | {r['model']} | {r['brier'] or ''} | {r['brier_n'] or ''} | {r['acc'] or ''} | {r['ev_total_pnl'] or ''} | {r['ev_bets'] or ''} |"
        )
    _write_md(md_path, md_lines)

    print(f"[info] wrote:\n  - {summary_path}\n  - {daily_path}\n  - {reliability_path}\n  - {md_path}")


if __name__ == "__main__":
    main()
