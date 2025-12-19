#!/usr/bin/env python
"""
Safe daily wrapper for the Rule A (taker) workflow.

Design goals:
  - Default to writing ONLY under `reports/execution_logs/` (plans, fills, grades).
  - Never overwrite daily ledgers; only fill blank outcomes when explicitly enabled.
  - Keep execution manual; this script does not place any orders.

Usage (from repo root):
  # Plan for a slate (writes plan CSVs)
  PYTHONPATH=. python chimera_v2c/tools/run_rule_a_daily.py plan --date YYYY-MM-DD --leagues nba,nhl

  # Reconcile fills + grade latest plans (Pacific date semantics via export_kalshi_fills.py)
  PYTHONPATH=. python chimera_v2c/tools/run_rule_a_daily.py reconcile --date YYYY-MM-DD --leagues nba,nhl

  # After games are final, optionally fill blank outcomes in the daily ledger, then re-grade
  PYTHONPATH=. python chimera_v2c/tools/run_rule_a_daily.py reconcile --date YYYY-MM-DD --leagues nba,nhl --apply-outcomes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date as date_type
from pathlib import Path
from typing import List, Optional


def _default_vote_calibration_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_vote_calibration_{league}.json"


def _default_weak_buckets_path(league: str) -> Path:
    return Path("chimera_v2c/data") / f"rule_a_weak_buckets_{league}.json"


def _iso_date(text: str) -> str:
    try:
        return date_type.fromisoformat(str(text).strip()).isoformat()
    except ValueError as exc:
        raise SystemExit(f"[error] invalid --date (expected YYYY-MM-DD): {text}") from exc


def _leagues_from_arg(arg: str) -> List[str]:
    out = [p.strip().lower() for p in str(arg).split(",") if p.strip()]
    for lg in out:
        if lg not in {"nba", "nhl", "nfl"}:
            raise SystemExit("[error] --leagues must be a comma list of nba,nhl,nfl")
    return out


def _run_cmd(cmd: List[str]) -> None:
    print(f"[info] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(f"[error] command failed: {' '.join(cmd)}")


def _csv_has_data_rows(path: Path) -> bool:
    """
    True if the CSV has at least one non-header row.
    """
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as f:
            header = f.readline()
            if not header:
                return False
            data = f.readline()
            return bool(data and data.strip())
    except OSError:
        return False


def _ymd(date_iso: str) -> str:
    return date_iso.replace("-", "")


def _latest_plan_csv(*, date_iso: str, league: str) -> Optional[Path]:
    root = Path("reports/execution_logs/rule_a_votes") / _ymd(date_iso)
    if not root.exists():
        return None
    # Filenames are UTC-stamped like ..._<YYYYMMDD_HHMMSSZ>.csv; lexicographic max is latest.
    candidates = [
        p
        for p in sorted(root.glob(f"rule_a_votes_plan_{league}_*.csv"))
        if p.suffix.lower() == ".csv" and "_results" not in p.stem
    ]
    return candidates[-1] if candidates else None


def _ensure_daily_ledger(date_iso: str) -> None:
    _run_cmd([sys.executable, "chimera_v2c/tools/ensure_daily_ledger.py", "--date", date_iso])


def _maybe_snapshot_external(*, league: str, date_iso: str, kalshi_public_base: str, write_snapshot: bool) -> None:
    if not write_snapshot:
        return
    _run_cmd(
        [
            sys.executable,
            "chimera_v2c/tools/external_snapshot.py",
            "--league",
            league,
            "--date",
            date_iso,
            "--kalshi-public-base",
            kalshi_public_base,
            "--write-snapshot",
        ]
    )


def _log_rule_a_plan(
    *,
    league: str,
    date_iso: str,
    size_mode: str,
    slippage_cents: int,
    append_v2c_plan_log: bool,
    kalshi_public_base: str,
    min_minutes_to_start: Optional[float],
    max_minutes_to_start: Optional[float],
    calibration_json: str,
    weak_buckets_json: str,
) -> None:
    cmd = [
        sys.executable,
        "chimera_v2c/tools/log_rule_a_votes_plan.py",
        "--league",
        league,
        "--date",
        date_iso,
        "--allow-empty",
        "--size-mode",
        size_mode,
        "--slippage-cents",
        str(int(slippage_cents)),
        "--kalshi-public-base",
        kalshi_public_base,
    ]
    if min_minutes_to_start is not None:
        cmd.extend(["--min-minutes-to-start", str(float(min_minutes_to_start))])
    if max_minutes_to_start is not None:
        cmd.extend(["--max-minutes-to-start", str(float(max_minutes_to_start))])
    if calibration_json:
        cmd.extend(["--calibration-json", calibration_json])
    if weak_buckets_json:
        cmd.extend(["--weak-buckets-json", weak_buckets_json])
    if append_v2c_plan_log:
        cmd.append("--append-v2c-plan-log")
    _run_cmd(cmd)


def _export_fills(*, date_iso: str, series_prefix: str) -> Path:
    out = Path("reports/execution_logs/kalshi_fills") / _ymd(date_iso) / f"kalshi_fills_{_ymd(date_iso)}_pst.csv"
    _run_cmd(
        [
            sys.executable,
            "chimera_v2c/tools/export_kalshi_fills.py",
            "--date",
            date_iso,
            "--series-prefix",
            series_prefix,
            "--out",
            str(out),
        ]
    )
    return out


def _append_exec_log(*, fills_csv: Path, date_iso: str, series_prefix: str) -> None:
    _run_cmd(
        [
            sys.executable,
            "chimera_v2c/tools/append_rule_a_execution_log_from_fills.py",
            "--fills-csv",
            str(fills_csv),
            "--date",
            date_iso,
            "--series-prefix",
            series_prefix,
        ]
    )


def _maybe_apply_outcomes(date_iso: str, apply_outcomes: bool) -> None:
    if not apply_outcomes:
        return
    _run_cmd([sys.executable, "chimera_v2c/tools/fill_missing_daily_outcomes.py", "--date", date_iso])


def _grade_plan(plan_csv: Path) -> None:
    if not _csv_has_data_rows(plan_csv):
        print(f"[warn] empty plan (no qualifying games), skipping grade: {plan_csv}")
        return
    _run_cmd([sys.executable, "chimera_v2c/tools/grade_rule_a_votes_plan.py", "--plan-csv", str(plan_csv)])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Safe Rule A daily wrapper (planning + reconciliation; no execution).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Create Rule A plan CSVs under reports/execution_logs/ (no ledger writes).")
    plan.add_argument("--date", required=True, help="Slate date YYYY-MM-DD (operator local / Pacific semantics).")
    plan.add_argument("--leagues", default="nba,nhl", help="Comma list of leagues (default: nba,nhl).")
    plan.add_argument("--size-mode", default="blind_plus_weighted_flip2", help="Sizing mode for log_rule_a_votes_plan.py.")
    plan.add_argument("--slippage-cents", type=int, default=1, help="Assumed slippage cents added to away ask (default: 1).")
    plan.add_argument("--append-v2c-plan-log", action="store_true", help="Also append planned Rule A trades to v2c_plan_log.json.")
    plan.add_argument("--write-external-snapshot", action="store_true", help="Write external snapshot CSVs (no ledger apply).")
    plan.add_argument(
        "--write-research-queue",
        action="store_true",
        help="Also write a pre-filtered research queue + prompt pack for manual LLM runs (Kalshi home-favored only).",
    )
    plan.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    plan.add_argument("--min-minutes-to-start", type=float, default=None, help="Min minutes_to_start filter.")
    plan.add_argument("--max-minutes-to-start", type=float, default=None, help="Max minutes_to_start filter.")
    plan.add_argument(
        "--calibration-json",
        default="",
        help="Optional calibration JSON passed to log_rule_a_votes_plan.py (per league only when supplied).",
    )
    plan.add_argument(
        "--weak-buckets-json",
        default="",
        help="Optional weak buckets JSON passed to log_rule_a_votes_plan.py (per league only when supplied).",
    )
    plan.add_argument(
        "--no-auto-config",
        action="store_true",
        help="Disable auto-loading per-league calibration/weak-buckets JSONs from chimera_v2c/data/ when args are omitted.",
    )

    rec = sub.add_parser("reconcile", help="Export fills, append execution log, and grade latest plans.")
    rec.add_argument("--date", required=True, help="Slate date YYYY-MM-DD (operator local / Pacific semantics).")
    rec.add_argument("--leagues", default="nba,nhl", help="Comma list of leagues (default: nba,nhl).")
    rec.add_argument(
        "--apply-outcomes",
        action="store_true",
        help="Fill blank actual_outcome cells in the daily ledger for --date (append-only).",
    )
    rec.add_argument(
        "--skip-fills-export",
        action="store_true",
        help="Skip exporting fills (assumes rule_a_execution_log.csv already up to date).",
    )
    rec.add_argument(
        "--plan-csv",
        default="",
        help="Optional explicit plan CSV to grade (if omitted, grades latest per league).",
    )
    rec.add_argument(
        "--kalshi-series-prefix",
        default="",
        help="Optional series ticker prefix to filter fills export (e.g. KXNHLGAME). If omitted, exports all fills for the date.",
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cmd = str(args.cmd)

    if cmd == "plan":
        date_iso = _iso_date(args.date)
        leagues = _leagues_from_arg(args.leagues)
        _ensure_daily_ledger(date_iso)
        for lg in leagues:
            auto = not bool(args.no_auto_config)
            calib = str(args.calibration_json).strip()
            weak = str(args.weak_buckets_json).strip()
            if auto and not calib:
                p = _default_vote_calibration_path(lg)
                if p.exists():
                    calib = str(p)
            if auto and not weak:
                p = _default_weak_buckets_path(lg)
                if p.exists():
                    weak = str(p)

            _maybe_snapshot_external(
                league=lg,
                date_iso=date_iso,
                kalshi_public_base=str(args.kalshi_public_base),
                write_snapshot=bool(args.write_external_snapshot),
            )
            if bool(args.write_research_queue):
                _run_cmd(
                    [
                        sys.executable,
                        "chimera_v2c/tools/write_rule_a_research_queue.py",
                        "--league",
                        lg,
                        "--date",
                        date_iso,
                        "--slippage-cents",
                        str(int(args.slippage_cents)),
                        "--kalshi-public-base",
                        str(args.kalshi_public_base),
                        "--write-prompts",
                    ]
                    + (["--min-minutes-to-start", str(float(args.min_minutes_to_start))] if args.min_minutes_to_start is not None else [])
                    + (["--max-minutes-to-start", str(float(args.max_minutes_to_start))] if args.max_minutes_to_start is not None else [])
                )
            _log_rule_a_plan(
                league=lg,
                date_iso=date_iso,
                size_mode=str(args.size_mode),
                slippage_cents=int(args.slippage_cents),
                append_v2c_plan_log=bool(args.append_v2c_plan_log),
                kalshi_public_base=str(args.kalshi_public_base),
                min_minutes_to_start=args.min_minutes_to_start,
                max_minutes_to_start=args.max_minutes_to_start,
                calibration_json=calib,
                weak_buckets_json=weak,
            )
        return

    if cmd == "reconcile":
        date_iso = _iso_date(args.date)
        leagues = _leagues_from_arg(args.leagues)
        _ensure_daily_ledger(date_iso)
        _maybe_apply_outcomes(date_iso, bool(args.apply_outcomes))

        fills_csv: Optional[Path] = None
        if not bool(args.skip_fills_export):
            series_prefix = str(args.kalshi_series_prefix).strip().upper()
            fills_csv = _export_fills(date_iso=date_iso, series_prefix=series_prefix)
            _append_exec_log(fills_csv=fills_csv, date_iso=date_iso, series_prefix=series_prefix)

        explicit_plan = str(args.plan_csv).strip()
        if explicit_plan:
            plan_path = Path(explicit_plan)
            if not plan_path.exists():
                raise SystemExit(f"[error] missing --plan-csv: {plan_path}")
            _grade_plan(plan_path)
            return

        for lg in leagues:
            plan_path = _latest_plan_csv(date_iso=date_iso, league=lg)
            if plan_path is None:
                print(f"[warn] no plan found for {lg} on {date_iso} under reports/execution_logs/rule_a_votes/{_ymd(date_iso)}/")
                continue
            _grade_plan(plan_path)
        return

    raise SystemExit(f"[error] unknown cmd: {cmd}")


if __name__ == "__main__":
    main()
