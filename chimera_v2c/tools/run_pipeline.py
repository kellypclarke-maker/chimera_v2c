"""
Unified pipeline runner for multiple leagues (nba|nhl|nfl).

Chains optional refresh -> calibration -> plan -> log -> backtest for each league.
Defaults: dry-run mode; append-only logs; no ledger writes.

Usage examples (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/run_pipeline.py --date 2025-12-10 --leagues nba,nhl --refresh-factors --fit-calibration --backtest-days 7 --skip-preflight
  PYTHONPATH=. python chimera_v2c/tools/run_pipeline.py --date 2025-12-10 --leagues all --dry-run
  PYTHONPATH=. OPENAI_API_KEY=... python chimera_v2c/tools/run_pipeline.py --date 2025-12-13 --leagues nhl --llm-injuries
"""
from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path
from typing import List

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.tools.preflight_check import check_injury_freshness, check_start_windows, check_data_freshness
from chimera_v2c.tools.refresh_slate_updates import refresh_slate_updates


def run_cmd(cmd: List[str]) -> None:
    print(f"[info] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(f"[error] command failed: {' '.join(cmd)}")


def leagues_from_arg(arg: str) -> List[str]:
    if arg.lower() == "all":
        return ["nba", "nhl", "nfl"]
    return [p.strip().lower() for p in arg.split(",") if p.strip()]


def _iso_date(date_str: str) -> str:
    try:
        return datetime.date.fromisoformat(str(date_str).strip()).isoformat()
    except ValueError as exc:
        raise SystemExit(f"[error] invalid --date (expected YYYY-MM-DD): {date_str}") from exc


def _apply_llm_injuries(*, league: str, date_iso: str, model: str) -> None:
    digest_path = Path("chimera_v2c/data") / f"news_{date_iso}_{league.lower()}.txt"
    if not digest_path.exists():
        raise SystemExit(f"[error] missing digest for LLM injuries: {digest_path}")

    if league.lower() == "nhl":
        cmd = [
            sys.executable,
            "chimera_v2c/tools/apply_llm_nhl_injuries.py",
            "--date",
            date_iso,
            "--input",
            str(digest_path),
            "--model",
            model,
        ]
    else:
        cmd = [
            sys.executable,
            "chimera_v2c/tools/apply_llm_injuries_v2.py",
            "--league",
            league.lower(),
            "--date",
            date_iso,
            "--input",
            str(digest_path),
            "--model",
            model,
        ]
    run_cmd(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-league v2c pipeline.")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--leagues", default="nba,nhl", help="Comma list or 'all' (default: nba,nhl).")
    ap.add_argument("--refresh-factors", action="store_true", help="Refresh factors (where supported).")
    ap.add_argument("--fit-calibration", action="store_true", help="Fit calibration before planning.")
    ap.add_argument(
        "--llm-injuries",
        action="store_true",
        help="After refresh_slate_updates, run LLM injury merge into injury_adjustments.json using the generated digest.",
    )
    ap.add_argument(
        "--moneypuck-injuries",
        action="store_true",
        help="(NHL only) Refresh MoneyPuck current injuries snapshot + write a slate-filtered digest under chimera_v2c/data/.",
    )
    ap.add_argument(
        "--llm-injuries-source",
        choices=["espn", "moneypuck"],
        default="espn",
        help="Input source for the LLM injuries prompt (default: espn). NHL supports moneypuck.",
    )
    ap.add_argument(
        "--llm-always",
        action="store_true",
        help="When using --llm-injuries-source moneypuck, always call the LLM even if the snapshot is unchanged (default: on-change).",
    )
    ap.add_argument(
        "--llm-model",
        default="gpt-5.1",
        help="OpenAI model for LLM injury merge (default: gpt-5.1).",
    )
    ap.add_argument("--backtest-days", type=int, help="If set, run backtest over last N days (log-only).")
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip refresh_slate_updates + preflight checks (injury/data freshness + start windows).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Planner dry-run only (default True).")
    args = ap.parse_args()

    load_env_from_env_list()
    date_iso = _iso_date(args.date)
    leagues = leagues_from_arg(args.leagues)
    for league in leagues:
        if not args.skip_preflight:
            refresh_slate_updates(league=league, date=date_iso)
            mp_result = None
            if args.moneypuck_injuries and league.lower() == "nhl":
                from chimera_v2c.tools.update_moneypuck_injuries_nhl import update_moneypuck_injuries_nhl

                mp_result = update_moneypuck_injuries_nhl(date_iso=date_iso, write_digest=True, force=False)
                print(f"[info] MoneyPuck injuries changed={mp_result.get('changed')}")
                if mp_result.get("diff_path"):
                    print(f"[alert] MoneyPuck diff: {mp_result.get('diff_path')}")

            if args.llm_injuries:
                if args.llm_injuries_source == "moneypuck":
                    if league.lower() != "nhl":
                        raise SystemExit("[error] --llm-injuries-source moneypuck is only supported for NHL")
                    if (not args.llm_always) and mp_result is not None and not bool(mp_result.get("changed")):
                        print("[info] skipping LLM injuries (MoneyPuck snapshot unchanged)")
                    else:
                        digest = (mp_result or {}).get("digest_path") if mp_result else None
                        if not digest:
                            digest = str(Path("chimera_v2c/data") / f"moneypuck_injuries_{date_iso}_nhl.txt")
                        # Call the NHL applier directly so we can pass the MoneyPuck digest.
                        run_cmd([
                            sys.executable,
                            "chimera_v2c/tools/apply_llm_nhl_injuries.py",
                            "--date",
                            date_iso,
                            "--input",
                            str(digest),
                            "--model",
                            args.llm_model,
                        ])
                else:
                    _apply_llm_injuries(league=league, date_iso=date_iso, model=args.llm_model)
            # Require fresh injuries/data and start windows before any planner/log calls.
            check_injury_freshness(12, league)
            check_data_freshness(24 * 7)
            check_start_windows(league, date_iso, 30)
        if league == "nhl":
            if args.refresh_factors:
                run_cmd([
                    sys.executable,
                    "chimera_v2c/tools/etl_nhl_factors.py",
                    "--out",
                    "chimera_v2c/data/team_four_factors_nhl.json",
                ])
            if args.fit_calibration:
                run_cmd([
                    sys.executable,
                    "chimera_v2c/tools/fit_calibration_from_daily.py",
                    "--league",
                    "nhl",
                    "--model-col",
                    "v2c",
                    "--allow-empty",
                    "--out",
                    "chimera_v2c/data/calibration_params_nhl.json",
                ])
            planner = [
                sys.executable,
                "chimera_v2c/tools/run_daily.py",
                "--config",
                "chimera_v2c/config/nhl_defaults.yaml",
                "--date",
                date_iso,
            ]
            planner.append("--skip-preflight")
            run_cmd(planner)
            log_cmd = [
                sys.executable,
                "chimera_v2c/tools/log_plan.py",
                "--date",
                date_iso,
                "--config",
                "chimera_v2c/config/nhl_defaults.yaml",
            ]
            log_cmd.append("--skip-preflight")
            run_cmd(log_cmd)
            if args.backtest_days:
                # Backtest over last N days ending at target date
                from datetime import datetime, timedelta
                try:
                    target_dt = datetime.fromisoformat(date_iso)
                    start_dt = target_dt - timedelta(days=args.backtest_days)
                    dr = f"{start_dt.date()}:{target_dt.date()}"
                except Exception:
                    dr = f"{date_iso}:{date_iso}"
                run_cmd([
                    sys.executable,
                    "chimera_v2c/tools/nhl_backtest.py",
                    "--date-range",
                    dr,
                    "--league",
                    "nhl",
                ])
        elif league == "nba":
            if args.fit_calibration:
                run_cmd([
                    sys.executable,
                    "chimera_v2c/tools/fit_calibration_from_daily.py",
                    "--league",
                    "nba",
                    "--model-col",
                    "v2c",
                    "--allow-empty",
                    "--out",
                    "chimera_v2c/data/calibration_params_nba.json",
                ])
            planner = [
                sys.executable,
                "chimera_v2c/tools/run_daily.py",
                "--config",
                "chimera_v2c/config/defaults.yaml",
                "--date",
                date_iso,
            ]
            planner.append("--skip-preflight")
            run_cmd(planner)
            log_cmd = [
                sys.executable,
                "chimera_v2c/tools/log_plan.py",
                "--date",
                date_iso,
                "--config",
                "chimera_v2c/config/defaults.yaml",
            ]
            log_cmd.append("--skip-preflight")
            run_cmd(log_cmd)
            if args.backtest_days:
                from datetime import datetime, timedelta
                try:
                    target_dt = datetime.fromisoformat(date_iso)
                    start_dt = target_dt - timedelta(days=args.backtest_days)
                    dr = f"{start_dt.date()}:{target_dt.date()}"
                except Exception:
                    dr = f"{date_iso}:{date_iso}"
                run_cmd([
                    sys.executable,
                    "chimera_v2c/tools/nba_backtest.py",
                    "--date-range",
                    dr,
                    "--league",
                    "nba",
                ])
        elif league == "nfl":
            # Stub: plan/log only with nfl_defaults.yaml if present
            planner = [
                sys.executable,
                "chimera_v2c/tools/run_daily.py",
                "--config",
                "chimera_v2c/config/nfl_defaults.yaml",
                "--date",
                date_iso,
            ]
            planner.append("--skip-preflight")
            run_cmd(planner)
            log_cmd = [
                sys.executable,
                "chimera_v2c/tools/log_plan.py",
                "--date",
                date_iso,
                "--config",
                "chimera_v2c/config/nfl_defaults.yaml",
            ]
            log_cmd.append("--skip-preflight")
            run_cmd(log_cmd)
        else:
            print(f"[warn] unsupported league: {league}")


if __name__ == "__main__":
    main()
