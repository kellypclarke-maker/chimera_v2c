"""
Lightweight scheduler to run injuries (deterministic + optional LLM) and plan near tip.

Behavior:
- Fetches game start times for the league/date.
- Sleeps until (earliest start - buffer minutes) if needed.
- Refreshes injuries + team-filtered news from ESPN (all leagues).
- Optionally runs LLM injury merger (from the ESPN digest by default).
- Runs run_daily.py for the plan (preflight can be skipped via flag).

Usage (keep the process running):
  PYTHONPATH=. python chimera_v2c/tools/scheduled_injury_plan.py \
    --league nba --date YYYY-MM-DD \
    --llm-injuries \
    --config chimera_v2c/config/defaults.yaml \
    --skip-preflight
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib import nhl_scoreboard

LEAGUE_FETCHERS: Dict[str, Callable[[str], Dict]] = {
    "nba": nhl_scoreboard.fetch_nba_scoreboard,
    "nhl": nhl_scoreboard.fetch_nhl_scoreboard,
    "nfl": nhl_scoreboard.fetch_nfl_scoreboard,
}


def parse_start_times(league: str, date: str):
    fetcher = LEAGUE_FETCHERS.get(league)
    if not fetcher:
        raise SystemExit(f"[error] unsupported league: {league}")
    sb = fetcher(date)
    if sb.get("status") != "ok":
        raise SystemExit(f"[error] scoreboard fetch failed: {sb.get('message')}")
    starts = []
    for g in sb.get("games") or []:
        start_str = g.get("start_time")
        if not start_str:
            continue
        try:
            starts.append(datetime.fromisoformat(start_str.replace("Z", "+00:00")))
        except Exception:
            continue
    return starts


def sleep_until(target: datetime):
    now = datetime.now(timezone.utc)
    delta = (target - now).total_seconds()
    if delta <= 0:
        return
    print(f"[info] sleeping {delta/60:.1f} minutes until {target.isoformat()}")
    time.sleep(delta)


def run_cmd(cmd: list[str]) -> None:
    print(f"[info] running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise SystemExit(f"[error] command failed: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Schedule injury+plan run near tip.")
    ap.add_argument("--league", default="nba", help="League (nba|nhl|nfl).")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--config", default="chimera_v2c/config/defaults.yaml")
    ap.add_argument("--buffer-minutes", type=int, default=30, help="Run this many minutes before earliest start (default: 30).")
    ap.add_argument(
        "--llm-injuries",
        action="store_true",
        help="Run LLM injury merge into injury_adjustments.json using the ESPN digest (default: off).",
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
        "--llm-input",
        help="Optional injury/news text file to feed the LLM injury merger (default: use the ESPN digest written by refresh_slate_updates).",
    )
    ap.add_argument("--llm-model", default="gpt-5.1", help="OpenAI model for LLM injury merge (default: gpt-5.1).")
    ap.add_argument(
        "--moneypuck-injuries",
        action="store_true",
        help="(NHL only) Refresh MoneyPuck current injuries snapshot + write a slate-filtered digest under chimera_v2c/data/.",
    )
    ap.add_argument("--refresh-factors", action="store_true", help="Run NHL factors ETL before planning.")
    ap.add_argument("--fit-calibration", action="store_true", help="Fit calibration for league before planning.")
    ap.add_argument("--tune-weights", action="store_true", help="Run LLM tuner for NHL weights (log-only).")
    ap.add_argument("--backtest-days", type=int, help="If set, run nhl_backtest over last N days (log-only).")
    ap.add_argument("--skip-preflight", action="store_true", help="Pass --skip-preflight to planner.")
    ap.add_argument("--log-plan", action="store_true", help="Run log_plan.py after planner.")
    ap.add_argument("--ensure-daily-ledger", action="store_true", help="Run ensure_daily_ledger.py after planner.")
    args = ap.parse_args()

    load_env_from_env_list()
    starts = parse_start_times(args.league.lower(), args.date)
    if not starts:
        print("[warn] no start times found; running immediately.")
        target_time = datetime.now(timezone.utc)
    else:
        earliest = min(starts)
        target_time = earliest - timedelta(minutes=args.buffer_minutes)
    sleep_until(target_time)

    # Refresh injuries + news (all leagues)
    run_cmd([
        sys.executable,
        "chimera_v2c/tools/refresh_slate_updates.py",
        "--league",
        args.league.lower(),
        "--date",
        args.date,
    ])

    mp_result = None
    if args.moneypuck_injuries and args.league.lower() == "nhl":
        from chimera_v2c.tools.update_moneypuck_injuries_nhl import update_moneypuck_injuries_nhl

        mp_result = update_moneypuck_injuries_nhl(date_iso=args.date, write_digest=True, force=False)
        print(f"[info] MoneyPuck injuries changed={mp_result.get('changed')}")
        if mp_result.get("diff_path"):
            print(f"[alert] MoneyPuck diff: {mp_result.get('diff_path')}")

    # Optional LLM merge
    run_llm = bool(args.llm_injuries or args.llm_input)
    if run_llm:
        if args.llm_injuries_source == "moneypuck":
            if args.league.lower() != "nhl":
                raise SystemExit("[error] --llm-injuries-source moneypuck is only supported for NHL")
            if args.llm_input:
                llm_input = Path(args.llm_input)
            else:
                digest = (mp_result or {}).get("digest_path") if mp_result else None
                if not digest:
                    digest = str(Path("chimera_v2c/data") / f"moneypuck_injuries_{args.date}_nhl.txt")
                llm_input = Path(str(digest))
            if (not args.llm_always) and mp_result is not None and not bool(mp_result.get("changed")):
                print("[info] skipping LLM injuries (MoneyPuck snapshot unchanged)")
                run_llm = False
        else:
            default_digest = Path("chimera_v2c/data") / f"news_{args.date}_{args.league.lower()}.txt"
            llm_input = Path(args.llm_input) if args.llm_input else default_digest
        if not llm_input.exists():
            raise SystemExit(f"[error] LLM injury input missing: {llm_input}")

        if run_llm and args.league.lower() == "nhl":
            run_cmd([
                sys.executable,
                "chimera_v2c/tools/apply_llm_nhl_injuries.py",
                "--date",
                args.date,
                "--input",
                str(llm_input),
                "--model",
                args.llm_model,
            ])
        elif run_llm:
            run_cmd([
                sys.executable,
                "chimera_v2c/tools/apply_llm_injuries_v2.py",
                "--league",
                args.league.lower(),
                "--date",
                args.date,
                "--input",
                str(llm_input),
                "--model",
                args.llm_model,
            ])

    # Optional NHL factors refresh
    if args.refresh_factors and args.league.lower() == "nhl":
        run_cmd([
            sys.executable,
            "chimera_v2c/tools/etl_nhl_factors.py",
            "--out",
            "chimera_v2c/data/team_four_factors_nhl.json",
        ])

    # Optional calibration fit
    if args.fit_calibration:
        cal_out = {
            "nba": "chimera_v2c/data/calibration_params_nba.json",
            "nhl": "chimera_v2c/data/calibration_params_nhl.json",
            "nfl": "chimera_v2c/data/calibration_params.json",
        }.get(args.league.lower(), f"chimera_v2c/data/calibration_params_{args.league.lower()}.json")
        run_cmd([
            sys.executable,
            "chimera_v2c/tools/fit_calibration_from_daily.py",
            "--league",
            args.league.lower(),
            "--model-col",
            "v2c",
            "--allow-empty",
            "--out",
            cal_out,
        ])

    # Planner
    planner_cmd = [
        sys.executable,
        "chimera_v2c/tools/run_daily.py",
        "--date", args.date,
        "--config", args.config,
    ]
    if args.skip_preflight:
        planner_cmd.append("--skip-preflight")
    run_cmd(planner_cmd)

    if args.log_plan:
        lp_cmd = [
            sys.executable,
            "chimera_v2c/tools/log_plan.py",
            "--date",
            args.date,
            "--config",
            args.config,
        ]
        if args.skip_preflight:
            lp_cmd.append("--skip-preflight")
        run_cmd(lp_cmd)

    if args.ensure_daily_ledger:
        edl_cmd = [
            sys.executable,
            "chimera_v2c/tools/ensure_daily_ledger.py",
            "--date",
            args.date,
        ]
        run_cmd(edl_cmd)

    # Optional tuner (log-only, NHL)
    if args.tune_weights and args.league.lower() == "nhl":
        run_cmd([
            sys.executable,
            "chimera_v2c/tools/llm_tune_nhl_weights.py",
            "--days",
            "30",
        ])

    # Optional backtest (log-only, NHL)
    if args.backtest_days and args.league.lower() == "nhl":
        try:
            target_dt = datetime.fromisoformat(args.date)
            start_dt = target_dt - timedelta(days=args.backtest_days)
            dr = f"{start_dt.date()}:{target_dt.date()}"
        except Exception:
            dr = f"{args.date}:{args.date}"
        run_cmd([
            sys.executable,
            "chimera_v2c/tools/nhl_backtest.py",
            "--date-range",
            dr,
            "--league",
            "nhl",
        ])


if __name__ == "__main__":
    main()
