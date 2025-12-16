"""
Log the daily v2c plan (including mids and p_yes) to a JSON/CSV for analysis.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/log_plan.py --date YYYY-MM-DD
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from chimera_v2c.tools.run_daily import parse_date
from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.pipeline import build_daily_plan
from chimera_v2c.tools.preflight_check import check_injury_freshness, check_start_windows, check_data_freshness

LOG_JSON = Path("reports/execution_logs/v2c_plan_log.json")


def log_plan(date_str: str, cfg_path: str) -> None:
    cfg = V2CConfig.load(cfg_path)
    target_date = parse_date(date_str)
    plans = build_daily_plan(cfg, target_date)
    entries = []
    league = (cfg.league or "").lower()
    for p in plans:
        home = p.home
        away = p.away
        p_home = p.p_final
        p_away = 1 - p_home
        planned_trade_yes = p.yes_team if (p.stake_fraction is not None and p.planned_orders) else None
        for s in p.sides:
            mid = s.market.mid if s.market else None
            edge_yes = s.p_yes - mid if (s.p_yes is not None and mid is not None) else None
            entries.append(
                {
                    "date": target_date.isoformat(),
                    "league": league,
                    "matchup": f"{away}@{home}",
                    "home": home,
                    "away": away,
                    "yes_team": s.yes_team,
                    "mid": mid,
                    "p_home": p_home,
                    "p_away": p_away,
                    "p_yes": s.p_yes,
                    "edge_yes": edge_yes,
                    "stake_fraction": s.stake_fraction,
                    "selected": bool(planned_trade_yes and s.yes_team == planned_trade_yes),
                    "ticker": s.market.ticker if s.market else None,
                }
            )
    LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    if LOG_JSON.exists():
        try:
            existing = json.loads(LOG_JSON.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                existing.extend(entries)
                entries = existing
        except Exception:
            pass
    LOG_JSON.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"[info] logged {len(plans)} games to {LOG_JSON}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Log v2c plan to JSON for analysis.")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--config", default="chimera_v2c/config/defaults.yaml")
    ap.add_argument(
        "--llm-injuries",
        action="store_true",
        help="Run LLM injury merge into injury_adjustments.json using the ESPN digest (default: off).",
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
        "--llm-input",
        help="Optional injury/news text file for the LLM injury merger (default: use chimera_v2c/data/news_<date>_<league>.txt).",
    )
    ap.add_argument("--llm-model", default="gpt-5.1", help="OpenAI model for LLM injury merge (default: gpt-5.1).")
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip injury/data/time window checks (default: enforce).",
    )
    args = ap.parse_args()

    target_date = parse_date(args.date)
    cfg = V2CConfig.load(args.config)
    if not args.skip_preflight:
        from chimera_v2c.tools.refresh_slate_updates import refresh_slate_updates

        refresh_slate_updates(league=cfg.league.lower(), date=target_date.strftime("%Y-%m-%d"))
        mp_result = None
        if args.moneypuck_injuries and cfg.league.lower() == "nhl":
            from chimera_v2c.tools.update_moneypuck_injuries_nhl import update_moneypuck_injuries_nhl

            mp_result = update_moneypuck_injuries_nhl(date_iso=target_date.strftime("%Y-%m-%d"), write_digest=True, force=False)
            print(f"[info] MoneyPuck injuries changed={mp_result.get('changed')}")
            if mp_result.get("diff_path"):
                print(f"[alert] MoneyPuck diff: {mp_result.get('diff_path')}")
        if args.llm_injuries:
            date_iso = target_date.strftime("%Y-%m-%d")
            llm_input: Path | None
            if args.llm_injuries_source == "moneypuck":
                if cfg.league.lower() != "nhl":
                    raise SystemExit("[error] --llm-injuries-source moneypuck is only supported for NHL")
                if args.llm_input:
                    llm_input = Path(args.llm_input)
                else:
                    digest = (mp_result or {}).get("digest_path") if mp_result else None
                    if not digest:
                        digest = str(Path("chimera_v2c/data") / f"moneypuck_injuries_{date_iso}_nhl.txt")
                    llm_input = Path(str(digest))
                if (not args.llm_always) and mp_result is not None and not bool(mp_result.get("changed")):
                    print("[info] skipping LLM injuries (MoneyPuck snapshot unchanged)")
                    llm_input = None
            else:
                default_digest = Path("chimera_v2c/data") / f"news_{date_iso}_{cfg.league.lower()}.txt"
                llm_input = Path(args.llm_input) if args.llm_input else default_digest
            if llm_input is not None and not llm_input.exists():
                raise SystemExit(f"[error] LLM injury input missing: {llm_input}")

            if llm_input is not None:
                llm_script = (
                    "chimera_v2c/tools/apply_llm_nhl_injuries.py"
                    if cfg.league.lower() == "nhl"
                    else "chimera_v2c/tools/apply_llm_injuries_v2.py"
                )
                cmd = [
                    sys.executable,
                    llm_script,
                    "--date",
                    date_iso,
                    "--input",
                    str(llm_input),
                    "--model",
                    args.llm_model,
                ]
                if cfg.league.lower() != "nhl":
                    cmd.extend(["--league", cfg.league.lower()])
                print(f"[info] running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
                if result.returncode != 0:
                    raise SystemExit(f"[error] command failed: {' '.join(cmd)}")
        check_injury_freshness(12, cfg.league.lower())
        check_data_freshness(24 * 7)
        check_start_windows(cfg.league.lower(), target_date.strftime("%Y-%m-%d"), 30)

    log_plan(args.date, args.config)


if __name__ == "__main__":
    main()
