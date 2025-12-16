from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import os
from pathlib import Path

# Fix Imports
sys.path.insert(0, os.getcwd())

from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.pipeline import build_daily_plan, plans_to_json
from chimera_v2c.lib.env_loader import load_env_from_env_list


def parse_date(val: str | None) -> datetime.date:
    if not val:
        return datetime.date.today()
    return datetime.datetime.strptime(val, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(description="v2c pre-game planner (dry run).")
    parser.add_argument("--config", default="chimera_v2c/config/defaults.yaml", help="Path to config yaml")
    parser.add_argument("--date", help="Date YYYY-MM-DD (default: today)")
    parser.add_argument("--out", help="Optional path to save JSON plan")
    parser.add_argument(
        "--llm-injuries",
        action="store_true",
        help="Run LLM injury merge into injury_adjustments.json using the ESPN digest (default: off).",
    )
    parser.add_argument(
        "--llm-input",
        help="Optional injury/news text file for the LLM injury merger (default: use chimera_v2c/data/news_<date>_<league>.txt).",
    )
    parser.add_argument("--llm-model", default="gpt-5.1", help="OpenAI model for LLM injury merge (default: gpt-5.1).")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip injury/data/time window preflight checks (default: enforce).",
    )
    args = parser.parse_args()

    load_env_from_env_list()
    cfg = V2CConfig.load(args.config)
    target_date = parse_date(args.date)

    date_iso = target_date.strftime("%Y-%m-%d")
    if not args.skip_preflight:
        from chimera_v2c.tools.refresh_slate_updates import refresh_slate_updates
        from chimera_v2c.tools.preflight_check import check_data_freshness, check_injury_freshness, check_start_windows

        refresh_slate_updates(league=cfg.league.lower(), date=date_iso)
        if args.llm_injuries:
            default_digest = Path("chimera_v2c/data") / f"news_{date_iso}_{cfg.league.lower()}.txt"
            llm_input = Path(args.llm_input) if args.llm_input else default_digest
            if not llm_input.exists():
                raise SystemExit(f"[error] LLM injury input missing: {llm_input}")

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
        check_start_windows(cfg.league.lower(), date_iso, 30)
    plans = build_daily_plan(cfg, target_date)

    print(f"=== Chimera v2c Plan for {target_date} ({cfg.league.upper()}) ===")
    for p in plans:
        p_home = p.p_final
        p_away = 1 - p_home
        print(f"- {p.key}: home_p={p_home:.3f}, away_p={p_away:.3f}")
        for s in p.sides:
            mid_str = f"{s.market.mid:.3f}" if s.market and s.market.mid is not None else "NA"
            edge_str = f"{s.edge:.3f}" if s.edge is not None else "NA"
            stake_str = f"{s.stake_fraction:.4f}" if s.stake_fraction is not None else "NA"
            ticker = s.market.ticker if s.market else "N/A"
            print(f"    yes={s.yes_team}: mid={mid_str}, p_yes={s.p_yes:.3f}, edge={edge_str}, stake_frac={stake_str}, ticker={ticker}")
            if s.planned_orders:
                for o in s.planned_orders:
                    print(f"      order: {o.market_ticker} {o.side} @{o.target_price:.2f} stake_frac={o.stake_fraction:.4f}")
            else:
                print("      [skip] no planned orders (edge/guardrails)")
        if p.diagnostics.get("reasons"):
            print(f"    diag: reasons={p.diagnostics.get('reasons')}")
        print(
            "    diag: "
            f"market_ts={p.diagnostics.get('market_fetch_ts')} "
            f"ratings_mtime={p.diagnostics.get('ratings_mtime')} "
            f"injury_mtime={p.diagnostics.get('injury_mtime')} "
            f"injury_home_delta={p.diagnostics.get('injury_home_delta')} "
            f"injury_away_delta={p.diagnostics.get('injury_away_delta')} "
            f"sharp_prior={p.diagnostics.get('sharp_prior_used')} "
            f"ws_overlay={p.diagnostics.get('ws_overlay_used')}"
        )

    if args.out:
        payload = plans_to_json(plans)
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"Wrote plan to {args.out}")


if __name__ == "__main__":
    main()
