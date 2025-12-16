"""
Shadow logger: collect reasons why games/sides have no orders (no quotes, low edge, guardrails).

Usage:
  PYTHONPATH=. python chimera_v2c/tools/shadow_no_edge_log.py --date YYYY-MM-DD --config chimera_v2c/config/defaults.yaml

Output: writes a JSON file with diagnostics per game to reports/execution_logs/shadow_no_edge_<date>.json
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.pipeline import build_daily_plan
from chimera_v2c.tools.run_daily import parse_date
from chimera_v2c.tools.preflight_check import check_injury_freshness, check_data_freshness, check_start_windows

OUT_DIR = Path("reports/execution_logs")


def main() -> None:
    ap = argparse.ArgumentParser(description="Shadow log: reasons for no-edge/no-orders per game.")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--config", default="chimera_v2c/config/defaults.yaml")
    ap.add_argument("--skip-preflight", action="store_true", help="Skip guardrail checks (default: enforce).")
    args = ap.parse_args()

    target_date = parse_date(args.date)
    cfg = V2CConfig.load(args.config)
    if not args.skip_preflight:
        from chimera_v2c.tools.refresh_slate_updates import refresh_slate_updates

        refresh_slate_updates(league=cfg.league.lower(), date=target_date.strftime("%Y-%m-%d"))
        check_injury_freshness(12, cfg.league.lower())
        check_data_freshness(24 * 7)
        check_start_windows(cfg.league.lower(), target_date.strftime("%Y-%m-%d"), 30)

    plans = build_daily_plan(cfg, target_date)
    entries = []
    for p in plans:
        entry = {
            "date": target_date.isoformat(),
            "league": cfg.league.lower(),
            "matchup": p.key,
            "p_home": p.p_final,
            "components": p.components,
            "edge": p.edge,
            "stake_fraction": p.stake_fraction,
            "diagnostics": p.diagnostics,
        }
        entries.append(entry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"shadow_no_edge_{target_date.strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"[info] wrote {len(entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
