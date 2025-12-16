import argparse
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Imports
sys.path.insert(0, os.getcwd())

from chimera_v2c.lib import kalshi_utils
from chimera_v2c.lib import kalshi_portfolio
from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib import drawdown_monitor

from chimera_v2c.src import logging as v2c_logging
from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.pipeline import build_daily_plan
from chimera_v2c.src.sentinel import is_system_halted, is_game_halted

LEDGER_PATH = 'reports/specialist_performance/specialist_manual_ledger.csv'


SENTINEL_PATH = Path("chimera_v2c/data/STOP_TRADING.flag")


def check_sentinel() -> bool:
    """Simple file-based halt gate."""
    return not SENTINEL_PATH.exists()

def _get_bankroll_cents() -> int:
    try:
        pf = kalshi_portfolio.load_portfolio()
        bal = pf.get("balance")
        if isinstance(bal, (int, float)):
            return int(float(bal))
        if isinstance(bal, dict):
            for key in ("available_cash", "cash", "balance", "portfolio_value"):
                if key in bal and bal[key] is not None:
                    return int(float(bal[key]))
    except Exception as exc: 
        print(f"[warn] unable to fetch portfolio balance: {exc}")
    return 10000

def _safe_price_cents(target: int, yes_bid: int | None, yes_ask: int | None) -> int:
    if yes_ask is not None and target >= yes_ask:
        target = yes_ask - 1
    if target <= 0:
        target = 1
    if yes_bid is not None and target <= yes_bid:
        target = yes_bid 
    return max(1, min(target, 98))


def _fetch_market_snapshot(ticker: str) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """
    Fetch latest yes_bid/yes_ask/mid for a ticker (best-effort; uses get_markets).
    Returns (mid_float, yes_bid_cents, yes_ask_cents).
    """
    try:
        markets = kalshi_utils.get_markets(get_all=True, status="open")
    except Exception as exc:
        print(f"[warn] get_markets failed for {ticker}: {exc}")
        return None, None, None
    for m in markets:
        if m.get("ticker") == ticker:
            yes_bid = m.get("yes_bid")
            yes_ask = m.get("yes_ask")
            mid = None
            try:
                if yes_bid is not None and yes_ask is not None:
                    mid = (float(yes_bid) + float(yes_ask)) / 200.0
                elif m.get("mid") is not None:
                    mid = float(m["mid"]) / 100.0
            except Exception:
                mid = None
            return mid, yes_bid, yes_ask
    return None, None, None

def main() -> None:
    parser = argparse.ArgumentParser(description="Execute v2c plan (maker-only).")
    parser.add_argument("--config", default="chimera_v2c/config/defaults.yaml")
    parser.add_argument("--date", help="YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Print but do not place orders")
    parser.add_argument("--skip-halt", action="store_true", help="Bypass sentinel/drawdown halt")
    parser.add_argument(
        "--llm-injuries",
        action="store_true",
        help="Run LLM injury merge into injury_adjustments.json using the ESPN digest (default: off).",
    )
    parser.add_argument(
        "--moneypuck-injuries",
        action="store_true",
        help="(NHL only) Refresh MoneyPuck current injuries snapshot + write a slate-filtered digest under chimera_v2c/data/.",
    )
    parser.add_argument(
        "--llm-injuries-source",
        choices=["espn", "moneypuck"],
        default="espn",
        help="Input source for the LLM injuries prompt (default: espn). NHL supports moneypuck.",
    )
    parser.add_argument(
        "--llm-always",
        action="store_true",
        help="When using --llm-injuries-source moneypuck, always call the LLM even if the snapshot is unchanged (default: on-change).",
    )
    parser.add_argument(
        "--llm-input",
        help="Optional injury/news text file for the LLM injury merger (default: use chimera_v2c/data/news_<date>_<league>.txt).",
    )
    parser.add_argument("--llm-model", default="gpt-5.1", help="OpenAI model for LLM injury merge (default: gpt-5.1).")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip injury/data/time window checks (default: enforce).",
    )
    parser.add_argument("--max-reprices", type=int, default=0, help="Number of reprice attempts before giving up.")
    parser.add_argument("--reprice-cents", type=int, default=2, help="Nudge maker price toward mid by this many cents on reprice.")
    parser.add_argument("--ttl-seconds", type=int, default=0, help="Optional sleep before cancel loop (not cancelling if 0).")
    args = parser.parse_args()

    # Sentinel Check
    if not args.skip_halt and is_system_halted():
        print("[STOP] Sentinel Halt Active.")
        return

    load_env_from_env_list()
    cfg = V2CConfig.load(args.config)
    target_date_obj = datetime.date.today() if not args.date else datetime.datetime.strptime(args.date, "%Y-%m-%d").date()
    target_date_str = target_date_obj.strftime("%Y-%m-%d")

    if not args.skip_preflight:
        from chimera_v2c.tools.preflight_check import check_injury_freshness, check_start_windows, check_data_freshness
        from chimera_v2c.tools.refresh_slate_updates import refresh_slate_updates

        refresh_slate_updates(league=cfg.league.lower(), date=target_date_str)
        mp_result = None
        if args.moneypuck_injuries and cfg.league.lower() == "nhl":
            from chimera_v2c.tools.update_moneypuck_injuries_nhl import update_moneypuck_injuries_nhl

            mp_result = update_moneypuck_injuries_nhl(date_iso=target_date_str, write_digest=True, force=False)
            print(f"[info] MoneyPuck injuries changed={mp_result.get('changed')}")
            if mp_result.get("diff_path"):
                print(f"[alert] MoneyPuck diff: {mp_result.get('diff_path')}")
        if args.llm_injuries:
            llm_input: Path | None
            if args.llm_injuries_source == "moneypuck":
                if cfg.league.lower() != "nhl":
                    raise SystemExit("[error] --llm-injuries-source moneypuck is only supported for NHL")
                if args.llm_input:
                    llm_input = Path(args.llm_input)
                else:
                    digest = (mp_result or {}).get("digest_path") if mp_result else None
                    if not digest:
                        digest = str(Path("chimera_v2c/data") / f"moneypuck_injuries_{target_date_str}_nhl.txt")
                    llm_input = Path(str(digest))
                if (not args.llm_always) and mp_result is not None and not bool(mp_result.get("changed")):
                    print("[info] skipping LLM injuries (MoneyPuck snapshot unchanged)")
                    llm_input = None
            else:
                default_digest = Path("chimera_v2c/data") / f"news_{target_date_str}_{cfg.league.lower()}.txt"
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
                    target_date_str,
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
        check_start_windows(cfg.league.lower(), target_date_str, 30)

    plans = build_daily_plan(cfg, target_date_obj)
    bankroll_cents = _get_bankroll_cents()
    print(f"[info] bankroll_cents={bankroll_cents}")

    if not args.skip_halt and drawdown_monitor.should_halt_trading():
        print("[STOP] Drawdown halt active. Use --skip-halt to override.")
        return

    orders_to_send: List[dict] = []
    used_fraction = 0.0
    for p in plans:
        # Check per-game halt
        # p.key is like "GSW@PHI"
        if is_game_halted(p.key):
            print(f"[SKIP] Game {p.key} is halted by sentinel.")
            continue

        if not p.planned_orders:
            continue
        if p.stake_fraction is None:
            continue
        po = p.planned_orders[0]
        if not po.market_ticker:
            continue
        yes_team = getattr(p, "yes_team", None)
        
        price_cents = int(round(po.target_price * 100))
        count = int((bankroll_cents * po.stake_fraction) / max(price_cents, 1))
        if count < 1:
            count = 1
            
        yes_bid = p.market.yes_bid if p.market else None
        yes_ask = p.market.yes_ask if p.market else None
        price_cents = _safe_price_cents(price_cents, yes_bid, yes_ask)
        # Respect daily cap at execution time as well
        if used_fraction + (po.stake_fraction or 0) > cfg.daily_max_fraction:
            print(f"[SKIP] {p.key} would exceed daily_max_fraction ({used_fraction:.4f} + {po.stake_fraction:.4f} > {cfg.daily_max_fraction:.4f})")
            continue
        used_fraction += po.stake_fraction or 0

        orders_to_send.append(
            {
                "ticker": po.market_ticker,
                "side": po.side,
                "yes_team": yes_team,
                "count": count,
                "price_cents": price_cents,
                "edge": p.edge,
                "stake_fraction": po.stake_fraction,
                "plan_details": {'key': p.key, 'date': target_date_str, 'p_final': p.p_final, 'p_yes': getattr(p, "p_yes_selected", None)},
                "anchor_price_cents": int(round(po.target_price * 100)),
                "planned_mid": p.market.mid if p.market else None,
                "planned_spread": (yes_ask - yes_bid) if (yes_ask is not None and yes_bid is not None) else None,
            }
        )

    if not orders_to_send:
        print("[info] no eligible orders to send.")
        v2c_logging.append_log(
            [
                {
                    "date": target_date_str,
                    "ticker": "",
                    "side": "",
                    "count": "",
                    "price_cents": "",
                    "edge": "",
                    "stake_fraction": "",
                    "status": "noop",
                    "message": "no eligible orders",
                }
            ]
        )
        return

    log_rows = []
    for o in orders_to_send:
        edge_val = o["edge"] if o["edge"] is not None else float("nan")
        yes_str = f" yes_team={o['yes_team']}" if o.get("yes_team") else ""
        print(f"[plan]{yes_str} {o['ticker']} {o['side']} count={o['count']} @ {o['price_cents']}c edge={edge_val:.3f}")
        if args.dry_run:
            log_rows.append(
                {
                    "date": target_date_str,
                    "ticker": o["ticker"],
                    "side": o["side"],
                    "count": o["count"],
                    "price_cents": o["price_cents"],
                    "edge": o["edge"],
                    "stake_fraction": o["stake_fraction"],
                    "status": "dry_run",
                    "message": f"not placed{yes_str}",
                    "anchor_price_cents": o.get("anchor_price_cents"),
                    "planned_mid": o.get("planned_mid"),
                    "planned_spread": o.get("planned_spread"),
                    "placed_mid": None,
                    "placed_spread": None,
                    "attempt": 0,
                }
            )
            continue
        attempts = 0
        placed = False
        current_price_cents = o["price_cents"]
        while attempts <= args.max_reprices:
            attempts += 1
            placed_mid = None
            placed_spread = None
            # Best-effort reprice using latest market snapshot
            snap_mid, snap_bid, snap_ask = _fetch_market_snapshot(o["ticker"])
            if snap_mid is not None:
                placed_mid = snap_mid
            if snap_bid is not None and snap_ask is not None:
                placed_spread = (snap_ask - snap_bid) / 100.0
            if snap_mid is not None and args.reprice_cents > 0 and attempts > 1:
                target_cents = int(round(snap_mid * 100)) - args.reprice_cents
                current_price_cents = _safe_price_cents(target_cents, snap_bid, snap_ask)
            try:
                resp = kalshi_utils.place_order(
                    ticker=o["ticker"],
                    side=o["side"],
                    count=o["count"],
                    price=current_price_cents,
                    action="buy",
                )
                print(f"  -> placed: {resp}")
                log_rows.append(
                    {
                        "date": target_date_str,
                        "ticker": o["ticker"],
                        "side": o["side"],
                        "count": o["count"],
                        "price_cents": current_price_cents,
                        "edge": o["edge"],
                        "stake_fraction": o["stake_fraction"],
                        "status": "placed",
                        "message": str(resp),
                        "anchor_price_cents": o.get("anchor_price_cents"),
                        "planned_mid": o.get("planned_mid"),
                        "planned_spread": o.get("planned_spread"),
                        "placed_mid": placed_mid,
                        "placed_spread": placed_spread,
                        "attempt": attempts,
                    }
                )
                placed = True
                break
            except Exception as exc: 
                print(f"  !! error placing {o['ticker']} (attempt {attempts}): {exc}")
                if "service_unavailable" not in str(exc) and "503" not in str(exc):
                    log_rows.append(
                        {
                            "date": target_date_str,
                            "ticker": o["ticker"],
                            "side": o["side"],
                            "count": o["count"],
                            "price_cents": current_price_cents,
                            "edge": o["edge"],
                            "stake_fraction": o["stake_fraction"],
                            "status": "error",
                            "message": str(exc),
                            "anchor_price_cents": o.get("anchor_price_cents"),
                            "planned_mid": o.get("planned_mid"),
                            "planned_spread": o.get("planned_spread"),
                            "placed_mid": placed_mid,
                            "placed_spread": placed_spread,
                            "attempt": attempts,
                        }
                    )
                    break
            if attempts <= args.max_reprices:
                continue
        if not placed and args.ttl_seconds > 0:
            time.sleep(min(args.ttl_seconds, 5))

    if log_rows:
        v2c_logging.append_log(log_rows)

if __name__ == "__main__":
    main()
