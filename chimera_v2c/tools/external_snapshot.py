#!/usr/bin/env python
"""
Capture timestamped external baselines (Kalshi + books + MoneyPuck) and (optionally) fill daily-ledger blanks.

Writes:
  reports/market_snapshots/YYYYMMDD_<league>_external_snapshot_<HHMMSSZ>.csv

Optionally fills (append-only) in:
  reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv
    - kalshi_mid
    - market_proxy
    - moneypuck (NHL only)

Safety:
- Respects reports/daily_ledgers/locked/YYYYMMDD.lock unless --force.
- Never overwrites non-blank daily-ledger cells; adds rows only when missing.
- Snapshots the daily-ledger file to reports/daily_ledgers/snapshots/ before writing.
- Dry-run by default; pass --write-snapshot and/or --apply.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/external_snapshot.py --league nhl --date YYYY-MM-DD --write-snapshot
  PYTHONPATH=. python chimera_v2c/tools/external_snapshot.py --league nhl --date YYYY-MM-DD --apply
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib.moneypuck import fetch_pregame, fetch_schedule, games_for_date, season_string_for_date
from chimera_v2c.lib.odds_history import BooksSnapshot, fetch_books_snapshot
from chimera_v2c.src.ledger.guard import LedgerGuardError, compute_append_only_diff, load_locked_dates, snapshot_file
from chimera_v2c.src.market_linker import fetch_markets, fetch_matchups, match_markets_to_games


DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"
LEDGER_SNAPSHOT_DIR = DAILY_DIR / "snapshots"

MARKET_SNAPSHOT_DIR = Path("reports/market_snapshots")

SERIES_TICKER_BY_LEAGUE = {
    "nba": "KXNBAGAME",
    "nhl": "KXNHLGAME",
    "nfl": "KXNFLGAME",
}


def _ledger_path(date_iso: str) -> Path:
    return DAILY_DIR / f"{date_iso.replace('-', '')}_daily_game_ledger.csv"


def ensure_daily_ledger(date_iso: str) -> None:
    path = _ledger_path(date_iso)
    if path.exists():
        return
    cmd = [
        sys.executable,
        "chimera_v2c/tools/ensure_daily_ledger.py",
        "--date",
        date_iso,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def _cell_blank(val: object) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and pd.isna(val):
        return True
    s = str(val).strip()
    if s == "":
        return True
    # Daily ledgers use `NR` as the canonical "reviewed missing" sentinel.
    # Treat it as fillable for market baselines.
    return s.upper() == "NR"


def _fmt_prob(p: Optional[float], decimals: int = 4) -> str:
    if p is None:
        return ""
    try:
        x = float(p)
    except Exception:
        return ""
    if x < 0.0 or x > 1.0:
        return ""
    return f"{x:.{decimals}f}"


def _parse_snapshot_iso(date_iso: str, *, snapshot_ts: Optional[str], snapshot_time: Optional[str]) -> str:
    if snapshot_ts and snapshot_time:
        raise SystemExit("[error] pass only one of --snapshot-ts or --snapshot-time")

    if snapshot_ts:
        s = snapshot_ts.strip()
        if not s:
            raise SystemExit("[error] empty --snapshot-ts")
        # Normalize common forms.
        if s.endswith("+00:00"):
            s = s.replace("+00:00", "Z")
        if s.endswith("z"):
            s = s[:-1] + "Z"
        if "T" not in s:
            raise SystemExit("[error] --snapshot-ts must be ISO like YYYY-MM-DDTHH:MM:SSZ")
        return s

    if snapshot_time:
        t = snapshot_time.strip()
        if not t:
            raise SystemExit("[error] empty --snapshot-time")
        if not t.endswith("Z"):
            raise SystemExit("[error] --snapshot-time must end with Z (e.g., 18:00:00Z)")
        return f"{date_iso}T{t}"

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _hhmmss_token(snapshot_iso: str) -> str:
    try:
        _, time_part = snapshot_iso.split("T", 1)
        time_part = time_part.replace("Z", "")
        hh, mm, ss = time_part.split(":", 2)
        return f"{hh}{mm}{ss}Z"
    except Exception:
        return datetime.now(timezone.utc).strftime("%H%M%SZ")


def _kalshi_home_mid(market_dict, *, home: str, away: str) -> Tuple[Optional[float], Dict[str, str]]:
    out: Dict[str, str] = {}
    home_quote = market_dict.get(home) if market_dict else None
    away_quote = market_dict.get(away) if market_dict else None

    if home_quote is not None:
        out["kalshi_ticker_home_yes"] = home_quote.ticker
        out["kalshi_yes_bid_home"] = "" if home_quote.yes_bid is None else str(int(home_quote.yes_bid))
        out["kalshi_yes_ask_home"] = "" if home_quote.yes_ask is None else str(int(home_quote.yes_ask))

    if away_quote is not None:
        out["kalshi_ticker_away_yes"] = away_quote.ticker
        out["kalshi_yes_bid_away"] = "" if away_quote.yes_bid is None else str(int(away_quote.yes_bid))
        out["kalshi_yes_ask_away"] = "" if away_quote.yes_ask is None else str(int(away_quote.yes_ask))

    home_mid = home_quote.mid if home_quote is not None else None
    if home_mid is None and away_quote is not None and away_quote.mid is not None:
        home_mid = 1.0 - float(away_quote.mid)

    return home_mid, out


def build_snapshot_rows(
    *,
    league: str,
    date_iso: str,
    snapshot_iso: str,
    kalshi_public_base: str,
    kalshi_status: str,
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Returns:
      rows: list of snapshot CSV rows (strings)
      fills: mapping matchup -> fields to fill in the daily ledger (strings)
    """
    lg = league.lower().strip()
    if lg not in SERIES_TICKER_BY_LEAGUE:
        raise SystemExit(f"[error] unsupported league: {league}")

    # Ensure Kalshi public base points at live markets unless the operator overrides it.
    os.environ["KALSHI_PUBLIC_BASE"] = kalshi_public_base

    target_date = datetime.strptime(date_iso, "%Y-%m-%d").date()

    matchups = fetch_matchups(lg, target_date)
    base_keys: List[Tuple[str, str, str]] = []
    for m in matchups:
        away = str(m["away"]).strip().upper()
        home = str(m["home"]).strip().upper()
        if not away or not home:
            continue
        base_keys.append((away, home, f"{away}@{home}"))

    # Kalshi snapshot (best effort).
    kalshi_map: Dict[str, Dict[str, object]] = {}
    try:
        kalshi_status_norm = (kalshi_status or "").strip().lower()
        if kalshi_status_norm not in ("open", "settled", "closed", "all"):
            raise SystemExit(f"[error] unsupported --kalshi-status: {kalshi_status}")

        # Kalshi public API uses `settled` for finalized markets; keep `closed` as a legacy alias.
        if kalshi_status_norm == "closed":
            kalshi_status_norm = "settled"

        if kalshi_status_norm == "all":
            open_markets = fetch_markets(lg, SERIES_TICKER_BY_LEAGUE[lg], use_private=False, status="open", target_date=target_date)
            settled_markets = fetch_markets(lg, SERIES_TICKER_BY_LEAGUE[lg], use_private=False, status="settled", target_date=target_date)
            by_ticker = {m.ticker: m for m in [*open_markets, *settled_markets]}
            markets = list(by_ticker.values())
        else:
            markets = fetch_markets(
                lg, SERIES_TICKER_BY_LEAGUE[lg], use_private=False, status=kalshi_status_norm, target_date=target_date
            )
        kalshi_market_map = match_markets_to_games(matchups, markets)
        for away, home, key in base_keys:
            md = kalshi_market_map.get(key, {})
            mid, extra = _kalshi_home_mid(md, home=home, away=away)
            kalshi_map[key] = {"kalshi_mid": mid, **extra}
    except Exception as exc:
        print(f"[warn] kalshi snapshot failed: {exc}")

    # Sportsbook snapshot (best effort).
    books_map: Dict[str, BooksSnapshot] = {}
    try:
        books_map = fetch_books_snapshot(league=lg, snapshot_iso=snapshot_iso)
    except Exception as exc:
        print(f"[warn] books snapshot failed: {exc}")

    # MoneyPuck snapshot (NHL only; best effort).
    moneypuck_map: Dict[str, Dict[str, str]] = {}
    if lg == "nhl":
        try:
            season = season_string_for_date(target_date)
            schedule = fetch_schedule(season)
            games = games_for_date(schedule, date_iso)
            for g in games:
                pre = fetch_pregame(g.game_id)
                if pre.moneypuck_home_win is None:
                    continue
                moneypuck_map[g.matchup] = {
                    "moneypuck": _fmt_prob(pre.moneypuck_home_win, 4),
                    "moneypuck_game_id": str(int(pre.game_id)),
                    "moneypuck_starting_goalie": "" if pre.starting_goalie is None else str(int(pre.starting_goalie)),
                }
        except Exception as exc:
            print(f"[warn] moneypuck snapshot failed: {exc}")

    rows: List[Dict[str, str]] = []
    fills: Dict[str, Dict[str, str]] = {}
    for away, home, key in base_keys:
        kal = kalshi_map.get(key, {})
        book = books_map.get(key, {})
        mp = moneypuck_map.get(key, {})

        kalshi_mid = kal.get("kalshi_mid")
        row: Dict[str, str] = {
            "date": date_iso,
            "league": lg,
            "matchup": key,
            "snapshot_ts": snapshot_iso,
            "kalshi_mid": _fmt_prob(float(kalshi_mid), 4) if kalshi_mid is not None else "",
            "kalshi_ticker_home_yes": str(kal.get("kalshi_ticker_home_yes") or ""),
            "kalshi_yes_bid_home": str(kal.get("kalshi_yes_bid_home") or ""),
            "kalshi_yes_ask_home": str(kal.get("kalshi_yes_ask_home") or ""),
            "kalshi_ticker_away_yes": str(kal.get("kalshi_ticker_away_yes") or ""),
            "kalshi_yes_bid_away": str(kal.get("kalshi_yes_bid_away") or ""),
            "kalshi_yes_ask_away": str(kal.get("kalshi_yes_ask_away") or ""),
            "market_proxy": _fmt_prob(book.get("market_proxy"), 4) if book.get("market_proxy") is not None else "",
            "books_home_ml": "" if book.get("books_home_ml") is None else str(int(book.get("books_home_ml"))),
            "books_away_ml": "" if book.get("books_away_ml") is None else str(int(book.get("books_away_ml"))),
            "moneypuck": str(mp.get("moneypuck") or ""),
            "moneypuck_game_id": str(mp.get("moneypuck_game_id") or ""),
            "moneypuck_starting_goalie": str(mp.get("moneypuck_starting_goalie") or ""),
        }
        rows.append(row)

        from chimera_v2c.src.ledger.formatting import format_prob_cell

        fills[key] = {
            "kalshi_mid": (
                format_prob_cell(row["kalshi_mid"], decimals=2, drop_leading_zero=True) if row["kalshi_mid"] else ""
            ),
            "market_proxy": (
                format_prob_cell(row["market_proxy"], decimals=2, drop_leading_zero=True)
                if row["market_proxy"]
                else ""
            ),
            "moneypuck": (
                format_prob_cell(row["moneypuck"], decimals=2, drop_leading_zero=True) if row["moneypuck"] else ""
            ),
        }

    return rows, fills


def apply_fills_to_daily_ledger(
    *,
    league: str,
    date_iso: str,
    fills: Dict[str, Dict[str, str]],
    apply: bool,
    force: bool,
) -> Tuple[int, int]:
    """
    Add missing matchup rows and fill blank cells only for:
      kalshi_mid, market_proxy, and (NHL) moneypuck.
    """
    ledger_path = _ledger_path(date_iso)
    if apply:
        ensure_daily_ledger(date_iso)
    if not ledger_path.exists():
        if not apply:
            return 0, 0
        raise SystemExit(f"[error] daily ledger missing: {ledger_path}")

    ymd = ledger_path.name.split("_")[0]
    locked = load_locked_dates(LOCK_DIR)
    if apply and ymd in locked and not force:
        raise SystemExit(f"[error] {ymd} is locked; refusing to modify {ledger_path} (pass --force to override)")

    original = pd.read_csv(ledger_path).fillna("")
    df = original.copy()

    key_fields = ["date", "league", "matchup"]
    for col in key_fields + [
        "kalshi_mid",
        "market_proxy",
        "moneypuck",
    ]:
        if col not in df.columns:
            df[col] = ""

    added_rows = 0
    filled_cells = 0

    lg = league.lower()
    for matchup, payload in fills.items():
        mask = (
            (df["date"].astype(str) == date_iso)
            & (df["league"].astype(str).str.lower() == lg)
            & (df["matchup"].astype(str) == matchup)
        )
        if not mask.any():
            new_row = {c: "" for c in df.columns}
            new_row.update({"date": date_iso, "league": lg, "matchup": matchup})
            if lg != "nhl" and "moneypuck" in df.columns:
                new_row["moneypuck"] = "NR"
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            added_rows += 1
            mask = (
                (df["date"].astype(str) == date_iso)
                & (df["league"].astype(str).str.lower() == lg)
                & (df["matchup"].astype(str) == matchup)
            )

        idx = df.index[mask][0]
        for field in (
            "kalshi_mid",
            "market_proxy",
            "moneypuck",
        ):
            new_val = (payload.get(field) or "").strip()
            if not new_val:
                continue
            if field not in df.columns:
                continue
            if not _cell_blank(df.at[idx, field]):
                continue
            df.at[idx, field] = new_val
            filled_cells += 1

    if not apply:
        return added_rows, filled_cells

    try:
        old_rows = original.fillna("").astype(str).to_dict("records")
        new_rows = df.fillna("").astype(str).to_dict("records")
        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=key_fields,
            value_fields=[c for c in df.columns if c not in key_fields],
            blank_sentinels={"NR"},
        )
    except LedgerGuardError as exc:
        raise SystemExit(f"[error] append-only guard failed: {exc}") from exc

    snapshot_file(ledger_path, LEDGER_SNAPSHOT_DIR)
    df.to_csv(ledger_path, index=False)
    return added_rows, filled_cells


def write_snapshot_csv(out_path: Path, rows: List[Dict[str, str]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["league", "matchup"])
    df.to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture timestamped external baseline snapshots (Kalshi + books + MoneyPuck).")
    ap.add_argument("--league", required=True, help="League (nba|nhl|nfl).")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD.")
    ap.add_argument("--snapshot-ts", help="UTC ISO timestamp like YYYY-MM-DDTHH:MM:SSZ (default: now).")
    ap.add_argument("--snapshot-time", help="UTC time suffix like HH:MM:SSZ to combine with --date.")
    ap.add_argument(
        "--kalshi-status",
        default="open",
        help="Kalshi market status to snapshot: open (default), settled, or all (open+settled). (closed is a legacy alias for settled)",
    )
    ap.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    ap.add_argument("--out-dir", default=str(MARKET_SNAPSHOT_DIR), help="Output dir for snapshot CSVs.")
    ap.add_argument("--write-snapshot", action="store_true", help="Write the snapshot CSV (default: dry-run).")
    ap.add_argument("--apply", action="store_true", help="Also fill blank daily-ledger market cells (implies --write-snapshot).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked daily ledgers (still fills blanks only).")
    args = ap.parse_args()

    load_env_from_env_list()

    date_iso = datetime.strptime(args.date, "%Y-%m-%d").date().isoformat()
    snapshot_iso = _parse_snapshot_iso(date_iso, snapshot_ts=args.snapshot_ts, snapshot_time=args.snapshot_time)
    time_token = _hhmmss_token(snapshot_iso)

    out_dir = Path(args.out_dir)
    ymd = date_iso.replace("-", "")
    out_path = out_dir / f"{ymd}_{args.league.lower()}_external_snapshot_{time_token}.csv"

    rows, fills = build_snapshot_rows(
        league=args.league,
        date_iso=date_iso,
        snapshot_iso=snapshot_iso,
        kalshi_public_base=str(args.kalshi_public_base),
        kalshi_status=str(args.kalshi_status),
    )
    if not rows:
        print(f"[warn] no matchups found for {args.league} on {date_iso}")
        return

    if args.apply:
        args.write_snapshot = True

    if args.write_snapshot:
        if out_path.exists():
            raise SystemExit(f"[error] snapshot already exists: {out_path}")
        write_snapshot_csv(out_path, rows)
        print(f"[ok] wrote snapshot: {out_path} rows={len(rows)}")
    else:
        print(f"[dry-run] snapshot would write: {out_path} rows={len(rows)}")

    added, filled = apply_fills_to_daily_ledger(
        league=args.league,
        date_iso=date_iso,
        fills=fills,
        apply=bool(args.apply),
        force=bool(args.force),
    )
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] daily ledger rows_added={added} cells_filled={filled}")


if __name__ == "__main__":
    main()
