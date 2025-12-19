#!/usr/bin/env python
"""
Export Kalshi private-account fills (executed trades) to a local CSV (read-only).

This is used to reconstruct what was *actually* filled when an operator traded manually,
so we can compare realized execution vs planned assumptions (taker @ ask + slippage).

Requires private creds:
  - KALSHI_API_KEY_ID
  - Either KALSHI_API_PRIVATE_KEY (PEM string) or KALSHI_PRIVATE_KEY_PATH (path to PEM file)

Note: config/env.list is loaded via load_env_from_env_list(), so ensure that file points to
the correct key path on this machine.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from zoneinfo import ZoneInfo

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib import kalshi_utils


def _parse_date_iso(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"[error] invalid --date (expected YYYY-MM-DD): {text}") from exc


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    s = (ts or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _safe_get(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("[error] no fills matched the requested filters")
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export Kalshi fills to CSV (private API; read-only).")
    ap.add_argument("--date", default="", help="YYYY-MM-DD (local date filter; see --tz).")
    ap.add_argument("--start-date", default="", help="YYYY-MM-DD inclusive (local date filter; see --tz).")
    ap.add_argument("--end-date", default="", help="YYYY-MM-DD inclusive (local date filter; see --tz).")
    ap.add_argument(
        "--tz",
        default="America/Los_Angeles",
        help="Timezone for interpreting --date/--start-date/--end-date (default: America/Los_Angeles).",
    )
    ap.add_argument("--limit", type=int, default=500, help="Max fills to fetch from API (default: 500).")
    ap.add_argument(
        "--series-prefix",
        default="",
        help="Optional ticker prefix filter (e.g. KXNBAGAME or KXNHLGAME).",
    )
    ap.add_argument("--ticker", default="", help="Optional exact ticker filter.")
    ap.add_argument("--out", default="", help="Output CSV path (default under reports/execution_logs/kalshi_fills/YYYYMMDD/).")
    ap.add_argument("--dump-json", action="store_true", help="Also dump raw response JSON next to the CSV (debug).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_env_from_env_list()

    try:
        tz = ZoneInfo(str(args.tz))
    except Exception as exc:
        raise SystemExit(f"[error] invalid --tz: {args.tz}") from exc

    if str(args.date).strip():
        start_date = _parse_date_iso(args.date)
        end_date = start_date
    else:
        if not str(args.start_date).strip() or not str(args.end_date).strip():
            raise SystemExit("[error] pass either --date or both --start-date and --end-date")
        start_date = _parse_date_iso(args.start_date)
        end_date = _parse_date_iso(args.end_date)
        if end_date < start_date:
            raise SystemExit("[error] --end-date must be >= --start-date")

    limit = int(args.limit)
    if limit < 1 or limit > 10_000:
        raise SystemExit("[error] --limit must be in [1, 10000]")

    if not kalshi_utils.has_private_creds():
        raise SystemExit(
            "[error] Kalshi private creds not configured (need KALSHI_API_KEY_ID + private key). "
            "Check config/env.list and ensure KALSHI_PRIVATE_KEY_PATH points to a real .pem file."
        )

    try:
        resp = kalshi_utils.get_fills(ticker=(args.ticker.strip() or None), limit=limit)
    except Exception as exc:
        raise SystemExit(
            f"[error] failed to fetch fills via private API: {exc}\n"
            "If config/env.list sets KALSHI_PRIVATE_KEY_PATH, ensure the file exists at that path."
        ) from exc

    fills = resp.get("fills") or []
    if not isinstance(fills, list):
        raise SystemExit("[error] unexpected fills payload shape (missing fills list)")

    series_prefix = (args.series_prefix or "").strip().upper()
    ticker_filter = (args.ticker or "").strip().upper()

    rows: List[Dict[str, object]] = []
    for f in fills:
        if not isinstance(f, dict):
            continue
        created = _parse_iso_utc(str(_safe_get(f, "created_time", "ts", "time") or ""))
        if created is None:
            continue
        created_local = created.astimezone(tz)
        if not (start_date <= created_local.date() <= end_date):
            continue
        ticker = str(_safe_get(f, "ticker") or "").strip().upper()
        if ticker_filter and ticker != ticker_filter:
            continue
        if series_prefix and not ticker.startswith(series_prefix):
            continue

        side = str(_safe_get(f, "side") or "").strip().lower()
        action = str(_safe_get(f, "action") or "").strip().lower()
        is_taker = _safe_get(f, "is_taker")
        count = _safe_get(f, "count", "fill_count", "filled_count")

        # Kalshi fills often include:
        #   price (dollars float), yes_price/no_price (cents int)
        yes_price = _safe_get(f, "yes_price", "yes_price_fixed")
        no_price = _safe_get(f, "no_price", "no_price_fixed")
        price_any = _safe_get(f, "price")
        price_cents: Optional[int] = None
        if side == "yes" and yes_price is not None:
            price_cents = int(yes_price)
        elif side == "no" and no_price is not None:
            price_cents = int(no_price)
        elif price_any is not None:
            try:
                p = float(price_any)
            except Exception:
                p = None
            if p is not None:
                # If it's already cents (e.g. 64), keep; if it's dollars (e.g. 0.64), convert.
                price_cents = int(round(p * 100.0)) if p <= 1.0 else int(round(p))

        rows.append(
            {
                "created_time_utc": created.isoformat().replace("+00:00", "Z"),
                "date_utc": created.date().isoformat(),
                "created_time_local": created_local.isoformat(),
                "date_local": created_local.date().isoformat(),
                "tz": str(args.tz),
                "ticker": ticker,
                "action": action,
                "side": side,
                "is_taker": "" if is_taker is None else bool(is_taker),
                "count": "" if count is None else int(count),
                "price_cents": "" if price_cents is None else int(price_cents),
                "yes_price_cents": "" if yes_price is None else int(yes_price),
                "no_price_cents": "" if no_price is None else int(no_price),
                "order_id": str(_safe_get(f, "order_id") or ""),
                "trade_id": str(_safe_get(f, "trade_id") or ""),
            }
        )

    rows.sort(key=lambda r: (r["created_time_utc"], r["ticker"], r["side"]))

    out_path: Path
    if args.out:
        out_path = Path(str(args.out))
    else:
        tag = f"{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}" if start_date != end_date else start_date.strftime("%Y%m%d")
        out_path = Path("reports/execution_logs/kalshi_fills") / end_date.strftime("%Y%m%d") / f"kalshi_fills_{tag}.csv"

    _write_csv(out_path, rows)
    print(f"[ok] wrote {len(rows)} fills -> {out_path}")

    if args.dump_json:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(resp, indent=2), encoding="utf-8")
        print(f"[ok] wrote raw JSON -> {json_path}")


if __name__ == "__main__":
    main()
