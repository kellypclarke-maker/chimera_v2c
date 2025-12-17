"""Build the append-only master ledger from canonical sources.

Sources (priority order for filling blanks):
1) Daily ledgers under reports/daily_ledgers/
2) Daily ledger snapshots (reports/daily_ledgers/snapshots/*.bak)
3) Archived ledgers: game_level_ml_master + league-specific CSVs
4) Plan logs with captured kalshi_mid (reports/execution_logs/*.json)

Only appends or fills blank cells; refuses to overwrite non-blank values.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from chimera_v2c.lib.team_mapper import normalize_team_code
from chimera_v2c.src.ledger.formatting import (
    DAILY_LEDGER_COLUMNS,
    DAILY_LEDGER_PROB_COLUMNS,
    MISSING_SENTINEL,
    format_prob_cell,
)
from chimera_v2c.src.ledger.guard import (
    LedgerGuardError,
    compute_append_only_diff,
    load_csv_records,
    load_locked_dates,
    snapshot_file,
    write_csv,
)
from chimera_v2c.src.ledger.outcomes import format_final_score, is_placeholder_outcome

FIELDNAMES = [
    # Intentionally identical to the canonical per-day daily-ledger schema.
    *DAILY_LEDGER_COLUMNS,
]

DAILY_DIR = Path("reports/daily_ledgers")
DAILY_SNAPSHOT_DIR = DAILY_DIR / "snapshots"
LOCK_DIR = DAILY_DIR / "locked"
ARCHIVE_FILES = [
    # Historical master (legacy, read-only) used for pre-2025-12-04 backfills.
    Path("reports/specialist_performance/game_level_ml_master.csv"),
    # Older environments may still store archived masters under archive/.
    Path("archive/specialist_performance/archive_old_ledgers/game_level_ml_master.csv"),
    Path("archive/specialist_performance/archive_old_ledgers/ledger_nba_2025-12-05_to_2025-12-07.csv"),
    Path("archive/specialist_performance/archive_old_ledgers/ledger_nhl_2025-12-05_to_2025-12-07.csv"),
    Path("archive/specialist_performance/archive_old_ledgers/ledger_nfl_2025-12-05_to_2025-12-07.csv"),
]
# Archive rows to ignore entirely (bad/misparsed legacy rows)
SKIP_ARCHIVE_KEYS = {
    ("2025-12-07", "WAS@MIN"),  # NHL CBJ@WSH mis-labeled
    ("2025-12-05", "BKN@TOR"),  # game did not occur
    ("2025-11-22", "WSH@TOR"),  # misdated; canonical is 2025-11-21 WAS@TOR
    ("2025-11-22", "WAS@TOR"),  # misdated; canonical is 2025-11-21 WAS@TOR
    ("2025-11-20", "CHA@IND"),  # duplicate/misdated row (correct is 2025-11-19)
    ("2025-11-20", "TOR@PHI"),  # misdated (correct is 2025-11-19)
    ("2025-11-21", "BUF@HOU"),  # misdated (correct is 2025-11-20 BUF@HOU)
    ("2025-11-21", "PHI@MIL"),  # misdated (correct is 2025-11-20 PHI@MIL)
    ("2025-11-21", "SAC@MEM"),  # misdated (correct is 2025-11-20 SAC@MEM)
    ("2025-11-22", "BKN@BOS"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "DEN@HOU"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "IND@CLE"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "MIA@CHI"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "MIN@PHX"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "NOP@DAL"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "OKC@UTA"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "CAR@WPG"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "CHI@BUF"),  # misdated (correct is 2025-11-21)
    ("2025-11-22", "MIN@PIT"),  # misdated (correct is 2025-11-21)
    ("2025-11-23", "LV@CLE"),  # matchup reversed (correct is 2025-11-23 CLE@LV)
    ("2025-11-23", "ATL@NOP"),  # misdated (correct is 2025-11-22 ATL@NOP)
    ("2025-11-23", "DET@MIL"),  # misdated (correct is 2025-11-22 DET@MIL)
    ("2025-11-23", "EDM@FLA"),  # misdated (correct is 2025-11-22 EDM@FLA)
    ("2025-11-24", "LAL@UTA"),  # misdated (correct is 2025-11-23 LAL@UTA)
    ("2025-11-30", "NOP@MIA"),  # phantom (NO@MIA exists; NOP@MIA does not)
    ("2025-12-04", "DEN@PHI"),  # phantom (no scheduled game)
}
EXECUTION_LOG_DIR = Path("reports/execution_logs")
MASTER_PATH = Path("reports/master_ledger/master_game_ledger.csv")
MASTER_SNAPSHOT_DIR = MASTER_PATH.parent / "snapshots"
BY_LEAGUE_DIR = MASTER_PATH.parent / "by_league"

# Known alias fixes from archived data
MATCHUP_ALIASES: Dict[Tuple[str, str], Dict[str, str]] = {
    ("2025-12-07", "CBJ@WAS"): {"matchup": "CBJ@WSH", "league": "nhl", "home_team": "WSH", "away_team": "CBJ"},
    ("2025-12-07", "NOP@TB"): {"matchup": "NO@TB", "league": "nfl", "home_team": "TB", "away_team": "NO"},
    ("2025-12-06", "ATL@WSH"): {"matchup": "ATL@WAS", "league": "nba", "home_team": "WAS", "away_team": "ATL"},
    ("2025-12-06", "NJ@BOS"): {"matchup": "NJD@BOS", "league": "nhl", "home_team": "BOS", "away_team": "NJD"},
    ("2025-12-06", "NOP@BRK"): {"matchup": "NOP@BKN", "league": "nba", "home_team": "BKN", "away_team": "NOP"},
    ("2025-12-06", "NYI@TB"): {"matchup": "NYI@TBL", "league": "nhl", "home_team": "TBL", "away_team": "NYI"},
    ("2025-12-07", "SJ@CAR"): {"matchup": "SJS@CAR", "league": "nhl", "home_team": "CAR", "away_team": "SJS"},
}

NUMERIC_FIELDS = {
    *DAILY_LEDGER_PROB_COLUMNS,
}

KEY_FIELDS = ["date", "matchup"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the master ledger from canonical sources.")
    parser.add_argument(
        "--start-date",
        help="ISO start date filter (inclusive). If omitted, include all available.",
    )
    parser.add_argument(
        "--end-date",
        help="ISO end date filter (inclusive). If omitted, include all available.",
    )
    parser.add_argument(
        "--allow-overwrite-locked",
        action="store_true",
        help=(
            "Allow overwriting existing master probability cells for keys present in daily ledgers "
            "(typically locked dates; explicit human intent only)."
        ),
    )
    args = parser.parse_args()

    records: Dict[Tuple[str, str], MutableMapping[str, str]] = {}
    daily_keys: set[Tuple[str, str]] = set()
    locked_tokens = load_locked_dates(LOCK_DIR)
    locked_iso_dates = {
        f"{t[:4]}-{t[4:6]}-{t[6:8]}" for t in locked_tokens if isinstance(t, str) and len(t) == 8 and t.isdigit()
    }

    def canonicalize_matchup(matchup: str, league: str | None) -> str:
        if not matchup or "@" not in matchup:
            return matchup
        away_raw, home_raw = matchup.split("@", 1)
        league_norm = (league or "").lower()
        away_norm = normalize_team_code(away_raw, league_norm) if league_norm else None
        home_norm = normalize_team_code(home_raw, league_norm) if league_norm else None
        if away_norm and home_norm:
            return f"{away_norm}@{home_norm}"
        return matchup

    def ensure_record(date_str: str, matchup: str) -> MutableMapping[str, str]:
        key = (date_str, matchup)
        if key not in records:
            rec = {field: "" for field in FIELDNAMES}
            rec["date"] = date_str
            rec["matchup"] = matchup
            records[key] = rec
        return records[key]

    def normalize_numeric_fields(rows: List[MutableMapping[str, str]]) -> List[MutableMapping[str, str]]:
        for r in rows:
            for field in NUMERIC_FIELDS:
                val = r.get(field, "")
                if val in ("", None):
                    continue
                formatted = format_prob_cell(val, decimals=2, drop_leading_zero=True)
                r[field] = formatted or str(val).strip()
        return rows

    def fill(rec: MutableMapping[str, str], field: str, value: str) -> None:
        if value in (None, ""):
            return
        if rec.get(field) not in ("", None):
            return
        rec[field] = str(value)

    def blankish(value: object) -> bool:
        s = str(value).strip() if value is not None else ""
        if not s:
            return True
        return s.upper() == MISSING_SENTINEL

    def _clean_cell(value: object) -> str:
        s = str(value).strip() if value is not None else ""
        if not s or s.lower() in {"nan", "none", "na"}:
            return ""
        return s

    def ingest_daily_file(path: Path, *, allow_new_keys: bool) -> None:
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row.get("date", "") or ""
                raw_matchup = row.get("matchup") or row.get("game_id") or ""
                league = row.get("league", "")
                matchup = canonicalize_matchup(raw_matchup, league)
                if not matchup or not date_str:
                    continue
                if (date_str, matchup) in SKIP_ARCHIVE_KEYS:
                    continue
                if not allow_new_keys and (date_str, matchup) not in records:
                    continue
                rec = ensure_record(date_str, matchup)
                fill(rec, "league", league)
                fill(rec, "v2c", row.get("v2c", ""))
                fill(rec, "grok", row.get("grok", ""))
                fill(rec, "gemini", row.get("gemini", ""))
                fill(rec, "gpt", row.get("gpt", ""))
                fill(rec, "kalshi_mid", row.get("kalshi_mid", ""))
                fill(rec, "market_proxy", row.get("market_proxy", ""))
                fill(rec, "moneypuck", row.get("moneypuck", ""))
                fill(rec, "actual_outcome", row.get("actual_outcome", ""))

    def overlay_daily_file(path: Path) -> None:
        """
        Daily ledgers are canonical. After any other sources are ingested, we
        overwrite master fields for keys present in daily ledgers so the master
        matches the per-day files exactly (including blanks).
        """
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = _clean_cell(row.get("date", ""))
                raw_matchup = _clean_cell(row.get("matchup") or row.get("game_id") or "")
                league = _clean_cell(row.get("league", ""))
                matchup = canonicalize_matchup(raw_matchup, league)
                if not matchup or not date_str:
                    continue
                if (date_str, matchup) in SKIP_ARCHIVE_KEYS:
                    continue
                rec = ensure_record(date_str, matchup)
                rec["league"] = league
                rec["v2c"] = _clean_cell(row.get("v2c", ""))
                rec["grok"] = _clean_cell(row.get("grok", ""))
                rec["gemini"] = _clean_cell(row.get("gemini", ""))
                rec["gpt"] = _clean_cell(row.get("gpt", ""))
                rec["kalshi_mid"] = _clean_cell(row.get("kalshi_mid", ""))
                rec["market_proxy"] = _clean_cell(row.get("market_proxy", ""))
                rec["moneypuck"] = _clean_cell(row.get("moneypuck", ""))
                rec["actual_outcome"] = _clean_cell(row.get("actual_outcome", ""))

    def normalize_from_archive(row: Mapping[str, str]) -> Tuple[str, str, str, str]:
        """Return matchup, league, home_team, away_team with alias corrections."""
        date_str = row.get("date", "") or ""
        raw_matchup = (
            row.get("game_id")
            or (
                row.get("away_team")
                and row.get("home_team")
                and f"{row['away_team']}@{row['home_team']}"
            )
            or ""
        )
        alias = MATCHUP_ALIASES.get((date_str, raw_matchup))
        matchup = alias.get("matchup", raw_matchup) if alias else raw_matchup
        league = alias.get("league") if alias else row.get("league", "")
        home_team = alias.get("home_team") if alias else row.get("home_team", "")
        away_team = alias.get("away_team") if alias else row.get("away_team", "")
        return matchup, league, home_team, away_team

    def outcome_from_scores(home_team: str, away_team: str, row: Mapping[str, str]) -> str:
        home_score = row.get("home_score", "")
        away_score = row.get("away_score", "")
        if all(str(x).strip() not in ("", "None") for x in (home_score, away_score, home_team, away_team)):
            try:
                return format_final_score(
                    away_team,
                    home_team,
                    float(str(away_score).strip()),
                    float(str(home_score).strip()),
                )
            except (TypeError, ValueError):
                return ""
        return ""

    def ingest_archive_file(path: Path) -> None:
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date_str = row.get("date", "") or ""
                matchup, league, home_team, away_team = normalize_from_archive(row)
                matchup = canonicalize_matchup(matchup, league)
                if not matchup or not date_str:
                    continue
                if (date_str, matchup) in SKIP_ARCHIVE_KEYS:
                    continue
                if (date_str, matchup) in daily_keys:
                    continue
                rec = ensure_record(date_str, matchup)
                fill(rec, "league", league)
                fill(rec, "v2c", row.get("p_home_v2c", ""))
                fill(rec, "grok", row.get("p_home_grok", ""))
                fill(rec, "gemini", row.get("p_home_gemini", ""))
                fill(rec, "gpt", row.get("p_home_gpt", ""))
                # Legacy archive uses p_home_market for the market mid.
                fill(rec, "kalshi_mid", row.get("kalshi_mid", "") or row.get("p_home_market", ""))
                fill(rec, "market_proxy", row.get("market_proxy", ""))
                # MoneyPuck is NHL-only; non-NHL rows use NR to avoid blanks.
                if str(league).strip().lower() != "nhl" and not str(rec.get("moneypuck", "")).strip():
                    rec["moneypuck"] = MISSING_SENTINEL
                outcome = outcome_from_scores(home_team, away_team, row)
                fill(rec, "actual_outcome", outcome)

    def ingest_execution_log(path: Path) -> None:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return
        plans = []
        if isinstance(data, list):
            plans = data
        elif isinstance(data, dict):
            if isinstance(data.get("plans"), list):
                plans = data["plans"]
            elif isinstance(data.get("entries"), list):
                plans = data["entries"]
        for entry in plans:
            matchup = entry.get("key") or ""
            market = entry.get("market") or {}
            components = entry.get("components") or {}
            orders = entry.get("planned_orders") or []
            date_str = ""
            league = ""
            if orders:
                date_str = orders[0].get("date", "")
                league = orders[0].get("league", "")
            if not date_str:
                date_str = entry.get("date", "")
            if not matchup or not date_str:
                continue
            matchup = canonicalize_matchup(matchup, league)
            if (date_str, matchup) in daily_keys:
                continue
            rec = ensure_record(date_str, matchup)
            fill(rec, "league", league)
            # Only trust kalshi_mid from execution logs to avoid mixing model variants.
            mid = components.get("kalshi_mid")
            if mid is None and market:
                bid = market.get("yes_bid")
                ask = market.get("yes_ask")
                if bid is not None and ask is not None:
                    mid = (float(bid) + float(ask)) / 200.0
            if mid is not None:
                fill(rec, "kalshi_mid", mid)

    # 1) Daily ledgers
    for daily_file in sorted(DAILY_DIR.glob("*_daily_game_ledger.csv")):
        ingest_daily_file(daily_file, allow_new_keys=True)
    daily_keys = set(records)

    # 2) Snapshots
    for snapshot_path in sorted(DAILY_SNAPSHOT_DIR.glob("*.bak")):
        # Snapshots are a fill-only source. They should not introduce new keys that
        # are not present in the current daily ledgers.
        ingest_daily_file(snapshot_path, allow_new_keys=False)

    # 3) Archive ledgers
    for archive_file in ARCHIVE_FILES:
        ingest_archive_file(archive_file)

    # 4) Execution logs (kalshi_mid)
    for log_file in EXECUTION_LOG_DIR.glob("*.json"):
        ingest_execution_log(log_file)

    # Final overlay: daily ledgers are authoritative for any keys they contain.
    for daily_file in sorted(DAILY_DIR.glob("*_daily_game_ledger.csv")):
        overlay_daily_file(daily_file)

    filtered_records = [
        rec
        for (date_str, _), rec in records.items()
        if _within_range(date_str, args.start_date, args.end_date)
    ]

    new_rows = sorted(filtered_records, key=lambda r: (r["date"], r.get("league", ""), r["matchup"]))

    def fill_missing_sentinels(rows: List[MutableMapping[str, str]]) -> List[MutableMapping[str, str]]:
        for row in rows:
            league = str(row.get("league", "")).strip().lower()
            if league != "nhl" and not str(row.get("moneypuck", "")).strip():
                row["moneypuck"] = MISSING_SENTINEL
            for field in FIELDNAMES:
                if field in ("date", "league", "matchup"):
                    if not str(row.get(field, "")).strip():
                        raise SystemExit(f"[error] master ledger row missing required field '{field}': {row}")
                    continue
                if field == "actual_outcome":
                    # Outcomes may be blank until games are final.
                    continue
                if not str(row.get(field, "")).strip():
                    row[field] = MISSING_SENTINEL
        return rows

    # Guard against overwriting existing master
    old_rows_raw = load_csv_records(MASTER_PATH)
    if old_rows_raw:
        # Canonicalize old rows to match new keying
        old_canon_map: Dict[Tuple[str, str], MutableMapping[str, str]] = {}
        field_aliases: Dict[str, Tuple[str, ...]] = {
            "v2c": ("p_home_v2c",),
            "grok": ("p_home_grok",),
            "gemini": ("p_home_gemini",),
            "gpt": ("p_home_gpt",),
            "kalshi_mid": ("p_home_market",),
        }
        for r in old_rows_raw:
            date_str = r.get("date", "")
            league = r.get("league", "")
            matchup_canon = canonicalize_matchup(r.get("matchup", ""), league)
            key = (date_str, matchup_canon)
            if key not in old_canon_map:
                old_canon_map[key] = {field: "" for field in FIELDNAMES}
                old_canon_map[key]["date"] = date_str
                old_canon_map[key]["matchup"] = matchup_canon
            for field in FIELDNAMES[1:]:
                val = ""
                for alias in (field, *field_aliases.get(field, ())):
                    candidate = r.get(alias, "")
                    if candidate not in ("", None):
                        val = candidate
                        break
                if val not in ("", None) and not old_canon_map[key].get(field):
                    old_canon_map[key][field] = val
        old_rows = list(old_canon_map.values())

        old_map = {(r.get("date", ""), r.get("matchup", "")): r for r in old_rows}
        for r in new_rows:
            key = (r.get("date", ""), r.get("matchup", ""))
            if key in old_map:
                old_r = old_map[key]
                for field in [f for f in FIELDNAMES if f not in KEY_FIELDS]:
                    new_val = r.get(field, "")
                    old_val = old_r.get(field, "")
                    # Fill missing values from old master for non-daily keys; keep new values on conflict.
                    if blankish(new_val) and not blankish(old_val) and key not in daily_keys:
                        r[field] = str(old_val)
        # Drop any previously written rows that are explicitly skipped so regeneration can remove them.
        old_rows = [
            r for r in old_rows if (r.get("date", ""), r.get("matchup", "")) not in SKIP_ARCHIVE_KEYS
        ]

        def _normalize_blank(value: object) -> str:
            s = str(value).strip() if value is not None else ""
            if not s:
                return ""
            if s.upper() == MISSING_SENTINEL:
                return ""
            return s

        fill_missing_sentinels(new_rows)
        old_rows = normalize_numeric_fields(old_rows)
        new_rows = normalize_numeric_fields(new_rows)

        # We allow `actual_outcome` to change (it is derived and can be corrected),
        # but never allow it to be cleared unless the old value was an obvious placeholder.
        old_map_out = {(r.get("date", ""), r.get("matchup", "")): r for r in old_rows}
        new_map_out = {(r.get("date", ""), r.get("matchup", "")): r for r in new_rows}
        for key in set(old_map_out) & set(new_map_out):
            old_out = _normalize_blank(old_map_out[key].get("actual_outcome", ""))
            new_out = _normalize_blank(new_map_out[key].get("actual_outcome", ""))
            if old_out and not new_out and not is_placeholder_outcome(old_out):
                raise LedgerGuardError(
                    f"Append-only violation for {key}: actual_outcome lost value '{old_out}'."
                )

        if args.allow_overwrite_locked:
            allow_overwrite_keys = set(daily_keys)
            if allow_overwrite_keys:
                new_map = {(r.get("date", ""), r.get("matchup", "")): r for r in new_rows}
                value_fields = [f for f in FIELDNAMES if f not in KEY_FIELDS and f != "actual_outcome"]
                for old_r in old_rows:
                    key = (old_r.get("date", ""), old_r.get("matchup", ""))
                    if key not in allow_overwrite_keys:
                        continue
                    new_r = new_map.get(key)
                    if not new_r:
                        continue
                    for field in value_fields:
                        old_val = _normalize_blank(old_r.get(field, ""))
                        new_val = _normalize_blank(new_r.get(field, ""))
                        if old_val and new_val and old_val != new_val:
                            # Permit explicit corrections on locked dates by treating the
                            # prior value as blank for diff purposes (but still disallow
                            # clearing values to blank/NR).
                            old_r[field] = ""

        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=KEY_FIELDS,
            value_fields=[f for f in FIELDNAMES if f not in KEY_FIELDS and f != "actual_outcome"],
            blank_sentinels={MISSING_SENTINEL},
        )
    else:
        fill_missing_sentinels(new_rows)

    if MASTER_PATH.exists():
        snapshot_file(MASTER_PATH, MASTER_SNAPSHOT_DIR)

    MASTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    normalize_numeric_fields(new_rows)
    write_csv(MASTER_PATH, new_rows, fieldnames=FIELDNAMES)

    print(f"Wrote {len(new_rows)} rows to {MASTER_PATH}")
    _write_by_league_ledgers(new_rows)
    _summarize_missing(new_rows)


def _within_range(date_str: str, start: str | None, end: str | None) -> bool:
    if not start and not end:
        return True
    dt = datetime.strptime(date_str, "%Y-%m-%d").date()
    if start:
        if dt < datetime.strptime(start, "%Y-%m-%d").date():
            return False
    if end:
        if dt > datetime.strptime(end, "%Y-%m-%d").date():
            return False
    return True


def _summarize_missing(rows: Sequence[Mapping[str, str]]) -> None:
    total = len(rows)
    missing = {field: 0 for field in FIELDNAMES}
    for row in rows:
        for field in FIELDNAMES:
            if not str(row.get(field, "")).strip():
                missing[field] += 1
    print(f"Total rows: {total}")
    for field in FIELDNAMES:
        print(f"{field}: missing {missing[field]}")


def _write_by_league_ledgers(rows: Sequence[Mapping[str, str]]) -> None:
    """Write analysis-friendly by-league ledgers (derived; not canonical)."""
    BY_LEAGUE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_dir = MASTER_SNAPSHOT_DIR / "by_league"

    def league_norm(row: Mapping[str, str]) -> str:
        return str(row.get("league", "")).strip().lower()

    for league in ("nba", "nhl", "nfl"):
        league_rows = [r for r in rows if league_norm(r) == league]
        if not league_rows:
            continue
        out_path = BY_LEAGUE_DIR / f"{league}_game_ledger.csv"
        out_fieldnames = FIELDNAMES if league == "nhl" else [c for c in FIELDNAMES if c != "moneypuck"]
        if out_path.exists():
            snapshot_file(out_path, snapshot_dir)
        write_csv(out_path, league_rows, fieldnames=out_fieldnames)
        print(f"Wrote {len(league_rows)} rows to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except LedgerGuardError as exc:
        raise SystemExit(f"Refusing to write master ledger: {exc}")
