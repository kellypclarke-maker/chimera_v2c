#!/usr/bin/env python
"""
Read-only validator to ensure team codes in ledgers and canonical reports are normalized.

Checks:
- Daily ledgers under reports/daily_ledgers/*_daily_game_ledger.csv
  * Valid league
  * matchup parses as AWAY@HOME
  * both sides normalize via team_mapper for that league
- Canonical specialist reports under reports/specialist_reports/{NBA,NHL,NFL}/**/*.txt
  * Game line codes normalize for the league

Outputs a summary to stdout; no files are modified.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from chimera_v2c.lib.team_mapper import normalize_team_code


DAILY_DIR = Path("reports/daily_ledgers")
CANONICAL_ROOTS = [Path("reports/specialist_reports/NBA"), Path("reports/specialist_reports/NHL"), Path("reports/specialist_reports/NFL")]


def normalize_team(team: str, league: str) -> str | None:
    return normalize_team_code(team, league.lower()) if team else None


def check_daily_ledgers() -> Tuple[int, List[str]]:
    issues: List[str] = []
    files = sorted(DAILY_DIR.glob("*_daily_game_ledger.csv"))
    for path in files:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            issues.append(f"{path.name}: failed to read ({exc})")
            continue
        for idx, row in df.iterrows():
            league = str(row.get("league", "")).lower()
            matchup = row.get("matchup")
            if league not in ("nba", "nhl", "nfl"):
                issues.append(f"{path.name}:{idx}: invalid league '{league}'")
                continue
            if not isinstance(matchup, str) or "@" not in matchup:
                issues.append(f"{path.name}:{idx}: invalid matchup '{matchup}'")
                continue
            away_raw, home_raw = matchup.split("@", 1)
            away_norm = normalize_team(away_raw, league)
            home_norm = normalize_team(home_raw, league)
            if not away_norm or not home_norm:
                issues.append(f"{path.name}:{idx}: unnormalized teams away='{away_raw}' home='{home_raw}'")
                continue
            if away_norm != away_raw or home_norm != home_raw:
                issues.append(
                    f"{path.name}:{idx}: noncanonical matchup '{matchup}' -> '{away_norm}@{home_norm}'"
                )
    return len(files), issues


GAME_LINE_RE = re.compile(
    r"Game:\s*(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+([A-Z0-9]+)@([A-Z0-9]+)",
    re.IGNORECASE,
)


def scan_canonical_reports() -> Tuple[int, List[str]]:
    files_checked = 0
    issues: List[str] = []
    for root in CANONICAL_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            files_checked += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                issues.append(f"{path}: read error ({exc})")
                continue
            m = GAME_LINE_RE.search(text)
            if not m:
                issues.append(f"{path}: missing/invalid Game line")
                continue
            _, league_raw, away_raw, home_raw = m.groups()
            league = league_raw.lower()
            away_norm = normalize_team(away_raw, league)
            home_norm = normalize_team(home_raw, league)
            if not away_norm or not home_norm:
                issues.append(f"{path}: unnormalized teams away='{away_raw}' home='{home_raw}'")
                continue
            if away_norm != away_raw or home_norm != home_raw:
                issues.append(f"{path}: noncanonical teams '{away_raw}@{home_raw}' -> '{away_norm}@{home_norm}'")
    return files_checked, issues


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only check for normalized team codes in ledgers and canonical reports.")
    ap.add_argument(
        "--skip-canonical",
        action="store_true",
        help="Skip scanning canonical specialist reports.",
    )
    args = ap.parse_args()

    daily_count, daily_issues = check_daily_ledgers()
    canonical_count, canonical_issues = (0, [])
    if not args.skip_canonical:
        canonical_count, canonical_issues = scan_canonical_reports()

    print(f"Checked {daily_count} daily ledger files.")
    if daily_issues:
        print(f"Daily ledger issues ({len(daily_issues)}):")
        for msg in daily_issues:
            print(f"  - {msg}")
    else:
        print("Daily ledgers: all matchups normalized.")

    if not args.skip_canonical:
        print(f"\nChecked {canonical_count} canonical reports.")
        if canonical_issues:
            print(f"Canonical issues ({len(canonical_issues)}):")
            for msg in canonical_issues:
                print(f"  - {msg}")
        else:
            print("Canonical reports: all Game lines normalized.")

    if daily_issues or canonical_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
