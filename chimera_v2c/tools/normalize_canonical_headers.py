#!/usr/bin/env python
"""
Normalize HELIOS headers in canonical specialist reports:
- Ensure Game line uses canonical team codes via team_mapper.
- Ensure Winner line uses canonical team codes.
- Ensure p_home is present when safely derivable (prefer existing p_home; else derive from Winner + p_true).

Append-only on meaning: rewrites headers in-place; does not touch daily ledgers.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

from chimera_v2c.lib.team_mapper import normalize_team_code


GAME_RE = re.compile(r"Game:\s*(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+([A-Z0-9]+)@([A-Z0-9]+)", re.IGNORECASE)
WINNER_RE = re.compile(r"Winner:\s*([A-Z0-9]+)", re.IGNORECASE)
PHOME_RE = re.compile(r"p_home:\s*([0-9.]+)", re.IGNORECASE)
PTRUE_RE = re.compile(r"p_true:\s*([0-9.]+)", re.IGNORECASE)
BODY_PROB_RE = re.compile(
    r"(p[_ ]?(true|home|prob|win|moneyline)|probability|home win)\s*[:=]?\s*([0-9.]+%?)",
    re.IGNORECASE,
)

ROOTS = [Path("reports/specialist_reports/NBA"), Path("reports/specialist_reports/NHL"), Path("reports/specialist_reports/NFL")]


def find_body_ptrue(text: str) -> Optional[float]:
    for m in BODY_PROB_RE.finditer(text):
        val_str = m.group(3)
        if val_str.endswith("%"):
            try:
                v = float(val_str.rstrip("%")) / 100.0
            except ValueError:
                continue
        else:
            try:
                v = float(val_str)
            except ValueError:
                continue
        if 0 <= v <= 1:
            return round(v, 4)
    return None


def canonicalize_game(line: str) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    m = GAME_RE.search(line)
    if not m:
        return line, None, None, None, None
    date, league_raw, away_raw, home_raw = m.groups()
    league = league_raw.lower()
    away = normalize_team_code(away_raw, league) or away_raw
    home = normalize_team_code(home_raw, league) or home_raw
    new_line = f"Game: {date} {league_raw.upper()} {away}@{home}"
    return new_line, date, league, away, home


def normalize_file(path: Path, apply: bool) -> Tuple[bool, bool, bool]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    changed_game = changed_ptrue = changed_winner = False
    changed_phome = False

    # Game line
    new_game_line, date, league, away, home = canonicalize_game(text)
    if date is None:
        return False, False, False
    text = GAME_RE.sub(new_game_line, text, count=1)
    if new_game_line not in path.read_text(encoding="utf-8", errors="ignore"):
        changed_game = True

    # Winner
    winner_match = WINNER_RE.search(text)
    winner = normalize_team_code(winner_match.group(1), league) if winner_match else None
    if winner and winner_match and winner != winner_match.group(1):
        text = text[: winner_match.start()] + f"Winner: {winner}" + text[winner_match.end() :]
        changed_winner = True

    # p_true
    header_p = None
    ptrue_match = PTRUE_RE.search(text)
    if ptrue_match:
        try:
            header_p = float(ptrue_match.group(1))
        except ValueError:
            header_p = None
    if header_p is None:
        header_p = find_body_ptrue(text)
        if header_p is not None:
            # insert/replace header p_true
            if ptrue_match:
                text = text[: ptrue_match.start()] + f"p_true: {header_p}" + text[ptrue_match.end() :]
            else:
                insert_pos = WINNER_RE.search(text).end() if WINNER_RE.search(text) else 0
                text = text[:insert_pos] + f"\np_true: {header_p}" + text[insert_pos:]
            changed_ptrue = True

    # p_home (prefer existing; otherwise derive from winner + p_true)
    p_home_match = PHOME_RE.search(text)
    existing_home = None
    if p_home_match:
        try:
            existing_home = float(p_home_match.group(1))
        except ValueError:
            existing_home = None

    header_home = existing_home
    if header_home is None and header_p is not None and winner and away and home:
        if winner == home:
            header_home = header_p
        elif winner == away:
            header_home = 1.0 - header_p

    if header_home is not None and 0 <= header_home <= 1:
        if p_home_match:
            # Only replace when the existing p_home is missing/unparsable.
            if existing_home is None:
                text = text[: p_home_match.start()] + f"p_home: {round(header_home, 4)}" + text[p_home_match.end() :]
                changed_phome = True
        else:
            insert_pos = WINNER_RE.search(text).end() if WINNER_RE.search(text) else 0
            text = text[:insert_pos] + f"\np_home: {round(header_home, 4)}" + text[insert_pos:]
            changed_phome = True

    if apply and (changed_game or changed_ptrue or changed_winner or changed_phome):
        path.write_text(text, encoding="utf-8")

    return changed_game, (changed_ptrue or changed_phome), changed_winner


def main() -> None:
    ap = argparse.ArgumentParser(description="Normalize canonical specialist HELIOS headers (teams, winner, p_true).")
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    args = ap.parse_args()

    cg = cp = cw = 0
    total = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            total += 1
            g, p, w = normalize_file(path, apply=args.apply)
            cg += 1 if g else 0
            cp += 1 if p else 0
            cw += 1 if w else 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] files checked={total}, game_line_changes={cg}, p_true_changes={cp}, winner_changes={cw}")


if __name__ == "__main__":
    main()
