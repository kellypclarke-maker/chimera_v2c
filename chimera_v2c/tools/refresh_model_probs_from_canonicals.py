#!/usr/bin/env python
"""
Scan canonical specialist reports and fill blank model probabilities in daily ledgers (append-only).

- Does NOT move or modify canonical files.
- Respects lockfiles unless --force.
- Determines model column from filename (grok/gemini/gpt) and header.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from chimera_v2c.lib.team_mapper import normalize_team_code
from chimera_v2c.src.ledger.formatting import MISSING_SENTINEL, format_prob_cell
from chimera_v2c.src.ledger.guard import compute_append_only_diff, load_locked_dates


CANON_ROOTS = [Path("reports/specialist_reports/NBA"), Path("reports/specialist_reports/NHL"), Path("reports/specialist_reports/NFL")]
DAILY_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_DIR / "locked"

GAME_RE = re.compile(r"Game:\s*(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+([A-Z0-9]+)@([A-Z0-9]+)", re.IGNORECASE)
WINNER_RE = re.compile(r"Winner:\s*([A-Z0-9]+)", re.IGNORECASE)
PHOME_RE = re.compile(r"p_home:\s*([0-9.]+)", re.IGNORECASE)
PTRUE_RE = re.compile(r"p_true:\s*([0-9.]+)", re.IGNORECASE)
PTRUE_RAW_RE = re.compile(r"p_true_raw:\s*([0-9.]+)", re.IGNORECASE)
PTRUE_CAL_RE = re.compile(r"p_true_calibrated:\s*([0-9.]+)", re.IGNORECASE)

MONEYLINE_BLOCK_RE = re.compile(
    r"Prediction_Moneyline:(.*?)(?:Prediction_Spread|Prediction_Total|HELIOS_PREDICTION_HEADER_END)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class Parsed:
    date: str
    league: str
    away: str
    home: str
    winner: Optional[str]
    p_true: Optional[float]
    p_home: Optional[float]
    model_label: str

    @property
    def matchup(self) -> str:
        return f"{self.away}@{self.home}"


def model_family(fname: str, header_model: str) -> Optional[str]:
    def detect(text: str) -> set[str]:
        if not text:
            return set()
        label = text.lower()
        families: set[str] = set()
        if "grok" in label:
            families.add("grok")
        if "gemini" in label:
            families.add("gemini")
        if "gpt" in label or "openai" in label:
            families.add("gpt")
        return families

    # Prefer a single unambiguous signal, but allow either filename or header to carry it.
    fams = detect(header_model) | detect(fname)
    if len(fams) == 1:
        return next(iter(fams))
    # If ambiguous, fall back to whichever source is unambiguous.
    header_fams = detect(header_model)
    if len(header_fams) == 1:
        return next(iter(header_fams))
    fname_fams = detect(fname)
    if len(fname_fams) == 1:
        return next(iter(fname_fams))
    return None


def parse_canonical(path: Path) -> Optional[Parsed]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = GAME_RE.search(text)
    if not m:
        return None
    date, league_raw, away_raw, home_raw = m.groups()
    league = league_raw.lower()
    away = normalize_team_code(away_raw, league) or away_raw
    home = normalize_team_code(home_raw, league) or home_raw
    w = WINNER_RE.search(text)
    winner = normalize_team_code(w.group(1), league) if w else None
    p_home = None
    ph = PHOME_RE.search(text)
    if ph:
        try:
            p_home = float(ph.group(1))
        except ValueError:
            p_home = None

    p_true = None
    moneyline_text = text
    moneyline_match = MONEYLINE_BLOCK_RE.search(text)
    if moneyline_match:
        moneyline_text = moneyline_match.group(1)

    for rx in (PTRUE_CAL_RE, PTRUE_RAW_RE, PTRUE_RE):
        m_true = rx.search(moneyline_text)
        if not m_true:
            continue
        try:
            p_true = float(m_true.group(1))
        except ValueError:
            p_true = None
        break
    header_model = ""
    model_line = re.search(r"Model:\s*(.+)", text, re.IGNORECASE)
    if model_line:
        header_model = model_line.group(1).strip()
    fam = model_family(path.name, header_model)
    if fam is None:
        return None
    return Parsed(date=date, league=league, away=away, home=home, winner=winner, p_true=p_true, p_home=p_home, model_label=fam)


def p_home_from(parsed: Parsed) -> Optional[float]:
    if parsed.p_home is not None:
        if 0 <= parsed.p_home <= 1:
            return parsed.p_home
        return None
    if parsed.p_true is None or not parsed.winner:
        return None
    if parsed.winner == parsed.home:
        return parsed.p_true
    if parsed.winner == parsed.away:
        return 1 - parsed.p_true
    return None


def update_ledger(parsed: Parsed, apply: bool, force: bool, overwrite_nr: bool) -> Tuple[bool, Path]:
    ymd = parsed.date.replace("-", "")
    ledger_path = DAILY_DIR / f"{ymd}_daily_game_ledger.csv"
    if not ledger_path.exists():
        return False, ledger_path
    locked = load_locked_dates(LOCK_DIR)
    if ymd in locked and not force:
        return False, ledger_path

    df = pd.read_csv(ledger_path).fillna("")
    if parsed.model_label not in df.columns:
        return False, ledger_path

    # Normalize daily matchups for matching
    def norm_matchup(val: str) -> str:
        if not isinstance(val, str) or "@" not in val:
            return val
        away_raw, home_raw = val.split("@", 1)
        away = normalize_team_code(away_raw, parsed.league) or away_raw
        home = normalize_team_code(home_raw, parsed.league) or home_raw
        return f"{away}@{home}"

    matchup_norm = df["matchup"].astype(str).apply(norm_matchup)
    mask = (matchup_norm == parsed.matchup) & (df["league"].astype(str).str.lower() == parsed.league)
    if not mask.any():
        return False, ledger_path
    row_idx = df.index[mask][0]
    existing = str(df.at[row_idx, parsed.model_label]).strip()
    if existing and not (overwrite_nr and existing.strip().upper() == MISSING_SENTINEL):
        return False, ledger_path

    p_home = p_home_from(parsed)
    if p_home is None:
        return False, ledger_path

    formatted = format_prob_cell(p_home, decimals=2, drop_leading_zero=True)
    if not formatted:
        return False, ledger_path
    df.at[row_idx, parsed.model_label] = formatted

    if apply:
        original = pd.read_csv(ledger_path).fillna("")
        old_rows = original.fillna("").astype(str).to_dict("records")
        new_rows = df.fillna("").astype(str).to_dict("records")
        compute_append_only_diff(
            old_rows=old_rows,
            new_rows=new_rows,
            key_fields=["date", "matchup"],
            value_fields=[c for c in df.columns if c not in ("date", "matchup")],
            blank_sentinels={MISSING_SENTINEL} if overwrite_nr else None,
        )
        df.to_csv(ledger_path, index=False)
    return True, ledger_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Fill blank model cells from canonical specialist reports (append-only).")
    ap.add_argument("--apply", action="store_true", help="Apply changes (default: dry-run).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked ledgers or overwrite non-blank cells.")
    ap.add_argument(
        "--overwrite-nr",
        action="store_true",
        help="Allow replacing existing NR cells when a canonical report exists.",
    )
    args = ap.parse_args()

    applied = 0
    touched_files = set()
    for root in CANON_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            parsed = parse_canonical(path)
            if not parsed:
                continue
            updated, ledger_path = update_ledger(parsed, apply=args.apply, force=args.force, overwrite_nr=bool(args.overwrite_nr))
            if updated:
                applied += 1
                touched_files.add(str(ledger_path))

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] updates={applied} ledgers_touched={len(touched_files)}")
    for lf in sorted(touched_files):
        print(f"  - {lf}")


if __name__ == "__main__":
    main()
