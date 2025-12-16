from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from chimera_v2c.lib import team_mapper


CANON_ROOT = Path("reports/specialist_reports")
DAILY_LEDGER_ROOT = Path("reports/daily_ledgers")


FILENAME_RE = re.compile(
    r"(?P<date>20\d{6})_"
    r"(?P<league>nba|nfl|nhl)_"
    r"(?P<away>[^_@]+)@(?P<home>[^_@]+)_"
    r"(?P<model>.+)\.txt$",
    re.IGNORECASE,
)


@dataclass
class CanonicalMeta:
    path: Path
    date_yyyymmdd: str
    league: str
    away_raw: str
    home_raw: str
    model_token: str

    @property
    def date_iso(self) -> str:
        return f"{self.date_yyyymmdd[:4]}-{self.date_yyyymmdd[4:6]}-{self.date_yyyymmdd[6:]}"

    def normalized_teams(self) -> Tuple[str, str]:
        league = self.league.lower()
        away = (
            team_mapper.normalize_team_code(self.away_raw, league)
            or self.away_raw.upper()
        )
        home = (
            team_mapper.normalize_team_code(self.home_raw, league)
            or self.home_raw.upper()
        )
        return away, home

    def model_label(self) -> str:
        """
        Produce a readable, stable model label for the HELIOS header.
        We keep this simple and avoid overfitting to historical naming quirks.
        """
        raw = self.model_token
        lower = raw.lower()
        family = "Unknown"
        if "grok" in lower:
            family = "Grok"
        elif "gemini" in lower:
            family = "Gemini"
        elif "gpt" in lower:
            family = "GPT"
        elif "v2c" in lower or "chimera" in lower:
            family = "Chimera_v2c"

        # If the raw token already looks like a detailed label, surface it.
        if any(tag in lower for tag in ("scientist", "directive", "helio", "v8", "v7", "v6")):
            return raw

        return f"{family}_{raw}"

    def ledger_key(self, away: str, home: str) -> Tuple[str, str, str]:
        return (self.date_yyyymmdd, self.league.lower(), f"{away}@{home}")

    def ledger_column(self) -> Optional[str]:
        lower = self.model_token.lower()
        if "grok" in lower:
            return "grok"
        if "gemini" in lower:
            return "gemini"
        if "gpt" in lower:
            return "gpt"
        if "v2c" in lower or "chimera" in lower:
            return "v2c"
        return None


def load_daily_ledger_index() -> Dict[Tuple[str, str, str], Dict[str, str]]:
    """
    Build an in-memory index over daily ledgers:
    key: (YYYYMMDD, league, matchup) -> row dict
    """
    index: Dict[Tuple[str, str, str], Dict[str, str]] = {}

    if not DAILY_LEDGER_ROOT.exists():
        return index

    for path in DAILY_LEDGER_ROOT.iterdir():
        if not path.name.endswith("_daily_game_ledger.csv"):
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                date = row.get("date") or ""
                if not date:
                    continue
                date_clean = date.replace("-", "")
                league = (row.get("league") or "").lower()
                matchup = (row.get("matchup") or "").upper()
                if not league or not matchup:
                    continue
                key = (date_clean, league, matchup)
                # Do not overwrite if duplicate keys appear; keep first seen.
                index.setdefault(key, row)
    return index


def parse_meta(path: Path) -> Optional[CanonicalMeta]:
    m = FILENAME_RE.search(path.name)
    if not m:
        return None
    gd = m.groupdict()
    return CanonicalMeta(
        path=path,
        date_yyyymmdd=gd["date"],
        league=gd["league"],
        away_raw=gd["away"],
        home_raw=gd["home"],
        model_token=gd["model"],
    )


def has_prediction_header(text: str) -> bool:
    return "HELIOS_PREDICTION_HEADER_START" in text


def wrap_existing_prediction_block(text: str) -> Optional[str]:
    """
    For files that already contain a Game/Model + Prediction_Moneyline block
    near the top, wrap that block with HELIOS_PREDICTION_HEADER_START/END.
    """
    if "Prediction_Moneyline:" not in text:
        return None

    lines = text.splitlines(keepends=True)
    try:
        pm_idx = next(i for i, ln in enumerate(lines) if "Prediction_Moneyline:" in ln)
    except StopIteration:
        return None

    # Find the end of the header as the first completely blank line after the
    # prediction block. If none exists, treat the whole file as header.
    end_idx = None
    for i in range(pm_idx + 1, len(lines)):
        if lines[i].strip() == "":
            end_idx = i + 1  # include the blank line
            break

    if end_idx is None:
        end_idx = len(lines)

    header_block = "".join(lines[:end_idx])
    rest = "".join(lines[end_idx:])

    if "HELIOS_PREDICTION_HEADER_START" in header_block:
        # Nothing to do.
        return None

    new_text = (
        "HELIOS_PREDICTION_HEADER_START\n"
        f"{header_block}"
        "HELIOS_PREDICTION_HEADER_END\n"
    )
    if rest:
        new_text += rest
    return new_text


def derive_moneyline_from_ledger(
    meta: CanonicalMeta,
    ledger_index: Dict[Tuple[str, str, str], Dict[str, str]],
) -> Tuple[Optional[str], Optional[float]]:
    """
    From the daily ledger, map to (winner_code, p_home) if possible.
    """
    away, home = meta.normalized_teams()
    key = meta.ledger_key(away, home)
    row = ledger_index.get(key)
    if not row:
        return None, None

    col = meta.ledger_column()
    if not col:
        return None, None

    val = (row.get(col) or "").strip()
    if not val:
        return None, None

    try:
        p_home = float(val)
    except ValueError:
        return None, None

    if p_home >= 0.5:
        return home, p_home
    return away, p_home


def synthesize_header(
    meta: CanonicalMeta,
    ledger_index: Dict[Tuple[str, str, str], Dict[str, str]],
) -> str:
    """
    Build a minimal HELIOS prediction header for a file that lacks one.
    We never infer probabilities heuristically; if we cannot safely obtain a
    numeric p_home from the daily ledgers, we leave it blank.
    """
    away, home = meta.normalized_teams()
    winner, p_home = derive_moneyline_from_ledger(meta, ledger_index)

    p_home_str = f"{p_home:.2f}" if p_home is not None else ""
    p_true_str = ""
    if p_home is not None and winner:
        p_true = p_home if winner == home else (1.0 - p_home)
        p_true_str = f"{p_true:.2f}"

    winner_str = winner or ""

    header_lines = [
        "HELIOS_PREDICTION_HEADER_START",
        f"Game: {meta.date_iso} {meta.league.upper()} {away}@{home}",
        f"Model: {meta.model_label()}",
        "",
        "Prediction_Moneyline:",
        f"Winner: {winner_str}",
        f"p_home: {p_home_str}",
        f"p_true: {p_true_str}",
        "ptcs: ",
        "HELIOS_PREDICTION_HEADER_END",
        "",
    ]
    return "\n".join(header_lines)


def process_file(
    meta: CanonicalMeta,
    ledger_index: Dict[Tuple[str, str, str], Dict[str, str]],
) -> None:
    """
    Ensure a given canonical report has a HELIOS prediction header.
    This function is intentionally conservative: it never modifies existing
    Prediction_Moneyline content, only wraps it, and only synthesizes a new
    header when none is present.
    """
    text = meta.path.read_text(encoding="utf-8", errors="ignore")

    if has_prediction_header(text):
        return

    # Case 1: There is already a Game/Model + Prediction_Moneyline block; just wrap it.
    wrapped = wrap_existing_prediction_block(text)
    if wrapped is not None:
        meta.path.write_text(wrapped, encoding="utf-8")
        return

    # Case 2: No explicit prediction block; synthesize a minimal header and prepend.
    header = synthesize_header(meta, ledger_index)
    new_text = header + text
    meta.path.write_text(new_text, encoding="utf-8")


def find_headerless_canonical_files() -> Dict[Path, CanonicalMeta]:
    metas: Dict[Path, CanonicalMeta] = {}
    if not CANON_ROOT.exists():
        return metas

    for league_dir in ("NBA", "NFL", "NHL"):
        base = CANON_ROOT / league_dir
        if not base.exists():
            continue
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                if not fn.endswith(".txt"):
                    continue
                path = Path(dirpath) / fn
                text = path.read_text(encoding="utf-8", errors="ignore")
                if has_prediction_header(text):
                    continue
                meta = parse_meta(path)
                if not meta:
                    continue
                metas[path] = meta
    return metas


def main() -> None:
    ledger_index = load_daily_ledger_index()
    metas = find_headerless_canonical_files()

    print(f"Found {len(metas)} canonical reports without HELIOS headers.")

    for path, meta in sorted(metas.items(), key=lambda kv: kv[0].as_posix()):
        process_file(meta, ledger_index)
        print(f"Patched HELIOS header for {path}")


if __name__ == "__main__":
    main()
