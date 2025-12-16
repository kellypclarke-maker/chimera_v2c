#!/usr/bin/env python
"""
Read-only parity audit across:
- League schedules (CSV) in reports/thesis_summaries/
- Canonical specialist reports in reports/specialist_reports/
- Daily ledgers in reports/daily_ledgers/
- Derived master ledger in reports/master_ledger/master_game_ledger.csv

This tool does not mutate any files. It exists to catch:
- Phantom/misdated games in daily ledgers vs schedule
- Incorrect final scores/outcome tags in daily ledgers vs schedule
- Canonical p_home mismatches vs daily ledger model columns (grok/gemini/gpt)
- Drift/missing keys between daily ledgers and the master ledger
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from chimera_v2c.lib.team_mapper import LEAGUE_MAP, normalize_team_code


DAILY_DIR = Path("reports/daily_ledgers")
MASTER_PATH = Path("reports/master_ledger/master_game_ledger.csv")
SCHEDULE_DIR = Path("reports/thesis_summaries")
CANON_ROOTS = [
    Path("reports/specialist_reports/NBA"),
    Path("reports/specialist_reports/NHL"),
    Path("reports/specialist_reports/NFL"),
]

SCHEDULE_GLOBS = {
    "nba": "nba_schedule_*.csv",
    "nhl": "nhl_schedule_*.csv",
    "nfl": "nfl_schedule_*.csv",
}

SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")

GAME_RE = re.compile(r"Game:\s*(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+([A-Z0-9]+)@([A-Z0-9]+)", re.IGNORECASE)
WINNER_RE = re.compile(r"Winner:\s*([A-Z0-9]+)", re.IGNORECASE)
PTRUE_RE = re.compile(r"p_true:\s*([0-9.]+)", re.IGNORECASE)
PTRUE_RAW_RE = re.compile(r"p_true_raw:\s*([0-9.]+)", re.IGNORECASE)
PTRUE_CAL_RE = re.compile(r"p_true_calibrated:\s*([0-9.]+)", re.IGNORECASE)
P_HOME_RE = re.compile(r"p_home:\s*([0-9.]+)", re.IGNORECASE)
MODEL_RE = re.compile(r"Model:\s*(.+)", re.IGNORECASE)

MONEYLINE_BLOCK_RE = re.compile(
    r"Prediction_Moneyline:(.*?)(?:Prediction_Spread|Prediction_Total|HELIOS_PREDICTION_HEADER_END)",
    re.IGNORECASE | re.DOTALL,
)

CANONICAL_FNAME_RE = re.compile(
    r"^(?P<date>\d{8})_(?P<league>nba|nhl|nfl)_(?P<away>[a-z0-9]+)@(?P<home>[a-z0-9]+)_(?P<model>gpt|grok|gemini)\.txt$",
    re.IGNORECASE,
)

CANONICAL_LEAGUE_FOLDERS = {
    "nba": "NBA",
    "nhl": "NHL",
    "nfl": "NFL",
}


def _clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9 ]+", " ", text.upper()).strip()


def normalize_team_loose(raw: str, league: str) -> Optional[str]:
    if raw is None:
        return None
    raw = str(raw).strip()
    if not raw:
        return None
    code = normalize_team_code(raw, league)
    if code:
        return code

    upper = _clean_text(raw)
    data = LEAGUE_MAP.get(league.lower(), {})
    hits: List[str] = []
    for cand_code, aliases in data.items():
        for alias in aliases:
            alias_u = _clean_text(alias)
            if alias_u and alias_u in upper:
                hits.append(cand_code)
                break
    hits = sorted(set(hits))
    if len(hits) == 1:
        return hits[0]

    for token in upper.split():
        code = normalize_team_code(token, league)
        if code:
            return code
    return None


def canonical_matchup(league: str, matchup_raw: str) -> Optional[str]:
    if not matchup_raw or "@" not in matchup_raw:
        return None
    away_raw, home_raw = matchup_raw.split("@", 1)
    away = normalize_team_loose(away_raw, league) or away_raw.strip().upper()
    home = normalize_team_loose(home_raw, league) or home_raw.strip().upper()
    return f"{away}@{home}"


def parse_canonical_filename(name: str) -> Optional[Tuple[str, str, str, str]]:
    m = CANONICAL_FNAME_RE.match(name)
    if not m:
        return None
    date_token = m.group("date")
    try:
        date_iso = dt.datetime.strptime(date_token, "%Y%m%d").date().isoformat()
    except ValueError:
        return None
    league = m.group("league").lower()
    away_raw = m.group("away").upper()
    home_raw = m.group("home").upper()
    away = normalize_team_code(away_raw, league) or away_raw
    home = normalize_team_code(home_raw, league) or home_raw
    matchup = f"{away}@{home}"
    model = m.group("model").lower()
    return (date_iso, league, matchup, model)


def expected_canonical_filename(date: str, league: str, matchup: str, model: str) -> str:
    date_token = date.replace("-", "")
    away, home = matchup.split("@", 1)
    return f"{date_token}_{league.lower()}_{away.lower()}@{home.lower()}_{model.lower()}.txt"


def expected_month_folder(date: str) -> str:
    return date[:7]


def expected_league_folder(league: str) -> str:
    return CANONICAL_LEAGUE_FOLDERS.get(league.lower(), league.upper())


@dataclass(frozen=True)
class Outcome:
    away_score: float
    home_score: float
    home_win: Optional[bool]


def parse_outcome(text: str) -> Optional[Outcome]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s or s.lower() == "nan":
        return None
    m = SCORE_RE.search(s)
    if not m:
        return None
    away_score = float(m.group(1))
    home_score = float(m.group(2))
    home_win = None
    sl = s.lower()
    if "(home" in sl:
        home_win = True
    elif "(away" in sl:
        home_win = False
    else:
        if home_score > away_score:
            home_win = True
        elif away_score > home_score:
            home_win = False
    return Outcome(away_score=away_score, home_score=home_score, home_win=home_win)


def find_latest_schedule(league: str) -> Optional[Path]:
    pattern = SCHEDULE_GLOBS.get(league.lower())
    if not pattern:
        return None
    candidates = list(SCHEDULE_DIR.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_schedule_map(league: str, path: Path) -> Dict[Tuple[str, str, str], Outcome]:
    df = pd.read_csv(path)
    required = {"date", "away", "home", "away_score", "home_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"[error] schedule {league} missing columns: {sorted(missing)} ({path})")

    df = df[df["date"].notna() & df["away"].notna() & df["home"].notna()].copy()
    df["away_code"] = df["away"].apply(lambda x: normalize_team_loose(x, league))
    df["home_code"] = df["home"].apply(lambda x: normalize_team_loose(x, league))
    df = df[df["away_code"].notna() & df["home_code"].notna()].copy()
    df["matchup_code"] = df["away_code"] + "@" + df["home_code"]

    df["outcome_str"] = df.get("outcome", "").fillna("").astype(str)
    df["has_outcome"] = df["outcome_str"].str.strip().ne("")
    df = df.sort_values(["date", "matchup_code", "has_outcome"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["date", "matchup_code"], keep="first")

    out: Dict[Tuple[str, str, str], Outcome] = {}
    for _, row in df.iterrows():
        date = str(row["date"]).strip()
        matchup = str(row["matchup_code"]).strip()
        try:
            away_score = float(row["away_score"])
            home_score = float(row["home_score"])
        except (TypeError, ValueError):
            continue
        parsed = parse_outcome(str(row.get("outcome_str", "")))
        home_win = parsed.home_win if parsed else None
        out[(date, league, matchup)] = Outcome(away_score=away_score, home_score=home_score, home_win=home_win)
    return out


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

    families = detect(header_model) | detect(fname)
    if len(families) == 1:
        return next(iter(families))
    header_fams = detect(header_model)
    if len(header_fams) == 1:
        return next(iter(header_fams))
    fname_fams = detect(fname)
    if len(fname_fams) == 1:
        return next(iter(fname_fams))
    return None


@dataclass(frozen=True)
class CanonicalProb:
    date: str
    league: str
    matchup: str
    model: str
    p_home: float
    path: Path


@dataclass(frozen=True)
class CanonicalHeader:
    date: str
    league: str
    matchup: str
    header_model: str


def parse_canonical_header(path: Path) -> Optional[CanonicalHeader]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = GAME_RE.search(text)
    if not m:
        return None
    date, league_raw, away_raw, home_raw = m.groups()
    league = league_raw.lower()
    away = normalize_team_code(away_raw, league) or away_raw
    home = normalize_team_code(home_raw, league) or home_raw
    matchup = f"{away}@{home}"
    header_model = ""
    model_line = MODEL_RE.search(text)
    if model_line:
        header_model = model_line.group(1).strip()
    return CanonicalHeader(date=date, league=league, matchup=matchup, header_model=header_model)


def parse_canonical_prob(path: Path) -> Optional[CanonicalProb]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    m = GAME_RE.search(text)
    if not m:
        return None
    date, league_raw, away_raw, home_raw = m.groups()
    league = league_raw.lower()
    away = normalize_team_code(away_raw, league) or away_raw
    home = normalize_team_code(home_raw, league) or home_raw
    matchup = f"{away}@{home}"

    winner_m = WINNER_RE.search(text)
    if not winner_m:
        return None
    winner = normalize_team_code(winner_m.group(1), league)
    if winner not in {away, home}:
        return None

    p_home = None
    p_true = None
    moneyline_text = text
    moneyline_match = MONEYLINE_BLOCK_RE.search(text)
    if moneyline_match:
        moneyline_text = moneyline_match.group(1)

    m_home = P_HOME_RE.search(moneyline_text)
    if m_home:
        try:
            p_home = float(m_home.group(1))
        except ValueError:
            p_home = None

    for rx in (PTRUE_CAL_RE, PTRUE_RAW_RE, PTRUE_RE):
        m_true = rx.search(moneyline_text)
        if not m_true:
            continue
        try:
            p_true = float(m_true.group(1))
        except ValueError:
            p_true = None
        break
    if p_home is None and p_true is None:
        return None

    header_model = ""
    model_line = MODEL_RE.search(text)
    if model_line:
        header_model = model_line.group(1).strip()
    fam = model_family(path.name, header_model)
    if fam is None:
        return None

    if p_home is None:
        p_home = p_true if winner == home else 1 - p_true
    return CanonicalProb(date=date, league=league, matchup=matchup, model=fam, p_home=p_home, path=path)


def _within(date_str: str, start: Optional[str], end: Optional[str]) -> bool:
    if not start and not end:
        return True
    try:
        d = dt.date.fromisoformat(date_str)
    except ValueError:
        return False
    if start and d < dt.date.fromisoformat(start):
        return False
    if end and d > dt.date.fromisoformat(end):
        return False
    return True


def ffloat(value: object) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only parity audit across schedule/canonicals/daily/master ledgers.")
    ap.add_argument("--start-date", help="ISO start date filter (inclusive).")
    ap.add_argument("--end-date", help="ISO end date filter (inclusive).")
    ap.add_argument("--tol", type=float, default=0.005, help="p_home tolerance for canonical vs ledger checks (default: 0.005).")
    ap.add_argument("--skip-master", action="store_true", help="Skip daily ↔ master drift checks (useful while repairing daily parity).")
    args = ap.parse_args()

    schedule_map: Dict[Tuple[str, str, str], Outcome] = {}
    for league in ("nba", "nhl", "nfl"):
        path = find_latest_schedule(league)
        if not path:
            raise SystemExit(f"[error] missing schedule CSV for {league} under {SCHEDULE_DIR}/")
        schedule_map.update(load_schedule_map(league, path))

    daily_files = sorted(DAILY_DIR.glob("*_daily_game_ledger.csv"))
    if not daily_files:
        raise SystemExit(f"[error] no daily ledgers found under {DAILY_DIR}/")

    phantom_rows: List[Tuple[str, str, str, str, Optional[str]]] = []
    score_mismatches: List[Tuple[str, str, str, str, str]] = []
    outcome_mismatches: List[Tuple[str, str, str, str, str]] = []

    daily_index: Dict[Tuple[str, str, str], Tuple[str, dict]] = {}
    for daily_path in daily_files:
        df = pd.read_csv(daily_path).fillna("")
        if not {"date", "league", "matchup"}.issubset(df.columns):
            continue
        for _, row in df.iterrows():
            date = str(row.get("date", "")).strip()
            league = str(row.get("league", "")).strip().lower()
            matchup = canonical_matchup(league, str(row.get("matchup", "")).strip())
            if not date or not league or not matchup:
                continue
            if not _within(date, args.start_date, args.end_date):
                continue
            key = (date, league, matchup)
            daily_index[key] = (daily_path.name, row.to_dict())

            sched = schedule_map.get(key)
            if sched is None:
                hint = None
                try:
                    d = dt.date.fromisoformat(date)
                    for delta in (-2, -1, 1, 2):
                        dd = (d + dt.timedelta(days=delta)).isoformat()
                        if (dd, league, matchup) in schedule_map:
                            hint = dd
                            break
                except ValueError:
                    hint = None
                phantom_rows.append((date, league, matchup, daily_path.name, hint))
                continue

            ledger_outcome = parse_outcome(str(row.get("actual_outcome", "")))
            if not ledger_outcome:
                continue

            if (ledger_outcome.away_score != sched.away_score) or (ledger_outcome.home_score != sched.home_score):
                score_mismatches.append(
                    (date, league, matchup, str(row.get("actual_outcome", "")), f"{sched.away_score}-{sched.home_score}")
                )
            if (
                ledger_outcome.home_win is not None
                and sched.home_win is not None
                and ledger_outcome.home_win != sched.home_win
            ):
                outcome_mismatches.append((date, league, matchup, str(row.get("actual_outcome", "")), str(sched.home_win)))

    # Canonical vs daily p_home
    canon_dupes: Dict[Tuple[str, str, str, str], List[Path]] = {}
    canon_map: Dict[Tuple[str, str, str, str], CanonicalProb] = {}
    canon_cell_mismatches: List[Tuple[str, str, str, str, float, float, str]] = []
    canon_missing_cell: List[Tuple[str, str, str, str, str]] = []
    canon_missing_schedule: List[Tuple[str, str, str, str, str, Optional[str]]] = []
    canon_bad_filename: List[Tuple[str, str, str, str, str, str]] = []
    canon_bad_folder: List[Tuple[str, str, str, str, str]] = []
    canon_missing_header: List[Tuple[str, str]] = []
    canon_missing_prob: List[Tuple[str, str, str]] = []
    canon_filename_header_mismatch: List[Tuple[str, str, str, str, str]] = []

    for root in CANON_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.txt"):
            fname_parsed = parse_canonical_filename(path.name)
            if fname_parsed and _within(fname_parsed[0], args.start_date, args.end_date):
                header = parse_canonical_header(path)
                if not header:
                    canon_missing_header.append((fname_parsed[0], str(path)))
                    continue
                if (header.date, header.league, header.matchup) != (fname_parsed[0], fname_parsed[1], fname_parsed[2]):
                    canon_filename_header_mismatch.append(
                        (fname_parsed[0], fname_parsed[1], fname_parsed[2], path.name, str(path))
                    )
            parsed = parse_canonical_prob(path)
            if not parsed:
                if fname_parsed and _within(fname_parsed[0], args.start_date, args.end_date):
                    header = parse_canonical_header(path)
                    if header and _within(header.date, args.start_date, args.end_date):
                        canon_missing_prob.append((header.date, header.league, header.matchup))
                continue
            if not _within(parsed.date, args.start_date, args.end_date):
                continue
            if (parsed.date, parsed.league, parsed.matchup) not in schedule_map:
                hint = None
                try:
                    d = dt.date.fromisoformat(parsed.date)
                    for delta in (-2, -1, 1, 2):
                        dd = (d + dt.timedelta(days=delta)).isoformat()
                        if (dd, parsed.league, parsed.matchup) in schedule_map:
                            hint = dd
                            break
                except ValueError:
                    hint = None
                canon_missing_schedule.append((parsed.date, parsed.league, parsed.matchup, parsed.model, str(path), hint))

            expected_name = expected_canonical_filename(parsed.date, parsed.league, parsed.matchup, parsed.model)
            if path.name != expected_name:
                canon_bad_filename.append(
                    (parsed.date, parsed.league, parsed.matchup, parsed.model, path.name, expected_name)
                )
            month = expected_month_folder(parsed.date)
            league_folder = expected_league_folder(parsed.league)
            if path.parent.name != month or path.parent.parent.name != league_folder:
                canon_bad_folder.append((parsed.date, parsed.league, parsed.matchup, str(path), f"{league_folder}/{month}"))

            key = (parsed.date, parsed.league, parsed.matchup, parsed.model)
            if key in canon_map:
                canon_dupes.setdefault(key, [canon_map[key].path]).append(parsed.path)
                # keep newest file by mtime
                if parsed.path.stat().st_mtime > canon_map[key].path.stat().st_mtime:
                    canon_map[key] = parsed
            else:
                canon_map[key] = parsed

    for (date, league, matchup, model), canon in canon_map.items():
        daily_key = (date, league, matchup)
        daily_hit = daily_index.get(daily_key)
        if not daily_hit:
            continue
        ledger_file, row = daily_hit
        if model not in row:
            continue
        val = ffloat(row.get(model, ""))
        if val is None:
            canon_missing_cell.append((date, league, matchup, model, ledger_file))
            continue
        if abs(val - canon.p_home) > (args.tol + 1e-12):
            canon_cell_mismatches.append((date, league, matchup, model, val, canon.p_home, ledger_file))

    # Daily vs master
    missing_in_master: List[Tuple[str, str, str]] = []
    only_master: List[Tuple[str, str, str]] = []
    master_mismatches: List[Tuple[str, str, str, str, str]] = []

    if MASTER_PATH.exists() and not args.skip_master:
        master = pd.read_csv(MASTER_PATH).fillna("")
        master["league"] = master["league"].astype(str).str.lower()
        master_map: Dict[Tuple[str, str, str], dict] = {}
        for _, r in master.iterrows():
            k = (str(r.get("date", "")).strip(), str(r.get("league", "")).strip().lower(), str(r.get("matchup", "")).strip())
            master_map[k] = r.to_dict()

        for k in daily_index:
            if k not in master_map:
                missing_in_master.append(k)

        for k in master_map:
            if k not in daily_index and _within(k[0], args.start_date, args.end_date):
                only_master.append(k)

        field_pairs = [
            ("v2c", "v2c"),
            ("gemini", "gemini"),
            ("grok", "grok"),
            ("gpt", "gpt"),
            ("kalshi_mid", "kalshi_mid"),
            ("market_proxy", "market_proxy"),
            ("moneypuck", "moneypuck"),
        ]
        for k, (ledger_file, row) in daily_index.items():
            m = master_map.get(k)
            if not m:
                continue
            for master_field, daily_field in field_pairs:
                mv = ffloat(m.get(master_field, ""))
                dv = ffloat(row.get(daily_field, ""))
                if mv is None and dv is None:
                    continue
                if mv is None or dv is None or abs(mv - dv) > 1e-6:
                    master_mismatches.append((k[0], k[1], k[2], master_field, ledger_file))
            # outcome string compare if both present
            mout = str(m.get("actual_outcome", "")).strip()
            dout = str(row.get("actual_outcome", "")).strip()
            if mout and dout and mout != dout:
                master_mismatches.append((k[0], k[1], k[2], "actual_outcome", ledger_file))

    # Report
    print("=== Schedule ↔ Daily Ledgers ===")
    print(f"phantom_rows={len(phantom_rows)} score_mismatches={len(score_mismatches)} outcome_mismatches={len(outcome_mismatches)}")
    if phantom_rows:
        for row in sorted(phantom_rows)[:20]:
            print(f"  phantom: {row}")
    if score_mismatches:
        for row in sorted(score_mismatches)[:20]:
            print(f"  score: {row}")
    if outcome_mismatches:
        for row in sorted(outcome_mismatches)[:20]:
            print(f"  outcome: {row}")

    print("\n=== Canonicals ↔ Daily Ledgers ===")
    print(f"canonical_probs={len(canon_map)} dup_keys={len(canon_dupes)} missing_cells={len(canon_missing_cell)} mismatches={len(canon_cell_mismatches)} tol={args.tol}")
    if canon_dupes:
        for k, paths in list(canon_dupes.items())[:10]:
            print(f"  dup: {k} -> {len(paths)} files")
    if canon_missing_cell:
        for row in sorted(canon_missing_cell)[:20]:
            print(f"  missing_cell: {row}")
    if canon_cell_mismatches:
        for row in sorted(canon_cell_mismatches)[:20]:
            print(f"  mismatch: {row}")

    print("\n=== Schedule ↔ Canonicals ===")
    print(
        f"missing_schedule={len(canon_missing_schedule)} bad_filenames={len(canon_bad_filename)} bad_folders={len(canon_bad_folder)} "
        f"missing_headers={len(canon_missing_header)} missing_probs={len(canon_missing_prob)} filename_header_mismatch={len(canon_filename_header_mismatch)}"
    )
    if canon_missing_schedule:
        for row in sorted(canon_missing_schedule)[:20]:
            print(f"  missing_schedule: {row}")
    if canon_bad_filename:
        for row in sorted(canon_bad_filename)[:20]:
            print(f"  bad_filename: {row}")
    if canon_bad_folder:
        for row in sorted(canon_bad_folder)[:20]:
            print(f"  bad_folder: {row}")
    if canon_missing_header:
        for row in sorted(canon_missing_header)[:20]:
            print(f"  missing_header: {row}")
    if canon_missing_prob:
        for row in sorted(canon_missing_prob)[:20]:
            print(f"  missing_prob: {row}")
    if canon_filename_header_mismatch:
        for row in sorted(canon_filename_header_mismatch)[:20]:
            print(f"  filename_header_mismatch: {row}")

    print("\n=== Daily Ledgers ↔ Master Ledger ===")
    if args.skip_master:
        print("skipped=1 (--skip-master)")
    elif not MASTER_PATH.exists():
        print(f"master_missing=1 (missing file {MASTER_PATH})")
    else:
        print(f"missing_in_master={len(missing_in_master)} only_master={len(only_master)} field_mismatches={len(master_mismatches)}")
        if missing_in_master:
            for row in sorted(missing_in_master)[:20]:
                print(f"  missing_in_master: {row}")
        if only_master:
            for row in sorted(only_master)[:20]:
                print(f"  only_master: {row}")
        if master_mismatches:
            for row in sorted(master_mismatches)[:20]:
                print(f"  field_mismatch: {row}")

    failures = (
        len(phantom_rows)
        + len(score_mismatches)
        + len(outcome_mismatches)
        + len(canon_missing_cell)
        + len(canon_cell_mismatches)
        + len(canon_missing_schedule)
        + len(canon_bad_filename)
        + len(canon_bad_folder)
        + len(canon_missing_header)
        + len(canon_missing_prob)
        + len(canon_filename_header_mismatch)
    )
    if not args.skip_master:
        failures += len(only_master) + len(master_mismatches)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
