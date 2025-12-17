#!/usr/bin/env python
"""
LLM-assisted specialist ingester with preview-first workflow.

Reads raw specialist reports (Gemini/Grok/GPT) from `reports/specialist_reports/raw/`
(preferred; legacy location: `reports/specialist_reports/archive/raw/`),
parses/synthesizes HELIOS headers, writes canonical per-game reports, and
fills blank daily-ledger model cells (append-only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.lib.team_mapper import normalize_team_code
from chimera_v2c.src.ledger.formatting import MISSING_SENTINEL, format_prob_cell
from chimera_v2c.src.ledger.guard import (
    LedgerGuardError,
    compute_append_only_diff,
    load_locked_dates,
    snapshot_file,
    write_csv,
)
from chimera_v2c.src.ledger.outcomes import parse_home_win


RAW_DEFAULT = Path("reports/specialist_reports/raw")
RAW_LEGACY = Path("reports/specialist_reports/archive/raw")
ARCHIVE_PROCESSED = Path("reports/specialist_reports/archive/raw_processed")
ARCHIVE_UNPARSED = Path("reports/specialist_reports/archive/raw_unparsed")
CANONICAL_ROOT = Path("reports/specialist_reports")
DAILY_LEDGER_DIR = Path("reports/daily_ledgers")
LOCK_DIR = DAILY_LEDGER_DIR / "locked"
DAILY_SNAPSHOT_DIR = DAILY_LEDGER_DIR / "snapshots"

HELIOS_START = "HELIOS_PREDICTION_HEADER_START"
HELIOS_END = "HELIOS_PREDICTION_HEADER_END"

MODEL_FALLBACK = {
    "grok": "grok",
    "gemini": "gemini",
    "gpt": "gpt",
}


@dataclass
class ParsedGame:
    date: str
    league: str
    away: str
    home: str
    winner: Optional[str]
    p_true: Optional[float]
    p_home: Optional[float]
    model_label: str
    source_file: Path
    raw_header: Optional[str] = None

    @property
    def matchup(self) -> str:
        return f"{self.away}@{self.home}"


def clamp_prob(p: float) -> float:
    return float(max(0.0, min(1.0, p)))


def normalize_team(value: str, league: Optional[str] = None) -> Optional[str]:
    if not value:
        return None
    league_norm = league.lower() if league else None
    if league_norm in ("nba", "nhl", "nfl"):
        norm = normalize_team_code(value, league_norm)
        if norm:
            return norm
    for lg in ("nba", "nhl", "nfl"):
        norm = normalize_team_code(value, lg)
        if norm:
            return norm
    return value.strip().upper()


def normalize_league(value: str) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in ("nba", "nhl", "nfl"):
        return v
    return None


def parse_helios_blocks(text: str, src: Path) -> List[ParsedGame]:
    games: List[ParsedGame] = []
    pattern = re.compile(rf"{HELIOS_START}(.*?){HELIOS_END}", re.DOTALL | re.IGNORECASE)
    for block in pattern.findall(text):
        header = block.strip()
        game_line = re.search(
            r"Game:\s*(\d{4}-\d{2}-\d{2})\s+([A-Za-z]+)\s+([A-Z0-9]+@[A-Z0-9]+)",
            header,
            re.IGNORECASE,
        )
        model_line = re.search(r"Model:\s*(.+)", header, re.IGNORECASE)
        winner_line = re.search(r"Winner:\s*([A-Z0-9]+)", header, re.IGNORECASE)
        p_home_line = re.search(r"p_home:\s*([0-9.]+)", header, re.IGNORECASE)
        p_true_line = re.search(r"p_true:\s*([0-9.]+)", header, re.IGNORECASE)
        if not game_line:
            continue
        date_str, league_raw, matchup = game_line.groups()
        league = normalize_league(league_raw)
        if not league:
            continue
        away_raw, home_raw = matchup.split("@", 1)
        away = normalize_team(away_raw, league)
        home = normalize_team(home_raw, league)
        if not away or not home:
            continue
        winner = normalize_team(winner_line.group(1), league) if winner_line else None
        p_home = None
        if p_home_line:
            try:
                p_home = clamp_prob(float(p_home_line.group(1)))
            except ValueError:
                p_home = None
        p_true = None
        if p_true_line:
            try:
                p_true = clamp_prob(float(p_true_line.group(1)))
            except ValueError:
                p_true = None
        model_label = model_line.group(1).strip() if model_line else ""
        games.append(
            ParsedGame(
                date=date_str,
                league=league,
                away=away,
                home=home,
                winner=winner,
                p_true=p_true,
                p_home=p_home,
                model_label=model_label,
                source_file=src,
                raw_header=f"{HELIOS_START}\n{header}\n{HELIOS_END}",
            )
        )
    return games


def parse_legacy_helios_prediction_headers(text: str, src: Path) -> List[ParsedGame]:
    """
    Parse older/raw reports that use repeated blocks like:

      HELIOS_PREDICTION_HEADER
      timestamp: 2025-12-15 15:47 PST
      league: NBA
      matchup: DET@BOS
      model: gpt-5.2-thinking
      winner: BOS
      p_home: 0.52

    Blocks are typically separated by '---'. This parser is deterministic and
    avoids an OpenAI call when the structure is present.
    """
    games: List[ParsedGame] = []
    blocks = re.split(r"(?im)^\s*HELIOS_PREDICTION_HEADER\s*$", text)
    for chunk in blocks[1:]:
        # Stop at the next obvious separator to keep parsing tight.
        head = chunk.split("\n---", 1)[0]
        date_str = None
        m_ts = re.search(r"(?im)^\s*timestamp:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\b", head)
        if m_ts:
            date_str = m_ts.group(1).strip()
        m_league = re.search(r"(?im)^\s*league:\s*([A-Za-z]+)\s*$", head)
        m_match = re.search(r"(?im)^\s*matchup:\s*([A-Z0-9]+@[A-Z0-9]+)\s*$", head)
        m_model = re.search(r"(?im)^\s*model:\s*(.+?)\s*$", head)
        m_winner = re.search(r"(?im)^\s*winner:\s*([A-Z0-9]+)\s*$", head)
        m_p_home = re.search(r"(?im)^\s*p_home:\s*([0-9.]+)\s*$", head)
        if not (date_str and m_league and m_match):
            continue
        league = normalize_league(m_league.group(1))
        if not league:
            continue
        away_raw, home_raw = m_match.group(1).split("@", 1)
        away = normalize_team(away_raw, league)
        home = normalize_team(home_raw, league)
        if not away or not home:
            continue
        winner = normalize_team(m_winner.group(1), league) if m_winner else None
        p_home = None
        if m_p_home:
            try:
                p_home = clamp_prob(float(m_p_home.group(1)))
            except ValueError:
                p_home = None
        model_label = m_model.group(1).strip() if m_model else ""
        games.append(
            ParsedGame(
                date=date_str,
                league=league,
                away=away,
                home=home,
                winner=winner,
                p_true=None,
                p_home=p_home,
                model_label=model_label,
                source_file=src,
                raw_header=None,
            )
        )
    return games


def infer_league_from_filename(file_name: str) -> Optional[str]:
    lowered = (file_name or "").lower()
    if " nba" in lowered or "nba" in lowered:
        return "nba"
    if " nhl" in lowered or "nhl" in lowered:
        return "nhl"
    if " nfl" in lowered or "nfl" in lowered:
        return "nfl"
    return None


def parse_compact_prediction_headers(text: str, src: Path) -> List[ParsedGame]:
    """
    Parse compact blocks like:

      HELIOS_PREDICTION_HEADER
      DATE: 2025-12-16
      MATCHUP: UTA@BOS
      P_HOME: 0.505
      WINNER: BOS

    This format appears in some NHL packets and does not include an explicit
    `league:` line. When absent, the league is inferred from the filename.
    """
    games: List[ParsedGame] = []
    blocks = re.split(r"(?im)^\s*HELIOS_PREDICTION_HEADER\s*$", text)
    league_hint = infer_league_from_filename(src.name)
    for chunk in blocks[1:]:
        head = chunk.strip()
        m_date = re.search(r"(?im)^\s*date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*$", head)
        m_match = re.search(r"(?im)^\s*matchup:\s*([A-Z0-9]+@[A-Z0-9]+)\s*$", head)
        m_p_home = re.search(r"(?im)^\s*p_home:\s*([0-9.]+)\s*$", head)
        m_winner = re.search(r"(?im)^\s*winner:\s*([A-Z0-9]+)\s*$", head)
        m_league = re.search(r"(?im)^\s*league:\s*([A-Za-z]+)\s*$", head)
        if not (m_date and m_match):
            continue
        date_str = m_date.group(1).strip()
        league = normalize_league(m_league.group(1)) if m_league else (league_hint or None)
        if not league:
            continue

        away_raw, home_raw = m_match.group(1).strip().upper().split("@", 1)
        away = normalize_team(away_raw, league)
        home = normalize_team(home_raw, league)
        if not away or not home:
            continue
        winner = normalize_team(m_winner.group(1), league) if m_winner else None
        p_home = None
        if m_p_home:
            try:
                p_home = clamp_prob(float(m_p_home.group(1)))
            except ValueError:
                p_home = None

        games.append(
            ParsedGame(
                date=date_str,
                league=league,
                away=away,
                home=home,
                winner=winner,
                p_true=None,
                p_home=p_home,
                model_label="",
                source_file=src,
                raw_header=None,
            )
        )
    return games


def call_llm_parser(text: str, model: str) -> List[ParsedGame]:
    if OpenAI is None:
        raise RuntimeError("openai package not available; cannot parse without HELIOS headers")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing for parsing")
    client = OpenAI(api_key=key)
    system = (
        "You are a strict JSON parser. Extract structured games from the text.\n"
        "Return JSON: {\"games\": [{\"date\": \"YYYY-MM-DD\", \"league\": \"nba|nhl|nfl\", "
        "\"away\": \"AAA\", \"home\": \"BBB\", \"winner\": \"AAA|BBB\", "
        "\"p_home\": 0.56 or null, \"p_true\": 0.64 or null, \"model_label\": \"Gemini_Scientist_v8.6\"}]}\n"
        "- Use league-standard team codes when possible.\n"
        "- winner must be exactly the away or home code.\n"
        "- p_home is the probability the home team wins.\n"
        "- If uncertain about probabilities, return null."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    payload = json.loads(content)
    games_raw = payload.get("games") or []
    parsed: List[ParsedGame] = []
    for g in games_raw:
        date_str = g.get("date") or ""
        league = normalize_league(g.get("league", ""))
        away = normalize_team(g.get("away", ""), league)
        home = normalize_team(g.get("home", ""), league)
        if not (date_str and league and away and home):
            continue
        winner_raw = g.get("winner")
        winner = normalize_team(winner_raw, league) if winner_raw else None
        p_home_val = g.get("p_home")
        p_home = None
        if p_home_val is not None:
            try:
                p_home = clamp_prob(float(p_home_val))
            except Exception:
                p_home = None
        p_true_val = g.get("p_true")
        p_true = None
        if p_true_val is not None:
            try:
                p_true = clamp_prob(float(p_true_val))
            except Exception:
                p_true = None
        model_label = g.get("model_label") or ""
        parsed.append(
            ParsedGame(
                date=date_str,
                league=league,
                away=away,
                home=home,
                winner=winner,
                p_true=p_true,
                p_home=p_home,
                model_label=model_label,
                source_file=Path(""),
            )
        )
    return parsed


def infer_model_label(file_name: str, override: Optional[str], existing: str) -> str:
    if override:
        return override
    if existing:
        return existing
    lowered = file_name.lower()
    if "grok" in lowered:
        return "Grok_Scientist_vX"
    if "gemini" in lowered:
        return "Gemini_Scientist_vX"
    if "gpt" in lowered or "openai" in lowered:
        return "GPT_Scientist_vX"
    return "Unknown_Model"


def model_family(label: str) -> Optional[str]:
    if not label:
        return None
    l = label.lower()
    for key, col in MODEL_FALLBACK.items():
        if key in l:
            return col
    return None


def render_header(game: ParsedGame) -> str:
    p_home = None
    if game.p_home is not None:
        p_home = round(clamp_prob(game.p_home), 4)
    elif game.p_true is not None and game.winner:
        winner_norm = normalize_team(game.winner, game.league)
        if winner_norm == game.home:
            p_home = round(clamp_prob(game.p_true), 4)
        elif winner_norm == game.away:
            p_home = round(clamp_prob(1.0 - game.p_true), 4)
    lines = [
        "HELIOS_PREDICTION_HEADER_START",
        f"Game: {game.date} {game.league.upper()} {game.matchup}",
        f"Model: {game.model_label}",
        "",
        "Prediction_Moneyline:",
        f"Winner: {game.winner or ''}",
        f"p_home: {'' if p_home is None else p_home}",
        f"p_true: {'' if game.p_true is None else round(game.p_true, 4)}",
        "ptcs: ",
        "HELIOS_PREDICTION_HEADER_END",
    ]
    return "\n".join(lines)


def ensure_daily_exists(date_str: str) -> None:
    ymd = datetime.strptime(date_str, "%Y-%m-%d").date().isoformat()
    ledger_path = DAILY_LEDGER_DIR / f"{ymd.replace('-', '')}_daily_game_ledger.csv"
    if ledger_path.exists():
        return
    cmd = ["python", "chimera_v2c/tools/ensure_daily_ledger.py", "--date", ymd]
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
    subprocess.run(cmd, check=True, env=env)


def infer_date_from_filename(file_name: str) -> Optional[str]:
    """
    Best-effort inference for raw report filenames like:
      - "12-13 grok NHL.txt"
      - "12_13 gemini NBA.txt"
    We assume the current year. This is only used as a safety fallback when
    parsed game dates don't match any existing daily ledger row but the
    filename-inferred date does.
    """
    m = re.match(r"\s*(\d{1,2})\s*[-_\./]\s*(\d{1,2})\b", file_name)
    if not m:
        return None
    month = int(m.group(1))
    day = int(m.group(2))
    year = datetime.now().year
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def ledger_has_matchup(ledger_path: Path, league: str, matchup: str) -> bool:
    if not ledger_path.exists():
        return False
    try:
        df = pd.read_csv(ledger_path)
    except Exception:
        return False
    if "matchup" not in df.columns or "league" not in df.columns:
        return False
    mask = (df["matchup"] == matchup) & (df["league"].astype(str).str.lower() == league.lower())
    return bool(mask.any())


def _parse_outcome_home_win(outcome: str) -> Optional[float]:
    return parse_home_win(outcome)


def update_daily_ledger(
    game: ParsedGame,
    apply: bool,
    force: bool,
) -> Tuple[Path, Optional[float], bool]:
    """
    Returns (ledger_path, p_home_to_write, wrote_flag)
    """
    ymd = datetime.strptime(game.date, "%Y-%m-%d").strftime("%Y%m%d")
    ledger_path = DAILY_LEDGER_DIR / f"{ymd}_daily_game_ledger.csv"
    if apply and not ledger_path.exists():
        ensure_daily_exists(game.date)

    locked_dates = load_locked_dates(LOCK_DIR)
    if ymd in locked_dates and not force:
        return ledger_path, None, False

    if not ledger_path.exists():
        return ledger_path, None, False

    df = pd.read_csv(ledger_path, dtype=str).fillna("")
    # Ensure required columns exist
    for col in ["v2c", "gemini", "grok", "gpt", "kalshi_mid", "actual_outcome"]:
        if col not in df.columns:
            df[col] = ""

    fam = model_family(game.model_label)
    if fam is None:
        return ledger_path, None, False

    # Convert probability to p_home
    p_home: Optional[float] = None
    if game.p_home is not None:
        p_home = round(clamp_prob(game.p_home), 2)
    elif game.p_true is not None and game.winner:
        winner_norm = normalize_team(game.winner, game.league)
        if winner_norm == game.home:
            p_home = game.p_true
        elif winner_norm == game.away:
            p_home = 1.0 - game.p_true
        if p_home is not None:
            p_home = round(clamp_prob(p_home), 2)

    # Find row
    mask = (df["matchup"] == game.matchup) & (df["league"].str.lower() == game.league.lower())
    if not mask.any():
        # append blank row
        new_row = {col: "" for col in df.columns}
        new_row.update({"date": game.date, "league": game.league, "matchup": game.matchup})
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        mask = (df["matchup"] == game.matchup) & (df["league"].str.lower() == game.league.lower())

    row_idx = df.index[mask][0]
    existing = str(df.at[row_idx, fam]).strip() if fam in df.columns else ""
    if existing and existing.upper() != MISSING_SENTINEL:
        if not force:
            return ledger_path, p_home, False
    if p_home is None:
        return ledger_path, None, False

    formatted = format_prob_cell(p_home, decimals=2, drop_leading_zero=True)
    if not formatted:
        return ledger_path, None, False
    df.at[row_idx, fam] = formatted

    # If this row is newly appended (or has empty sentinels), ensure other prob-like
    # columns are at least marked as missing so the ledger stays schema-consistent.
    for c in ("v2c", "gemini", "grok", "gpt", "kalshi_mid", "market_proxy", "moneypuck"):
        if c in df.columns and str(df.at[row_idx, c]).strip() == "":
            df.at[row_idx, c] = MISSING_SENTINEL

    # Append-only guard
    df = df.fillna("")
    if apply:
        original = pd.read_csv(ledger_path, dtype=str).fillna("")
        old_rows = original.fillna("").astype(str).to_dict("records")
        new_rows = df.fillna("").astype(str).to_dict("records")
        try:
            compute_append_only_diff(
                old_rows=old_rows,
                new_rows=new_rows,
                key_fields=["date", "matchup"],
                value_fields=[c for c in df.columns if c not in ("date", "matchup")],
                blank_sentinels={MISSING_SENTINEL},
            )
        except LedgerGuardError:
            if not force:
                raise
        # Write
        snapshot_file(ledger_path, DAILY_SNAPSHOT_DIR)
        write_csv(ledger_path, new_rows, fieldnames=list(df.columns))
    return ledger_path, p_home, apply


def write_canonical(game: ParsedGame, apply: bool, force: bool) -> Path:
    league_dir = CANONICAL_ROOT / game.league.upper()
    month_dir = league_dir / game.date[:7]
    month_dir.mkdir(parents=True, exist_ok=True)
    fam = model_family(game.model_label)
    family_token = fam or "unknown"
    matchup_token = f"{game.away.lower()}@{game.home.lower()}"
    fname = f"{game.date.replace('-', '')}_{game.league}_{matchup_token}_{family_token}.txt"
    out_path = month_dir / fname
    if out_path.exists() and not force:
        return out_path
    if apply:
        header = render_header(game)
        body = game.raw_header or ""
        content = header
        if body:
            content = f"{header}\n\n{body}"
        out_path.write_text(content, encoding="utf-8")
    return out_path


def move_raw(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    shutil.move(str(src), target)


def ingest_file(
    raw_path: Path,
    model_override: Optional[str],
    openai_model: str,
    apply: bool,
    force: bool,
) -> Dict:
    text = raw_path.read_text(encoding="utf-8")
    parsed = parse_helios_blocks(text, raw_path)
    if not parsed:
        parsed = parse_legacy_helios_prediction_headers(text, raw_path)
    if not parsed:
        parsed = parse_compact_prediction_headers(text, raw_path)
    used_llm = False
    if not parsed:
        try:
            parsed = call_llm_parser(text, model=openai_model)
            used_llm = True
        except Exception as exc:
            return {"file": raw_path, "error": str(exc), "parsed": [], "used_llm": False}

    games: List[ParsedGame] = []
    for g in parsed:
        g.model_label = infer_model_label(raw_path.name, model_override, g.model_label)
        fam = model_family(g.model_label)
        if fam is None:
            continue
        # If the parsed date doesn't map to any existing ledger row for this matchup,
        # but a filename-inferred date does, prefer the filename date.
        inferred_date = infer_date_from_filename(raw_path.name)
        if inferred_date:
            parsed_ymd = datetime.strptime(g.date, "%Y-%m-%d").strftime("%Y%m%d")
            inferred_ymd = datetime.strptime(inferred_date, "%Y-%m-%d").strftime("%Y%m%d")
            parsed_ledger = DAILY_LEDGER_DIR / f"{parsed_ymd}_daily_game_ledger.csv"
            inferred_ledger = DAILY_LEDGER_DIR / f"{inferred_ymd}_daily_game_ledger.csv"
            if not ledger_has_matchup(parsed_ledger, g.league, g.matchup) and ledger_has_matchup(
                inferred_ledger, g.league, g.matchup
            ):
                g.date = inferred_date
        games.append(g)

    results = []
    for g in games:
        ledger_path, p_home, wrote = update_daily_ledger(g, apply=apply, force=force)
        canonical_path = write_canonical(g, apply=apply, force=force)
        results.append(
            {
                "game": g,
                "ledger_path": ledger_path,
                "p_home_written": p_home,
                "ledger_updated": wrote,
                "canonical_path": canonical_path,
            }
        )
    return {"file": raw_path, "error": None, "parsed": results, "used_llm": used_llm, "raw_games": games}


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview-first specialist ingester for Chimera v2c.")
    ap.add_argument(
        "--raw-dir",
        default=str(RAW_DEFAULT),
        help="Directory containing raw specialist reports (default: reports/specialist_reports/raw).",
    )
    ap.add_argument("--apply", action="store_true", help="Apply changes (write canonical + ledger + move files).")
    ap.add_argument("--dry-run", action="store_true", help="Preview only (default).")
    ap.add_argument("--force", action="store_true", help="Allow edits to locked dates or overwriting non-blank cells.")
    ap.add_argument("--model-name", help="Optional model label override if filenames/headers are ambiguous.")
    ap.add_argument("--openai-model", default="gpt-4o-mini", help="OpenAI model for parsing (default: gpt-4o-mini).")
    args = ap.parse_args()

    load_env_from_env_list()

    if args.apply and args.dry_run:
        raise SystemExit("Specify either --apply or --dry-run, not both.")
    apply = bool(args.apply)
    dry_run = args.dry_run or not args.apply

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        if raw_dir == RAW_DEFAULT and RAW_LEGACY.exists():
            print(f"[warn] raw dir missing at {raw_dir}; using legacy location {RAW_LEGACY}")
            raw_dir = RAW_LEGACY
        else:
            raise SystemExit(f"[error] raw dir missing: {raw_dir}")

    raw_files = sorted(raw_dir.glob("*.txt"))
    if not raw_files:
        print(f"[info] no raw files found in {raw_dir}")
        return

    processed = []
    failed = []
    for raw_file in raw_files:
        result = ingest_file(
            raw_path=raw_file,
            model_override=args.model_name,
            openai_model=args.openai_model,
            apply=apply,
            force=args.force,
        )
        if result["error"]:
            failed.append(result)
        else:
            processed.append(result)

    for res in processed:
        print(f"\n=== {res['file']} ===")
        if res["used_llm"]:
            print("Parsed via OpenAI JSON parser.")
        for entry in res["parsed"]:
            g: ParsedGame = entry["game"]
            print(
                f"- {g.date} {g.league.upper()} {g.matchup} {g.model_label} "
                f"winner={g.winner} p_true={g.p_true}"
            )
            if entry["p_home_written"] is not None:
                print(f"  ledger: {entry['ledger_path']} -> {entry['p_home_written']} "
                      f"(updated={entry['ledger_updated']})")
            print(f"  canonical: {entry['canonical_path']}")

    if failed:
        print("\nFailed to parse:")
        for res in failed:
            print(f"- {res['file']}: {res['error']}")

    if apply:
        for res in processed:
            if res["error"]:
                continue
            move_raw(res["file"], ARCHIVE_PROCESSED)
        for res in failed:
            move_raw(res["file"], ARCHIVE_UNPARSED)
    else:
        print("\n[dry-run] No files written or moved.")


if __name__ == "__main__":
    main()
