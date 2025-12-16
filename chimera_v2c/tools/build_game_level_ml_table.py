"""
Build a single game-level ML table with one row per game.

Target schema (per row):
  - date
  - league
  - game_id      (AWAY@HOME)
  - home_team
  - away_team
  - p_home_gemini
  - p_home_grok
  - p_home_gpt
  - p_home_v2c
  - p_home_market
  - home_score
  - away_score
  - home_win     (1, 0, or 0.5 for pushes/ties)

Inputs (all read-only; existing output is used as a seed to stay append-only):
  - reports/specialist_performance/specialist_predictions_master.csv (primary)
  - reports/specialist_performance/specialist_manual_ledger_rebuilt.csv (or specialist_manual_ledger.csv)
  - reports/specialist_performance/v2c_master_ledger.csv (optional)
  - reports/specialist_performance/ledger_*.csv (optional league-specific snapshots)

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/build_game_level_ml_table.py

This reuses the existing game_level_ml_master.csv as a base (append-only)
and writes:
  reports/specialist_performance/game_level_ml_master.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

from chimera_v2c.lib import team_mapper


OUT_PATH_DEFAULT = Path("reports/specialist_performance/game_level_ml_master.csv")


@dataclass
class GameRow:
    date: str
    league: str
    game_id: str  # AWAY@HOME
    home_team: str
    away_team: str
    p_home_gemini: Optional[float] = None
    p_home_grok: Optional[float] = None
    p_home_gpt: Optional[float] = None
    p_home_v2c: Optional[float] = None
    p_home_market: Optional[float] = None
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    home_win: Optional[float] = None

    def to_dict(self) -> Dict[str, str]:
        out = asdict(self)
        # Convert None to "" for CSV, keep floats as strings
        for k, v in list(out.items()):
            if v is None:
                out[k] = ""
            else:
                out[k] = str(v)
        return out


def parse_float(val: str) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none"}:
        return None
    try:
        return float(s)
    except Exception:
        return None


def parse_home_win(val: str) -> Optional[float]:
    """
    Normalize home_win as float: 1.0, 0.0, or 0.5.
    Accepts bool-ish strings or numeric.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none"}:
        return None
    # Common encodings: True/False, 1/0, 1.0/0.0, 0.5
    lower = s.lower()
    if lower in {"true", "t", "yes", "y"}:
        return 1.0
    if lower in {"false", "f", "no", "n"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return None


def parse_actual_score(score_str: str, away_team: str, home_team: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse an 'actual_score' string like 'GSW 96-110 MIA' into (home_score, away_score).

    We assume the string is either:
      AWAY 96-110 HOME
    or (less commonly)
      HOME 110-96 AWAY
    and use the matchup to disambiguate.
    """
    if not score_str:
        return None, None
    s = score_str.strip()
    # Simple pattern: TEAM1 96-110 TEAM2
    m = re.match(r"([A-Za-z]{2,5})\s+(\d+)\s*-\s*(\d+)\s+([A-Za-z]{2,5})", s)
    if not m:
        return None, None
    team1, s1, s2, team2 = m.groups()
    team1 = team1.upper()
    team2 = team2.upper()
    score1 = parse_float(s1)
    score2 = parse_float(s2)
    if score1 is None or score2 is None:
        return None, None

    away = away_team.upper()
    home = home_team.upper()

    if team1 == away and team2 == home:
        return score2, score1  # home_score, away_score
    if team1 == home and team2 == away:
        return score1, score2
    # Fallback: don't trust the mapping if team codes don't align.
    return None, None


def resolve_team_from_text(text: str, league: str) -> Optional[str]:
    """
    Try to map a freeform team string (e.g., 'BOSTON CELTICS', 'GOLDEN STATE')
    to a league code using team_mapper with a few fallbacks.
    """
    if not text:
        return None
    s = text.strip()
    league_lower = league.lower()
    candidates = [s]
    parts = s.split()
    if len(parts) >= 2:
        candidates.append(" ".join(parts[:2]))
    if parts:
        candidates.append(parts[0])
        candidates.append(parts[-1])
    for cand in candidates:
        code = team_mapper.normalize_team_code(cand, league_lower)
        if code:
            return code
    return None


def ensure_game(
    games: Dict[Tuple[str, str, str], GameRow],
    date: str,
    league: str,
    game_id: str,
    home_team: str,
    away_team: str,
) -> GameRow:
    key = (date, league.lower(), game_id)
    if key in games:
        row = games[key]
        # If teams are missing in existing row, fill them
        if not row.home_team and home_team:
            row.home_team = home_team
        if not row.away_team and away_team:
            row.away_team = away_team
        return row
    row = GameRow(
        date=date,
        league=league.lower(),
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
    )
    games[key] = row
    return row


def infer_league(home_raw: str, away_raw: str) -> Optional[str]:
    for league in ("nba", "nfl", "nhl"):
        if team_mapper.normalize_team_code(home_raw, league) and team_mapper.normalize_team_code(away_raw, league):
            return league
    return None


def normalize_team(team: str, league: str) -> Optional[str]:
    if not team or not league:
        return None
    league_lower = league.lower()
    team_upper = team.strip().upper()
    alias_map = {
        "nba": {
            "BRK": "BKN",
            "NO": "NOP",
            "GS": "GSW",
            "NY": "NYK",
            "SA": "SAS",
            "PHO": "PHX",
            "WAS": "WSH",
        },
        "nhl": {
            "LA": "LAK",
            "NJ": "NJD",
            "TB": "TBL",
            "SJ": "SJS",
            "MON": "MTL",
            "CLB": "CBJ",
        },
        "nfl": {
            "JAC": "JAX",
            "LA": "LAR",
            "SD": "LAC",
        },
    }
    team_upper = alias_map.get(league_lower, {}).get(team_upper, team_upper)
    return team_mapper.normalize_team_code(team_upper, league_lower) or team_upper


def compute_p_home_from_plan(plan: Dict, league: str, home: str, away: str) -> Optional[float]:
    # 1) Prefer explicitly home-based probabilities if present.
    direct_home = plan.get("home_p_v2c")
    if direct_home is not None:
        return parse_float(direct_home)

    # In v2c, p_final is defined as **home win probability** (see docs).
    p_home = parse_float(plan.get("p_final"))
    if p_home is not None:
        return p_home

    # 2) Fallback: use a yes-side probability and map it to home.
    p_yes = parse_float(plan.get("p_yes_selected") or plan.get("p_yes"))
    if p_yes is None:
        return None

    yes_team = plan.get("yes_team") or (plan.get("market") or {}).get("yes_team")
    if yes_team:
        yes_norm = team_mapper.normalize_team_code(yes_team, league)
        if yes_norm == home:
            return p_yes
        if yes_norm == away:
            return 1.0 - p_yes
        return None

    # If no yes_team is provided, treat p_yes as home probability.
    return p_yes


def _normalize_price(val: Optional[float]) -> Optional[float]:
    price = parse_float(val)
    if price is None:
        return None
    return price / 100.0 if price > 1 else price


def compute_market_prob(plan: Dict, league: str, home: str, away: str) -> Optional[float]:
    # Proprietary format: market_mid as home prob
    if plan.get("market_mid") is not None:
        mid = parse_float(plan.get("market_mid"))
        if mid is not None:
            return mid

    market = plan.get("market") or {}
    if not isinstance(market, dict):
        return None

    yes_team = market.get("yes_team") or plan.get("yes_team")
    bid = _normalize_price(market.get("yes_bid"))
    ask = _normalize_price(market.get("yes_ask"))
    mid = None
    if bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    elif bid is not None:
        mid = bid
    elif ask is not None:
        mid = ask

    if mid is None:
        return None

    if yes_team:
        yes_norm = team_mapper.normalize_team_code(yes_team, league)
        if yes_norm == home:
            return mid
        if yes_norm == away:
            return 1.0 - mid
        return None
    # If no yes_team, assume mid is home probability
    return mid


def load_v2c_plan_jsons(
    games: Dict[Tuple[str, str, str], GameRow],
    paths: Tuple[Path, ...],
) -> None:
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue

        date = None
        m = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
        if m:
            date = m.group(1)
        if isinstance(data, dict) and data.get("date"):
            date = data["date"]
        if not isinstance(data, dict):
            continue

        plans = data.get("plans") or data.get("games")
        if not plans:
            continue

        # Derive default league from filename if possible
        path_lower = path.name.lower()
        default_league = None
        if "nfl" in path_lower:
            default_league = "nfl"
        elif "nhl" in path_lower:
            default_league = "nhl"
        elif "nba" in path_lower:
            default_league = "nba"

        for plan in plans:
            home_raw = plan.get("home")
            away_raw = plan.get("away")
            key = plan.get("key") or plan.get("game_id")
            if not key and home_raw and away_raw:
                key = f"{away_raw}@{home_raw}"
            if not key or "@" not in key or not date:
                continue

            league = plan.get("league") or default_league or infer_league(home_raw, away_raw)
            if not league:
                continue

            away_norm = normalize_team(away_raw, league)
            home_norm = normalize_team(home_raw, league)
            game_id = f"{away_norm}@{home_norm}"

            gr = ensure_game(games, date, league, game_id, home_norm, away_norm)

            p_home = compute_p_home_from_plan(plan, league, home_norm, away_norm)
            if p_home is not None and gr.p_home_v2c is None:
                gr.p_home_v2c = p_home

            p_market = compute_market_prob(plan, league, home_norm, away_norm)
            if p_market is not None and gr.p_home_market is None:
                gr.p_home_market = p_market


def seed_from_existing_ledger(
    games: Dict[Tuple[str, str, str], GameRow],
    path: Path,
) -> None:
    """
    Seed from an existing game-level ledger so rebuilds never drop rows or
    overwrite populated cells.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = (row.get("date") or "").strip()
            league = (row.get("league") or "").strip().lower()
            game_id = (row.get("game_id") or "").strip()
            home_team = (row.get("home_team") or "").strip()
            away_team = (row.get("away_team") or "").strip()

            # Normalize teams/game_id to reduce duplicate rows (e.g., BRK vs BKN).
            norm_home = home_team
            norm_away = away_team
            raw_game_id = game_id
            if "@" in raw_game_id and league:
                raw_away, raw_home = raw_game_id.split("@", 1)
                norm_home = team_mapper.normalize_team_code(home_team or raw_home, league) or (home_team or raw_home).strip().upper()
                norm_away = team_mapper.normalize_team_code(away_team or raw_away, league) or (away_team or raw_away).strip().upper()
                game_id = f"{norm_away}@{norm_home}"
            else:
                if not norm_home and not norm_away and "@" in raw_game_id:
                    raw_away, raw_home = raw_game_id.split("@", 1)
                    norm_home = raw_home.strip().upper()
                    norm_away = raw_away.strip().upper()
                    game_id = raw_game_id

            home_team = norm_home
            away_team = norm_away
            if not date or not league or not game_id:
                continue
            gr = ensure_game(games, date, league, game_id, home_team, away_team)

            def maybe_set(attr: str, value) -> None:
                if value is None:
                    return
                if getattr(gr, attr) is None:
                    setattr(gr, attr, value)

            maybe_set("p_home_gemini", parse_float(row.get("p_home_gemini")))
            maybe_set("p_home_grok", parse_float(row.get("p_home_grok")))
            maybe_set("p_home_gpt", parse_float(row.get("p_home_gpt")))
            maybe_set("p_home_v2c", parse_float(row.get("p_home_v2c")))
            maybe_set("p_home_market", parse_float(row.get("p_home_market")))
            maybe_set("home_score", parse_float(row.get("home_score")))
            maybe_set("away_score", parse_float(row.get("away_score")))
            maybe_set("home_win", parse_home_win(row.get("home_win")))


def load_predictions_master(
    games: Dict[Tuple[str, str, str], GameRow],
    path: Path,
) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.lower() for c in reader.fieldnames or []]
        for row in reader:
            date = (row.get("date") or "").strip()
            league = (row.get("league") or "").strip().lower()
            matchup = (row.get("matchup") or row.get("game_id") or "").strip()
            home_team = (row.get("home_team") or "").strip()
            away_team = (row.get("away_team") or "").strip()
            if not date or not league or not matchup or "@" not in matchup:
                continue
            norm_home = normalize_team(home_team, league) or home_team
            norm_away = normalize_team(away_team, league) or away_team
            game_id = f"{norm_away}@{norm_home}"
            gr = ensure_game(games, date, league, game_id, norm_home, norm_away)

            if {"model_name", "bet_target", "p_true"} <= set(fieldnames):
                model_name = (row.get("model_name") or "").lower()
                bet_target = (row.get("bet_target") or "").strip()
                p_true = parse_float(row.get("p_true"))
                if not bet_target or p_true is None:
                    continue
                target_norm = resolve_team_from_text(bet_target, league) or bet_target.strip().upper()
                p_home = None
                if target_norm == norm_home:
                    p_home = p_true
                elif target_norm == norm_away:
                    p_home = 1.0 - p_true
                if p_home is None:
                    continue
                if "gemini" in model_name and gr.p_home_gemini is None:
                    gr.p_home_gemini = p_home
                elif "grok" in model_name and gr.p_home_grok is None:
                    gr.p_home_grok = p_home
                elif "gpt" in model_name and gr.p_home_gpt is None:
                    gr.p_home_gpt = p_home
                continue

            # Model probabilities are already home-based; never overwrite populated cells.
            gem = parse_float(row.get("Gemini"))
            if gem is not None and gr.p_home_gemini is None:
                gr.p_home_gemini = gem
            grok = parse_float(row.get("Grok"))
            if grok is not None and gr.p_home_grok is None:
                gr.p_home_grok = grok
            gpt = parse_float(row.get("GPT"))
            if gpt is not None and gr.p_home_gpt is None:
                gr.p_home_gpt = gpt
            v2c = parse_float(row.get("v2c"))
            if v2c is not None and gr.p_home_v2c is None:
                gr.p_home_v2c = v2c
            mkt = parse_float(row.get("Kalshi"))
            if mkt is not None and gr.p_home_market is None:
                gr.p_home_market = mkt

            hs = parse_float(row.get("home_score"))
            if hs is not None and gr.home_score is None:
                gr.home_score = hs
            as_ = parse_float(row.get("away_score"))
            if as_ is not None and gr.away_score is None:
                gr.away_score = as_
            hw = parse_home_win(row.get("home_win"))
            if hw is not None and gr.home_win is None:
                gr.home_win = hw


def load_v2c_ledger_like(
    games: Dict[Tuple[str, str, str], GameRow],
    path: Path,
) -> None:
    """
    Load data from v2c_master_ledger or league-specific ledger files.
    Schema:
      date,league,game_id,home_team,away_team,
      p_home_gemini,p_home_grok,p_home_gpt,p_home_v2c,p_home_market,
      home_score,away_score,home_win
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = (row.get("date") or "").strip()
            league = (row.get("league") or "").strip().lower()
            game_id = (row.get("game_id") or row.get("matchup") or "").strip()
            home_team = (row.get("home_team") or "").strip()
            away_team = (row.get("away_team") or "").strip()
            if not date or not league or not game_id or "@" not in game_id:
                continue
            norm_home = normalize_team(home_team, league) or home_team
            norm_away = normalize_team(away_team, league) or away_team
            game_id = f"{norm_away}@{norm_home}"
            gr = ensure_game(games, date, league, game_id, norm_home, norm_away)

            # Only fill if not already set from predictions_master
            if gr.p_home_gemini is None:
                gr.p_home_gemini = parse_float(row.get("p_home_gemini"))
            if gr.p_home_grok is None:
                gr.p_home_grok = parse_float(row.get("p_home_grok"))
            if gr.p_home_gpt is None:
                gr.p_home_gpt = parse_float(row.get("p_home_gpt"))
            if gr.p_home_v2c is None:
                gr.p_home_v2c = parse_float(row.get("p_home_v2c"))
            if gr.p_home_market is None:
                gr.p_home_market = parse_float(row.get("p_home_market"))

            # Scores and outcome: prefer existing, otherwise fill
            if gr.home_score is None:
                gr.home_score = parse_float(row.get("home_score"))
            if gr.away_score is None:
                gr.away_score = parse_float(row.get("away_score"))
            if gr.home_win is None:
                gr.home_win = parse_home_win(row.get("home_win"))


def load_manual_ledger(
    games: Dict[Tuple[str, str, str], GameRow],
    path: Path,
) -> None:
    """
    Use the specialist manual ledger to backfill missing model probabilities
    and scores for games that are present there but not in the predictions tables.

    We only consider ML predictions and normalize them into home-side p_true.
    """
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            league = (row.get("league") or "").strip().lower()
            game_date = (row.get("game_date") or row.get("date") or "").strip()
            matchup = (row.get("matchup") or "").strip()
            model_family = (row.get("model_family") or "").strip()
            ml_team = (row.get("ml_team") or "").strip().upper()
            ml_p_true = parse_float(row.get("ml_p_true") or row.get("p_true"))
            if not league or not game_date or not matchup or "@" not in matchup:
                continue
            if ml_p_true is None:
                continue
            away_team, home_team = matchup.split("@", 1)
            away_team = normalize_team(away_team.strip(), league) or away_team.strip().upper()
            home_team = normalize_team(home_team.strip(), league) or home_team.strip().upper()

            # Normalize model family to a slot
            mf_lower = model_family.lower()
            if "gemini" in mf_lower:
                slot = "gemini"
            elif "grok" in mf_lower:
                slot = "grok"
            elif "gpt" in mf_lower:
                slot = "gpt"
            else:
                continue

            # Map to home probability
            if ml_team == home_team:
                p_home = ml_p_true
            elif ml_team == away_team:
                p_home = 1.0 - ml_p_true
            else:
                # If team codes don't match matchup, skip.
                continue

            game_id = f"{away_team}@{home_team}"
            gr = ensure_game(games, game_date, league, game_id, home_team, away_team)

            if slot == "gemini" and gr.p_home_gemini is None:
                gr.p_home_gemini = p_home
            elif slot == "grok" and gr.p_home_grok is None:
                gr.p_home_grok = p_home
            elif slot == "gpt" and gr.p_home_gpt is None:
                gr.p_home_gpt = p_home

            # Backfill scores if missing and ledger has an actual_score
            if gr.home_score is None or gr.away_score is None or gr.home_win is None:
                actual_score = (row.get("actual_score") or "").strip()
                hs, as_ = parse_actual_score(actual_score, away_team, home_team)
                if hs is not None and as_ is not None:
                    if gr.home_score is None:
                        gr.home_score = hs
                    if gr.away_score is None:
                        gr.away_score = as_
                    if gr.home_win is None:
                        if hs > as_:
                            gr.home_win = 1.0
                        elif hs < as_:
                            gr.home_win = 0.0
                        else:
                            gr.home_win = 0.5


def build_game_level_ml_table(out_path: Path) -> None:
    games: Dict[Tuple[str, str, str], GameRow] = {}

    base_dir = Path("reports/specialist_performance")
    archive_dir = base_dir / "archive_old_ledgers"
    ok_base_dir = Path("ok/reports/specialist_performance")
    ok_archive_dir = ok_base_dir / "archive_old_ledgers"

    # 0) Seed from existing output to avoid clobbering historical rows.
    seed_from_existing_ledger(games, out_path)
    seeded_rows = len(games)

    # 1) seed from specialist_predictions_* (canonical + ok copy + dated files)
    prediction_sources = [
        base_dir / "specialist_predictions_master.csv",
        archive_dir / "specialist_predictions_master.csv",
        ok_base_dir / "specialist_predictions_master.csv",
    ]
    dated_prediction_files = list(base_dir.glob("specialist_predictions_2025-12-0*.csv")) + list(ok_base_dir.glob("specialist_predictions_2025-12-0*.csv"))
    prediction_sources.extend(dated_prediction_files)
    seen_pred_paths = set()
    for p in prediction_sources:
        if p.exists() and p not in seen_pred_paths:
            load_predictions_master(games, p)
            seen_pred_paths.add(p)

    # 2) augment from v2c_master_ledger and league-specific ledgers
    v2c_sources = [
        base_dir / "v2c_master_ledger.csv",
        archive_dir / "v2c_master_ledger.csv",
        ok_base_dir / "v2c_master_ledger.csv",
        ok_archive_dir / "v2c_master_ledger.csv",
    ]
    for name in [
        "ledger_nba_2025-12-05_to_2025-12-07.csv",
        "ledger_nhl_2025-12-05_to_2025-12-07.csv",
        "ledger_nfl_2025-12-05_to_2025-12-07.csv",
    ]:
        v2c_sources.append(base_dir / name)
        v2c_sources.append(archive_dir / name)
        v2c_sources.append(ok_base_dir / name)
        v2c_sources.append(ok_archive_dir / name)
    seen_v2c_paths = set()
    for p in v2c_sources:
        if p.exists() and p not in seen_v2c_paths:
            load_v2c_ledger_like(games, p)
            seen_v2c_paths.add(p)

    # 2b) overlay v2c plan logs (append-only fill for p_home_v2c and market)
    logs_dir = Path("reports/execution_logs")
    ok_logs_dir = Path("ok/reports/execution_logs")
    extra_plan_dir = Path("ok/chimera_v2c/data")
    plan_paths = []
    if logs_dir.exists():
        plan_paths.extend(sorted(logs_dir.glob("*plan*.json")))
        plan_paths.extend(sorted(logs_dir.glob("plan_2025-12-08_proprietary.json")))
    if ok_logs_dir.exists():
        plan_paths.extend(sorted(ok_logs_dir.glob("*plan*.json")))
    if extra_plan_dir.exists():
        plan_paths.extend(sorted(extra_plan_dir.glob("plan_2025-12-0*.json")))
    load_v2c_plan_jsons(games, tuple(plan_paths))

    # 3) backfill from manual ledger (rebuilt if present)
    manual_rebuilt = base_dir / "specialist_manual_ledger_rebuilt.csv"
    manual_default = base_dir / "specialist_manual_ledger.csv"
    manual_ok = ok_base_dir / "specialist_manual_ledger.csv"
    manual_ok_rebuilt = ok_base_dir / "specialist_manual_ledger_rebuilt.csv"
    for manual in [manual_rebuilt, manual_default, manual_ok_rebuilt, manual_ok]:
        if manual.exists():
            load_manual_ledger(games, manual)

    # Merge duplicates by (date, game_id) across leagues and prefer inferred league/non-empty fields.
    merged: Dict[Tuple[str, str], GameRow] = {}
    for row in games.values():
        key = (row.date, row.game_id)
        inferred_league = row.league or infer_league(row.home_team, row.away_team)
        if key not in merged:
            new_row = GameRow(
                date=row.date,
                league=inferred_league,
                game_id=row.game_id,
                home_team=row.home_team,
                away_team=row.away_team,
                p_home_gemini=row.p_home_gemini,
                p_home_grok=row.p_home_grok,
                p_home_gpt=row.p_home_gpt,
                p_home_v2c=row.p_home_v2c,
                p_home_market=row.p_home_market,
                home_score=row.home_score,
                away_score=row.away_score,
                home_win=row.home_win,
            )
            merged[key] = new_row
            continue
        dst = merged[key]
        if inferred_league and dst.league != inferred_league:
            dst.league = inferred_league
        for attr in [
            "p_home_gemini",
            "p_home_grok",
            "p_home_gpt",
            "p_home_v2c",
            "p_home_market",
            "home_score",
            "away_score",
            "home_win",
        ]:
            if getattr(dst, attr) is None and getattr(row, attr) is not None:
                setattr(dst, attr, getattr(row, attr))

    out_rows = list(merged.values())
    # Sort and write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Temporarily make the ledger writable (if read-only), then restore to read-only after write.
    original_mode = None
    if out_path.exists():
        original_mode = out_path.stat().st_mode
        try:
            out_path.chmod(original_mode | 0o200)  # add owner write
        except Exception:
            pass
    fieldnames = [
        "date",
        "league",
        "game_id",
        "home_team",
        "away_team",
        "p_home_gemini",
        "p_home_grok",
        "p_home_gpt",
        "p_home_v2c",
        "p_home_market",
        "home_score",
        "away_score",
        "home_win",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(out_rows, key=lambda r: (r.date, r.league, r.game_id)):
            writer.writerow(row.to_dict())
    try:
        out_path.chmod(0o444)  # lock down to read-only for everyone
    except Exception:
        if original_mode is not None:
            try:
                out_path.chmod(original_mode)
            except Exception:
                pass

    # Basic summary
    total_games = len(out_rows)
    have_gemini = sum(1 for r in out_rows if r.p_home_gemini is not None)
    have_grok = sum(1 for r in out_rows if r.p_home_grok is not None)
    have_gpt = sum(1 for r in out_rows if r.p_home_gpt is not None)
    have_v2c = sum(1 for r in out_rows if r.p_home_v2c is not None)
    have_mkt = sum(1 for r in out_rows if r.p_home_market is not None)
    print(f"[info] seeded from existing ledger: {seeded_rows}")
    print(f"[info] games total: {total_games}")
    print(f"[info] games with Gemini p_home: {have_gemini}")
    print(f"[info] games with Grok p_home: {have_grok}")
    print(f"[info] games with GPT p_home: {have_gpt}")
    print(f"[info] games with v2c p_home: {have_v2c}")
    print(f"[info] games with market p_home: {have_mkt}")
    print(f"[info] wrote game-level table: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build 1-row-per-game ML table from existing specialist ledgers.")
    ap.add_argument(
        "--out",
        default=str(OUT_PATH_DEFAULT),
        help=f"Output CSV path (default: {OUT_PATH_DEFAULT})",
    )
    args = ap.parse_args()
    out_path = Path(args.out)
    build_game_level_ml_table(out_path)


if __name__ == "__main__":
    main()
