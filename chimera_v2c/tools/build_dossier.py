"""
Build a structured game dossier for LLM input.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/build_dossier.py --league nba --date YYYY-MM-DD --game AWAY@HOME --out reports/dossiers/...
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import pandas as pd
from chimera_v2c.lib import espn_schedule
from chimera_v2c.lib import team_mapper
from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src import market_linker
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats
from nba_api.stats.static import players as static_players
from chimera_v2c.lib import nhl_scoreboard

DB_PATH = Path("chimera_v2c/data/chimera.db")
RATINGS_PATH = Path("chimera_v2c/data/team_ratings.json")
FACTORS_PATH = Path("chimera_v2c/data/team_four_factors.json")
INJURY_PATH = Path("chimera_v2c/data/injury_adjustments.json")
RAW_INJURY_PATH = Path("chimera_v2c/data/raw_injuries.json")
ROSTER_CACHE: Dict[str, Any] = {}
PLAYER_STATS_CACHE: Dict[str, Optional[pd.DataFrame]] = {}
PLAYER_META: Dict[int, Dict[str, str]] = {}
TEAM_EFF_CACHE: Dict[str, Dict[str, Any]] = {}
NHL_RATINGS_PATH = Path("chimera_v2c/data/team_ratings_nhl.json")
TEAM_CODE_TO_ID = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "BKN": 1610612751,
    "CHA": 1610612766,
    "CHI": 1610612741,
    "CLE": 1610612739,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "DET": 1610612765,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "IND": 1610612754,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MEM": 1610612763,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "NOP": 1610612740,
    "NYK": 1610612752,
    "OKC": 1610612760,
    "ORL": 1610612753,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "WAS": 1610612764,
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_ratings_for_league(league: str, cfg: V2CConfig) -> Dict[str, Any]:
    # Allow league-specific ratings path if available
    path = RATINGS_PATH
    if league == "nhl" and NHL_RATINGS_PATH.exists():
        path = NHL_RATINGS_PATH
    if cfg.paths and cfg.paths.get("ratings"):
        path = Path(cfg.paths["ratings"])
    return load_json(path)


def fetch_market_probs(matchup: str, league: str, cfg: V2CConfig) -> Tuple[Optional[float], Optional[float], str, str]:
    away, home = matchup.split("@")
    source = "fallback"
    quality = "low"
    date_str = getattr(cfg, "date_override", datetime.utcnow().date().isoformat())
    target_date = datetime.fromisoformat(date_str).date()
    # Try Kalshi markets first
    try:
        matchups = market_linker.fetch_matchups(league, target_date)
        markets = market_linker.fetch_markets(
            league,
            cfg.series_ticker,
            use_private=bool(cfg.execution.get("use_private", False)),
            target_date=target_date,
        )
        market_map = market_linker.match_markets_to_games(matchups, markets)
        key = f"{away}@{home}"
        if key in market_map:
            # pick home yes if present
            market_dict = market_map[key]
            home_mq = market_dict.get(home)
            away_mq = market_dict.get(away)
            if home_mq and home_mq.mid is not None:
                source = "kalshi"
                quality = "high"
                p_home = home_mq.mid
                p_away = 1 - p_home if away_mq is None or away_mq.mid is None else away_mq.mid
                print(f"[info] market source=kalshi for {matchup} ({target_date}): home_prob={p_home:.3f}")
                return p_home, p_away, source, quality
            if away_mq and away_mq.mid is not None:
                source = "kalshi"
                quality = "high"
                p_away = away_mq.mid
                p_home = 1 - p_away
                print(f"[info] market source=kalshi (away leg) for {matchup} ({target_date}): home_prob={p_home:.3f}")
                return p_home, p_away, source, quality
    except Exception:
        print(f"[warn] Kalshi market lookup failed for {matchup} on {target_date}; falling back")
    # Fallback: ESPN probability if present
    try:
        sb = espn_schedule.get_scoreboard(league, target_date)
        for ev in sb.get("events", []):
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            teams = []
            for c in comp.get("competitors", []):
                t = c.get("team") or {}
                code = team_mapper.normalize_team_code(t.get("abbreviation"), league)
                teams.append((code, c))
            if len(teams) != 2:
                continue
            codes = {teams[0][0], teams[1][0]}
            if {away, home} != codes:
                continue
            home_prob = None
            away_prob = None
            for code, comp_entry in teams:
                prob = comp_entry.get("probability")
                if prob is not None:
                    if code == home:
                        home_prob = prob
                    else:
                        away_prob = prob
            if home_prob is None and away_prob is None:
                continue
            if home_prob is not None and away_prob is None:
                away_prob = 1 - home_prob
            if away_prob is not None and home_prob is None:
                home_prob = 1 - away_prob
            source = "espn_prob"
            quality = "medium"
            print(f"[info] market source=espn for {matchup} ({target_date}): home_prob={home_prob:.3f}")
            return home_prob, away_prob, source, quality
    except Exception:
        print(f"[warn] ESPN probability lookup failed for {matchup} on {target_date}; falling back to 0.5/0.5")
    print(f"[info] market source=fallback for {matchup} ({target_date}): home_prob=0.500")
    return 0.5, 0.5, source, quality


def compute_record(conn: sqlite3.Connection, team: str, season_start: str, cutoff_date: str) -> Dict[str, Any]:
    cur = conn.cursor()
    def _count(where: str, params: tuple) -> Tuple[int,int]:
        wins = cur.execute(
            f"""
            SELECT COUNT(*) FROM games
            WHERE game_date >= ? AND game_date < ? AND {where}
            """,
            params,
        ).fetchone()[0]
        losses = cur.execute(
            f"""
            SELECT COUNT(*) FROM games
            WHERE game_date >= ? AND game_date < ? AND {where.replace('home_score > away_score','home_score < away_score').replace('away_score > home_score','away_score < home_score')}
            """,
            params,
        ).fetchone()[0]
        return wins, losses
    w_all, l_all = _count(
        "(home_team = ? AND home_score > away_score) OR (away_team = ? AND away_score > home_score)",
        (season_start, cutoff_date, team, team),
    )
    w_home, l_home = _count(
        "home_team = ? AND home_score > away_score",
        (season_start, cutoff_date, team),
    )
    # last 10
    rows = cur.execute(
        """
        SELECT home_team, away_team, home_score, away_score
        FROM games
        WHERE (home_team = ? OR away_team = ?) AND home_score IS NOT NULL AND away_score IS NOT NULL AND game_date >= ? AND game_date < ?
        ORDER BY game_date DESC, game_id DESC
        LIMIT 10
        """,
        (team, team, season_start, cutoff_date),
    ).fetchall()
    last10 = []
    for h, a, hs, ascore in rows:
        win = (h == team and hs > ascore) or (a == team and ascore > hs)
        last10.append("W" if win else "L")
    streak = ""
    if last10:
        first = last10[0]
        count = 1
        for r in last10[1:]:
            if r == first:
                count += 1
            else:
                break
        streak = f"{first}{count}"
    return {
        "overall": f"{w_all}-{l_all}",
        "home": f"{w_home}-{l_home}",
        "last_10": f"{last10.count('W')}-{last10.count('L')}",
        "streak": streak,
        "last_5": last10[:5][::-1]  # chronological order
    }


def load_team_stats(db: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT DISTINCT team_id, off_rating, def_rating, pace, efg_pct, tov_pct, orb_pct, ft_rate
        FROM team_stats
        """
    ).fetchall()
    conn.close()
    out: Dict[str, Dict[str, Any]] = {}
    code_to_teamid: Dict[str, str] = {}
    for tid, ortg, drtg, pace, efg, tov, orb, ftr in rows:
        code = None
        # team_id is stored as little-endian int bytes
        try:
            num = int.from_bytes(tid, "little")
            code = team_mapper.normalize_team_code(str(num), "nba")
            if code:
                code_to_teamid[code] = str(num)
        except Exception:
            code = team_mapper.normalize_team_code(tid, "nba")
        if not code:
            continue
        out[code] = {
            "off_rating": ortg,
            "def_rating": drtg,
            "pace": pace,
            "efg_pct": efg,
            "tov_pct": tov,
            "orb_pct": orb,
            "ft_rate": ftr,
        }
    return out, code_to_teamid


def fetch_team_efficiency(season_label: str) -> Dict[str, Dict[str, Any]]:
    if season_label in TEAM_EFF_CACHE:
        return TEAM_EFF_CACHE[season_label]
    eff: Dict[str, Dict[str, Any]] = {}
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(
            season=season_label,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Advanced",
            season_type_all_star="Regular Season",
        ).get_data_frames()[0]
        id_to_code = {v: k for k, v in TEAM_CODE_TO_ID.items()}
        for _, row in df.iterrows():
            code = team_mapper.normalize_team_code(
                row.get("TEAM_ABBREVIATION") or row.get("TEAM_NAME"), "nba"
            )
            if not code:
                tid = row.get("TEAM_ID")
                try:
                    code = id_to_code.get(int(tid))
                except Exception:
                    code = None
            if not code:
                continue
            eff[code] = {
                "off_rating": row.get("OFF_RATING"),
                "def_rating": row.get("DEF_RATING"),
                "pace": row.get("PACE"),
            }
    except Exception as exc:
        print(f"[warn] failed to fetch team efficiency stats for season {season_label}: {exc}")
    TEAM_EFF_CACHE[season_label] = eff
    return eff


def season_start_for(date_str: str) -> str:
    dt = datetime.fromisoformat(date_str)
    year = dt.year
    # Approx: NBA season starts Oct 1 of year (if date before Aug, use previous year)
    if dt.month < 8:
        year -= 1
    return f"{year}-10-01"


def season_label_for(date_str: str) -> str:
    start = season_start_for(date_str)
    start_year = int(start[:4])
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def load_player_meta() -> None:
    global PLAYER_META
    if PLAYER_META:
        return
    try:
        for p in static_players.get_players():
            pid = p.get("id")
            if pid is None:
                continue
            PLAYER_META[int(pid)] = {
                "name": p.get("full_name") or p.get("display_first_last") or p.get("display_last_comma_first") or "",
                "position": p.get("position") or "",
            }
    except Exception:
        PLAYER_META = {}


def get_player_stats_df(season_label: str) -> Optional[pd.DataFrame]:
    if season_label in PLAYER_STATS_CACHE:
        return PLAYER_STATS_CACHE[season_label]
    try:
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_label,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
        ).get_data_frames()[0]
        PLAYER_STATS_CACHE[season_label] = df
        return df
    except Exception as exc:
        print(f"[warn] failed to fetch league-wide player stats for season {season_label}: {exc}")
        PLAYER_STATS_CACHE[season_label] = None
        return None


def _role_from_position(pos: str, pts: Optional[float], ast: Optional[float], reb: Optional[float]) -> str:
    p = (pos or "").upper()
    if p.startswith("G"):
        return "guard"
    if p.startswith("F"):
        return "wing"
    if p.startswith("C"):
        return "big"
    # Simple stat-based heuristics when position missing
    try:
        if pts is not None and pts >= 24:
            return "primary_scorer"
        if ast is not None and ast >= 6:
            return "primary_creator"
        if reb is not None and reb >= 8:
            return "big"
    except Exception:
        pass
    return "rotation"


def fetch_players_for_team(
    code: str,
    code_to_tid: Dict[str, str],
    injuries_for_date: Dict[str, List[Dict[str, Any]]],
    season_label: str,
) -> List[Dict[str, Any]]:
    tid = code_to_tid.get(code) or TEAM_CODE_TO_ID.get(code)
    if not tid:
        return []
    cache_key = f"{tid}:{season_label}"
    if cache_key in ROSTER_CACHE:
        return ROSTER_CACHE[cache_key]

    df = get_player_stats_df(season_label)
    if df is None:
        print(f"[warn] no league-wide player stats for season {season_label}; players empty for {code}")
        return []

    try:
        team_df = df[df["TEAM_ID"] == int(tid)].copy()
    except Exception:
        print(f"[warn] failed to slice player stats for team {code} (TEAM_ID={tid})")
        return []

    if team_df.empty:
        print(f"[warn] no player stats rows for team {code} (TEAM_ID={tid})")
        return []

    load_player_meta()
    inj_lookup = {
        (entry.get("player") or "").lower(): (entry.get("status") or "active")
        for entry in injuries_for_date.get(code, [])
        if entry.get("player")
    }

    team_df = team_df.sort_values(by="MIN", ascending=False)
    players: List[Dict[str, Any]] = []
    for _, row in team_df.head(10).iterrows():
        pid = int(row.get("PLAYER_ID"))
        mins = row.get("MIN")
        pts = row.get("PTS")
        ast = row.get("AST")
        reb = row.get("REB")
        name = row.get("PLAYER_NAME") or PLAYER_META.get(pid, {}).get("name") or ""
        pos = PLAYER_META.get(pid, {}).get("position") or row.get("POSITION", "") or ""
        status = inj_lookup.get(name.lower(), "active")
        role = _role_from_position(pos, pts, ast, reb)
        players.append(
            {
                "name": name,
                "position": pos,
                "role": role,
                "status": status,
                "minutes": mins,
                "usage": None,
                "pts": pts,
                "ast": ast,
                "reb": reb,
            }
        )
    ROSTER_CACHE[cache_key] = players
    return players


def build_dossier(league: str, date_str: str, matchup: str) -> Dict[str, Any]:
    league = league.lower()
    away, home = matchup.split("@")
    cfg_path = "chimera_v2c/config/defaults.yaml" if league == "nba" else f"chimera_v2c/config/{league}_defaults.yaml"
    cfg = V2CConfig.load(cfg_path)
    ratings = load_ratings_for_league(league, cfg)
    factors = load_json(FACTORS_PATH)
    injuries = load_json(INJURY_PATH)
    raw_inj = load_json(RAW_INJURY_PATH)
    team_stats, code_to_tid = load_team_stats(DB_PATH) if league == "nba" else ({}, {})
    conn = sqlite3.connect(DB_PATH) if league == "nba" else None
    season_start = season_start_for(date_str)
    season_label = season_label_for(date_str)
    injuries_for_date = raw_inj.get(league.upper(), {}).get(date_str, {})
    eff_map = fetch_team_efficiency(season_label) if league == "nba" else {}

    def team_entry(code: str) -> Dict[str, Any]:
        rating_val = ratings.get(code)
        rating_obj = {"elo": rating_val, "source": "ratings_file" if rating_val is not None else "fallback"}
        record: Dict[str, Any] = {}
        efficiency: Dict[str, Any] = {}
        players: List[Dict[str, Any]] = []
        injuries_list = injuries_for_date.get(code, [])

        if league == "nba":
            record = compute_record(conn, code, season_start, date_str)
            efficiency = {
                "off_rating": (eff_map.get(code) or {}).get("off_rating") or (team_stats.get(code) or {}).get("off_rating"),
                "def_rating": (eff_map.get(code) or {}).get("def_rating") or (team_stats.get(code) or {}).get("def_rating"),
                "pace": (eff_map.get(code) or {}).get("pace") or (team_stats.get(code) or {}).get("pace"),
            }
            players = fetch_players_for_team(code, code_to_tid, injuries_for_date, season_label)
        elif league == "nhl":
            efficiency = {
                "goals_for_per_game": None,
                "goals_against_per_game": None,
            }
            record = {}
            if injuries_list:
                players = [
                    {
                        "name": inj.get("player", ""),
                        "status": inj.get("status", "unknown"),
                        "role": "forward",
                        "minutes": None,
                        "usage": None,
                        "pts": None,
                        "ast": None,
                        "reb": None,
                    }
                    for inj in injuries_list
                ]
            if not players:
                players = [{"name": "N/A", "status": "healthy", "role": "forward", "minutes": None, "usage": None, "pts": None, "ast": None, "reb": None}]
        elif league == "nfl":
            efficiency = {
                "points_for_per_game": None,
                "points_against_per_game": None,
            }
            record = {}
            if injuries_list:
                players = [
                    {
                        "name": inj.get("player", ""),
                        "status": inj.get("status", "unknown"),
                        "role": "starter",
                        "minutes": None,
                        "usage": None,
                        "pts": None,
                        "ast": None,
                        "reb": None,
                    }
                    for inj in injuries_list
                ]
            if not players:
                players = [
                    {"name": "QB1", "status": "healthy", "role": "QB", "minutes": None, "usage": None, "pts": None, "ast": None, "reb": None},
                    {"name": "WR1", "status": "healthy", "role": "WR", "minutes": None, "usage": None, "pts": None, "ast": None, "reb": None},
                ]

        return {
            "team": code,
            "ratings": rating_obj,
            "four_factors": factors.get(code) or {} if league == "nba" else {},
            "injury_delta": injuries.get(league.upper(), {}).get(date_str, {}).get(code, 0.0),
            "injuries": injuries_list,
            "efficiency": efficiency,
            "record": record,
            "players": players,
        }

    setattr(cfg, "date_override", date_str)
    p_home_mkt, p_away_mkt, market_source, market_quality = fetch_market_probs(matchup, league, cfg)

    dossier = {
        "version": 1,
        "league": league,
        "date": date_str,
        "game_id": matchup,
        "home_team": home,
        "away_team": away,
        "teams": {
            home: team_entry(home),
            away: team_entry(away),
        },
        "market": {
            "source": market_source or "unknown",
            "quality": market_quality,
            "home_prob": p_home_mkt,
            "away_prob": p_away_mkt,
        },
    }
    if conn:
        conn.close()
    return dossier


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a structured game dossier for LLM input.")
    ap.add_argument("--league", required=True, help="nba or nhl")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--game", required=True, help="Matchup key, e.g., LAL@BOS (away@home)")
    ap.add_argument("--out", help="Optional path to write dossier JSON")
    args = ap.parse_args()

    dossier = build_dossier(args.league, args.date, args.game)
    payload = json.dumps(dossier, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"[info] wrote dossier to {out_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
