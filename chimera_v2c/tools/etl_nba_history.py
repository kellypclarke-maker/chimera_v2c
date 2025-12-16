import time
import pandas as pd
import sqlite3
import sys
import os
from datetime import datetime

# Add root path
sys.path.insert(0, os.getcwd())
from chimera_v2c.lib.team_mapper import normalize_team_code

# Database Path (v2c-local)
DB_PATH = 'chimera_v2c/data/chimera.db'

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def fetch_game_logs(season):
    print(f"Fetching Game Logs for {season}...")
    from nba_api.stats.endpoints import leaguegamelog
    
    try:
        gl = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='T')
        df = gl.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error fetching logs for {season}: {e}")
        return None

def process_and_save_games(df, season, rolling_window: int = 5):
    if df is None:
        return
    conn = get_db_connection()

    df["GAME_DATE_DT"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values("GAME_DATE_DT")

    features = []
    by_team = df.groupby("TEAM_ID")
    for team_id, tdf in by_team:
        tdf = tdf.sort_values("GAME_DATE_DT")
        tdf["efg"] = (tdf["FGM"] + 0.5 * tdf["FG3M"]) / tdf["FGA"]
        tdf["tov_pct"] = tdf["TOV"] / (tdf["FGA"] + 0.44 * tdf["FTA"] + tdf["TOV"])
        tdf["orb_pct"] = tdf["OREB"] / (tdf["OREB"] + tdf["DREB"].shift(-1))  # approx vs opp DREB
        tdf["ft_rate"] = tdf["FTA"] / tdf["FGA"]
        tdf["poss"] = tdf["FGA"] + 0.44 * tdf["FTA"] + tdf["TOV"] - tdf["OREB"]
        tdf["pace"] = tdf["poss"] / (tdf["MIN"] / 5.0) if "MIN" in tdf else 0
        tdf["days_rest"] = tdf["GAME_DATE_DT"].diff().dt.days.fillna(3).clip(lower=0)
        roll = tdf[["efg", "tov_pct", "orb_pct", "ft_rate", "pace"]].rolling(rolling_window, min_periods=1).mean().shift(1)
        tdf["efg_roll"] = roll["efg"]
        tdf["tov_roll"] = roll["tov_pct"]
        tdf["orb_roll"] = roll["orb_pct"]
        tdf["ft_roll"] = roll["ft_rate"]
        tdf["pace_roll"] = roll["pace"]
        features.append(tdf)
    df = pd.concat(features)

    games = df.groupby("GAME_ID")
    count = 0
    for g_id, group in games:
        if len(group) != 2:
            continue

        row1 = group.iloc[0]
        row2 = group.iloc[1]
        if "vs." in row1["MATCHUP"]:
            home = row1
            away = row2
        else:
            home = row2
            away = row1

        h_abbr = normalize_team_code(home["TEAM_ABBREVIATION"], "nba")
        a_abbr = normalize_team_code(away["TEAM_ABBREVIATION"], "nba")
        winner = h_abbr if home["PTS"] > away["PTS"] else a_abbr

        conn.execute(
            """INSERT OR REPLACE INTO games 
            (game_id, game_date, matchup, home_team, away_team, home_team_id, away_team_id, home_score, away_score, winner) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                g_id,
                home["GAME_DATE"],
                home["MATCHUP"],
                h_abbr,
                a_abbr,
                home["TEAM_ID"],
                away["TEAM_ID"],
                int(home["PTS"]),
                int(away["PTS"]),
                winner,
            ),
        )

        def insert_team(row):
            conn.execute(
                """INSERT OR REPLACE INTO team_stats 
                (game_id, team_id, off_rating, def_rating, efg_pct, tov_pct, orb_pct, ft_rate, pace, days_rest, efg_roll, tov_roll, orb_roll, ft_roll, pace_roll) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    g_id,
                    row["TEAM_ID"],
                    0,
                    0,
                    row.get("efg", 0),
                    row.get("tov_pct", 0),
                    row.get("orb_pct", 0),
                    row.get("ft_rate", 0),
                    row.get("pace", 0),
                    row.get("days_rest", 0),
                    row.get("efg_roll", 0),
                    row.get("tov_roll", 0),
                    row.get("orb_roll", 0),
                    row.get("ft_roll", 0),
                    row.get("pace_roll", 0),
                ),
            )

        insert_team(home)
        insert_team(away)
        count += 1

    conn.commit()
    print(f"Processed {count} games for season {season}")

def run_etl():
    seasons = ['2023-24', '2024-25']
    for s in seasons:
        df = fetch_game_logs(s)
        process_and_save_games(df, s)
        time.sleep(1)

if __name__ == "__main__":
    run_etl()
