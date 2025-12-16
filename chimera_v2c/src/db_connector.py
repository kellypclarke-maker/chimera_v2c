import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("chimera_v2c/data/chimera.db")

def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found at {DB_PATH}")
    return sqlite3.connect(DB_PATH)

def get_team_averages(conn, season_start='2024-10-01'):
    query = f"""
    SELECT 
        CASE WHEN g.home_team_id = ts.team_id THEN g.home_team ELSE g.away_team END as team_abbr,
        AVG(efg_pct) as avg_efg,
        AVG(tov_pct) as avg_tov,
        AVG(orb_pct) as avg_orb,
        AVG(ft_rate) as avg_ft
    FROM team_stats ts
    JOIN games g ON ts.game_id = g.game_id
    WHERE g.game_date > '{season_start}'
    GROUP BY team_abbr
    """
    df = pd.read_sql(query, conn)
    return df.set_index('team_abbr')

def get_latest_sharp_odds(conn, date_str):
    """
    Fetch the latest sharp probability for games on a given date.
    """
    # Join games with odds_history
    # We need to match game_id or team abbreviations.
    # Our odds_collector in v2b stored home_team/away_team abbreviations in odds_history.
    
    # We want rows for the specific game date.
    # Note: odds_history doesn't have game_date, but 'timestamp'.
    # We should link via 'games' table if possible, or just use the teams.
    
    # Simple approach: Get all odds from odds_history where timestamp is recent?
    # Better: The caller usually knows the matchup (Home vs Away).
    # Let's just return a dataframe of all recent odds and let the caller filter.
    
    query = """
    SELECT home_team, away_team, home_prob_devig, away_prob_devig, timestamp
    FROM odds_history
    ORDER BY timestamp DESC
    """
    df = pd.read_sql(query, conn)
    
    # Filter for duplicates (take latest per matchup)
    # We can do this in pandas
    df['matchup_id'] = df.apply(lambda x: tuple(sorted([x['home_team'], x['away_team']])), axis=1)
    df = df.drop_duplicates(subset='matchup_id', keep='first')
    return df.set_index(['home_team', 'away_team'])
