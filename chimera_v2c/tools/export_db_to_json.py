import sqlite3
import pandas as pd
import json
import sys
import os
from pathlib import Path

# DB Source
DB_PATH = Path('chimera_v2c/data/chimera.db')

# JSON Targets
RATINGS_PATH = Path('chimera_v2c/data/team_ratings.json')
FACTORS_PATH = Path('chimera_v2c/data/team_four_factors.json')

def get_db_conn():
    if not DB_PATH.exists():
        print(f"Error: DB not found at {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)

def export_four_factors(conn):
    print("Exporting Four Factors from DB to JSON...")
    
    # Calculate Rolling Averages from DB
    query = """
    SELECT 
        CASE WHEN g.home_team_id = ts.team_id THEN g.home_team ELSE g.away_team END as team_abbr,
        AVG(efg_pct) as avg_efg,
        AVG(tov_pct) as avg_tov,
        AVG(orb_pct) as avg_orb,
        AVG(ft_rate) as avg_ft
    FROM team_stats ts
    JOIN games g ON ts.game_id = g.game_id
    WHERE g.game_date > '2024-10-01' -- Current Season
    GROUP BY team_abbr
    """
    
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        print(f"Error reading stats: {e}")
        return

    # Transform to JSON format expected by ProbabilityEngine
    # Format: {"LAL": {"efg": 0.55, "tov": 0.14, "orb": 0.25, "ftr": 0.20}, ...}
    
    factors_dict = {}
    for _, row in df.iterrows():
        abbr = row['team_abbr']
        if not abbr: continue
        factors_dict[abbr.upper()] = {
            "efg": round(row['avg_efg'], 4),
            "tov": round(row['avg_tov'], 4),
            "orb": round(row['avg_orb'], 4),
            "ftr": round(row['avg_ft'], 4)
        }
        
    # Write to JSON
    with open(FACTORS_PATH, 'w', encoding='utf-8') as f:
        json.dump(factors_dict, f, indent=2)
    
    print(f"Exported factors for {len(factors_dict)} teams to {FACTORS_PATH}")

def export_ratings(conn):
    # If we had Elo ratings in DB, we would export them here.
    # Currently, v2b EloEngine calculates them in-memory or we need to persist them.
    # For now, we will create a placeholder or reuse existing if DB has no ratings table.
    
    # We didn't create a 'ratings' table in DB yet, EloEngine uses in-memory dict.
    # To bridge this, we should have EloEngine save to DB, then export here.
    
    # Placeholder: Just ensure file exists if missing
    if not RATINGS_PATH.exists():
        with open(RATINGS_PATH, 'w') as f:
            json.dump({}, f)
        print("Created empty ratings file.")
    else:
        print("Skipping ratings export (DB table not yet implemented). Keeping existing JSON.")

def main():
    conn = get_db_conn()
    export_four_factors(conn)
    export_ratings(conn)
    conn.close()

if __name__ == "__main__":
    main()
