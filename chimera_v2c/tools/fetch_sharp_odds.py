import requests
import sqlite3
import pandas as pd
from datetime import datetime
from scipy.optimize import newton
import sys
import os

# Add path
sys.path.insert(0, os.getcwd())
from chimera_v2c.lib.team_mapper import normalize_team_code

DB_PATH = 'chimera_v2c/data/chimera.db'

def american_to_decimal(us_odds):
    try:
        us_odds = float(us_odds)
    except:
        return 0.0
        
    if us_odds > 0:
        return (us_odds / 100) + 1
    else:
        return (100 / abs(us_odds)) + 1

def power_method_devig(odds_list):
    if any(o <= 1 for o in odds_list): return None
    def target_func(k):
        return sum([(1/d)**k for d in odds_list]) - 1
    try:
        k = newton(target_func, 1.0) 
        probs = [(1/d)**k for d in odds_list]
        return probs
    except Exception:
        implied = [1/d for d in odds_list]
        total = sum(implied)
        return [p/total for p in implied]

def fetch_and_store_odds(date_str=None):
    if not date_str:
        # Default to today + tomorrow? Or just tomorrow?
        # Let's fetch tomorrow for planning
        date_str = (datetime.now() + pd.Timedelta(days=1)).strftime("%Y%m%d")
        
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    print(f"Fetching Odds from ESPN for {date_str}...")
    
    try:
        resp = requests.get(url)
        data = resp.json()
    except Exception as e:
        print(f"Network error: {e}")
        return

    conn = sqlite3.connect(DB_PATH)
    timestamp = datetime.now().isoformat()
    
    count = 0
    
    for evt in data.get('events', []):
        try:
            game_id = evt['id']
            comp = evt['competitions'][0]
            if 'odds' not in comp: continue
            odds_provider = comp['odds'][0]
            provider_name = odds_provider.get('provider', {}).get('name', 'ESPN/DK')
            
            ml = odds_provider.get('moneyline', {})
            if not ml: continue
            
            h_us = ml.get('home', {}).get('close', {}).get('odds')
            a_us = ml.get('away', {}).get('close', {}).get('odds')
            
            if not h_us or not a_us: continue
            
            h_abbr_raw = odds_provider.get('homeTeamOdds', {}).get('team', {}).get('abbreviation')
            a_abbr_raw = odds_provider.get('awayTeamOdds', {}).get('team', {}).get('abbreviation')
            
            h_abbr = normalize_team_code(h_abbr_raw, 'nba')
            a_abbr = normalize_team_code(a_abbr_raw, 'nba')
            
            h_dec = american_to_decimal(h_us)
            a_dec = american_to_decimal(a_us)
            
            true_probs = power_method_devig([h_dec, a_dec]) 
            
            if true_probs:
                h_prob, a_prob = true_probs
                conn.execute('''INSERT OR REPLACE INTO odds_history
                    (game_id, bookmaker, timestamp, home_team, away_team, home_odds, away_odds, home_prob_devig, away_prob_devig)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (game_id, provider_name, timestamp, h_abbr, a_abbr, h_dec, a_dec, h_prob, a_prob))
                count += 1
                
        except Exception:
            continue
            
    conn.commit()
    conn.close()
    print(f"Stored sharp odds for {count} games.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="YYYYMMDD")
    args = parser.parse_args()
    fetch_and_store_odds(args.date)
