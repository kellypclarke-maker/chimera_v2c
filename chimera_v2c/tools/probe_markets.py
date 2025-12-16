import sys
import os
sys.path.insert(0, os.getcwd())

from chimera_v2c.lib import kalshi_utils

print("--- SEARCHING FOR NBA ---")
try:
    resp = kalshi_utils.list_public_markets(limit=1000, status="open")
    mkts = resp.get("markets", [])
    nba = [m for m in mkts if "NBA" in m.get("ticker", "")]
    print(f"Found {len(nba)} NBA markets.")
    for m in nba[:5]:
        print(f"Ticker: {m.get('ticker')}")
except Exception as e:
    print(f"Error: {e}")
