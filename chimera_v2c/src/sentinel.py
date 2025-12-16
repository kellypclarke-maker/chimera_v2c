import json
from pathlib import Path

STOP_FLAG = Path("chimera_v2c/data/STOP_TRADING.flag")
HALTED_GAMES_JSON = Path("chimera_v2c/data/halted_games.json")

def is_system_halted() -> bool:
    return STOP_FLAG.exists()

def is_game_halted(game_id: str, ticker: str = None) -> bool:
    if is_system_halted():
        return True
        
    if not HALTED_GAMES_JSON.exists():
        return False
        
    try:
        with HALTED_GAMES_JSON.open('r') as f:
            data = json.load(f)
            
        halted_ids = set(data.get('game_ids', []))
        halted_tickers = set(data.get('tickers', []))
        
        if game_id in halted_ids: return True
        if ticker and ticker in halted_tickers: return True
        
    except Exception as e:
        print(f"Error reading halt file: {e}")
        return False
        
    return False

def halt_game(game_id: str, reason: str):
    data = {'game_ids': [], 'tickers': [], 'reasons': {}}
    if HALTED_GAMES_JSON.exists():
        with HALTED_GAMES_JSON.open('r') as f:
            data = json.load(f)
            
    if game_id not in data['game_ids']:
        data['game_ids'].append(game_id)
        data['reasons'][game_id] = reason
        
    with HALTED_GAMES_JSON.open('w') as f:
        json.dump(data, f, indent=2)
    print(f"Halted Game {game_id}: {reason}")
