"""
Team Mapping Service for bridging ESPN, Kalshi, and Internal Identifiers.
Handles cross-league code collisions (e.g. PHI in NFL vs PHI in NBA).
"""

# Data split by league to avoid key collisions
NFL_TEAMS = {
    "ARI": ["ARIZONA", "CARDINALS", "ARI"],
    "ATL": ["ATLANTA", "FALCONS", "ATL"],
    "BAL": ["BALTIMORE", "RAVENS", "BAL"],
    "BUF": ["BUFFALO", "BILLS", "BUF"],
    "CAR": ["CAROLINA", "PANTHERS", "CAR"],
    "CHI": ["CHICAGO", "BEARS", "CHI"],
    "CIN": ["CINCINNATI", "BENGALS", "CIN"],
    "CLE": ["CLEVELAND", "BROWNS", "CLE"],
    "DAL": ["DALLAS", "COWBOYS", "DAL"],
    "DEN": ["DENVER", "BRONCOS", "DEN"],
    "DET": ["DETROIT", "LIONS", "DET"],
    "GB":  ["GREEN BAY", "PACKERS", "GB", "G.B."],
    "HOU": ["HOUSTON", "TEXANS", "HOU"],
    "IND": ["INDIANAPOLIS", "COLTS", "IND"],
    "JAX": ["JACKSONVILLE", "JAGUARS", "JAX", "JAC"],
    "KC":  ["KANSAS CITY", "CHIEFS", "KC", "K.C."],
    "LV":  ["LAS VEGAS", "RAIDERS", "LV", "L.V."],
    "LAC": ["LOS ANGELES CHARGERS", "CHARGERS", "LAC", "L.A. CHARGERS"],
    "LAR": ["LOS ANGELES RAMS", "RAMS", "LAR", "LA", "L.A. RAMS"],
    "MIA": ["MIAMI", "DOLPHINS", "MIA"],
    "MIN": ["MINNESOTA", "VIKINGS", "MIN"],
    "NE":  ["NEW ENGLAND", "PATRIOTS", "NE", "N.E."],
    "NO":  ["NEW ORLEANS", "SAINTS", "NO", "N.O.", "NOP"],
    "NYG": ["NEW YORK GIANTS", "GIANTS", "NYG", "N.Y. GIANTS", "NEW YORK G"],
    "NYJ": ["NEW YORK JETS", "JETS", "NYJ", "N.Y. JETS"],
    "PHI": ["PHILADELPHIA", "EAGLES", "PHI"],
    "PIT": ["PITTSBURGH", "STEELERS", "PIT"],
    "SF":  ["SAN FRANCISCO", "49ERS", "SF", "S.F."],
    "SEA": ["SEATTLE", "SEAHAWKS", "SEA"],
    "TB":  ["TAMPA BAY", "BUCCANEERS", "BUCS", "TB", "T.B."],
    "TEN": ["TENNESSEE", "TITANS", "TEN"],
    "WAS": ["WASHINGTON", "COMMANDERS", "WAS", "WSH"],
}

NBA_TEAMS = {
    "ATL": ["ATLANTA", "HAWKS", "ATL"],
    "BKN": ["BROOKLYN", "NETS", "BKN", "BRK"],
    "BOS": ["BOSTON", "CELTICS", "BOS"],
    "CHA": ["CHARLOTTE", "HORNETS", "CHA", "CHO"],
    "CHI": ["CHICAGO", "BULLS", "CHI"],
    "CLE": ["CLEVELAND", "CAVALIERS", "CLE", "CAVS"],
    "DAL": ["DALLAS", "MAVERICKS", "DAL", "MAVS"],
    "DEN": ["DENVER", "NUGGETS", "DEN"],
    "DET": ["DETROIT", "PISTONS", "DET"],
    "GSW": ["GOLDEN STATE", "WARRIORS", "GSW", "GS"],
    "HOU": ["HOUSTON", "ROCKETS", "HOU"],
    "IND": ["INDIANA", "PACERS", "IND"],
    "LAC": ["LOS ANGELES CLIPPERS", "CLIPPERS", "LAC", "L.A. CLIPPERS", "LOS ANGELES C"],
    "LAL": ["LOS ANGELES LAKERS", "LAKERS", "LAL", "L.A. LAKERS", "LOS ANGELES L"],
    "MEM": ["MEMPHIS", "GRIZZLIES", "MEM"],
    "MIA": ["MIAMI", "HEAT", "MIA"],
    "MIL": ["MILWAUKEE", "BUCKS", "MIL"],
    "MIN": ["MINNESOTA", "TIMBERWOLVES", "MIN", "WOLVES"],
    "NOP": ["NEW ORLEANS", "PELICANS", "NOP", "NO"],
    "NYK": ["NEW YORK", "KNICKS", "NYK", "NY", "NEW YORK K"],
    "OKC": ["OKLAHOMA CITY", "THUNDER", "OKC"],
    "ORL": ["ORLANDO", "MAGIC", "ORL"],
    "PHI": ["PHILADELPHIA", "76ERS", "SIXERS", "PHI"],
    "PHX": ["PHOENIX", "SUNS", "PHX", "PHO"],
    "POR": ["PORTLAND", "TRAIL BLAZERS", "BLAZERS", "POR"],
    "SAC": ["SACRAMENTO", "KINGS", "SAC"],
    "SAS": ["SAN ANTONIO", "SPURS", "SAS", "SA"],
    "TOR": ["TORONTO", "RAPTORS", "TOR"],
    "UTA": ["UTAH", "JAZZ", "UTA"],
    "WAS": ["WASHINGTON", "WIZARDS", "WAS", "WSH"],
}

NHL_TEAMS = {
    "ANA": ["ANAHEIM", "DUCKS", "ANA"],
    "BOS": ["BOSTON", "BRUINS", "BOS"],
    "BUF": ["BUFFALO", "SABRES", "BUF"],
    "CGY": ["CALGARY", "FLAMES", "CGY", "CAL"],
    "CAR": ["CAROLINA", "HURRICANES", "CANES", "CAR"],
    "CHI": ["CHICAGO", "BLACKHAWKS", "HAWKS", "CHI"],
    "COL": ["COLORADO", "AVALANCHE", "AVS", "COL"],
    "CBJ": ["COLUMBUS", "BLUE JACKETS", "JACKETS", "CBJ"],
    "DAL": ["DALLAS", "STARS", "DAL"],
    "DET": ["DETROIT", "RED WINGS", "WINGS", "DET"],
    "EDM": ["EDMONTON", "OILERS", "EDM"],
    "FLA": ["FLORIDA", "PANTHERS", "CATS", "FLA"],
    "LAK": ["LOS ANGELES", "KINGS", "LAK", "LA"],
    "MIN": ["MINNESOTA", "WILD", "MIN"],
    "MTL": ["MONTREAL", "CANADIENS", "HABS", "MTL"],
    "NSH": ["NASHVILLE", "PREDATORS", "PREDS", "NSH"],
    "NJD": ["NEW JERSEY", "DEVILS", "NJD", "NJ"],
    "NYI": ["NEW YORK I", "ISLANDERS", "ISLES", "NYI"],
    "NYR": ["NEW YORK R", "RANGERS", "RAGS", "NYR"],
    "OTT": ["OTTAWA", "SENATORS", "SENS", "OTT"],
    "PHI": ["PHILADELPHIA", "FLYERS", "PHI"],
    "PIT": ["PITTSBURGH", "PENGUINS", "PENS", "PIT"],
    "SJS": ["SAN JOSE", "SHARKS", "SJS", "SJ"],
    "SEA": ["SEATTLE", "KRAKEN", "SEA"],
    "STL": ["ST. LOUIS", "ST LOUIS", "BLUES", "STL"],
    "TBL": ["TAMPA BAY", "LIGHTNING", "BOLTS", "TBL", "TB"],
    "TOR": ["TORONTO", "MAPLE LEAFS", "LEAFS", "TOR"],
    "UTA": ["UTAH", "UTAH HOCKEY CLUB", "HOCKEY CLUB", "UTA"],
    "VAN": ["VANCOUVER", "CANUCKS", "VAN"],
    "VGK": ["VEGAS", "GOLDEN KNIGHTS", "KNIGHTS", "VGK", "VEG"],
    "WSH": ["WASHINGTON", "CAPITALS", "CAPS", "WSH", "WAS"],
    "WPG": ["WINNIPEG", "JETS", "WPG"],
}

LEAGUE_MAP = {
    'nfl': NFL_TEAMS,
    'nba': NBA_TEAMS,
    'nhl': NHL_TEAMS
}

def get_alias_candidates(code, league):
    """
    Return a set of aliases (including the canonical code) for a given team code and league.
    Useful for matching ESPN vs Kalshi naming differences.
    """
    if not code or not league:
        return []
    league_data = LEAGUE_MAP.get(league.lower(), {})
    norm_code = normalize_team_code(code, league)
    aliases = set()
    if norm_code and norm_code in league_data:
        for alias in league_data[norm_code]:
            aliases.add(alias.upper())
        aliases.add(norm_code.upper())
    else:
        # Fallback to whatever was provided
        aliases.add(code.upper())
        aliases.update(a.upper() for a in league_data.get(code.upper(), []))
    return list(aliases)

def normalize_team_code(input_str, league):
    """
    Given an input string (e.g., 'JAC', 'Jaguars', 'Jacksonville'), return the standard 3-letter code (JAX).
    """
    if not input_str or not league:
        return None
        
    data = LEAGUE_MAP.get(league.lower())
    if not data:
        return None
        
    upper_input = input_str.upper().strip()
    
    # Direct lookup
    if upper_input in data:
        return upper_input
        
    # Alias lookup
    for code, aliases in data.items():
        if upper_input in aliases:
            return code
            
    return None

def get_aliases(code, league):
    data = LEAGUE_MAP.get(league.lower(), {})
    return data.get(code, [])

def match_teams_in_string(text, league):
    """
    Find two distinct teams in a string (e.g., Kalshi Ticker '25NOV23JACARI').
    Returns tuple (TeamCode1, TeamCode2) or None.
    """
    if not text or not league:
        return None
        
    data = LEAGUE_MAP.get(league.lower())
    if not data:
        return None
    
    text_upper = text.upper()
    
    # Strategy 0: Strip standard Kalshi date prefix (e.g. 25DEC08) if present
    # Pattern: 2 digits, 3 letters, 2 digits. Length 7.
    # But just stripping the first 7 chars if they look like a date is safer.
    clean_text = text_upper
    if len(text_upper) > 6 and text_upper[0:2].isdigit():
        # Heuristic: If it starts with digits, it's likely a date prefix.
        # Kalshi format is usually YYMMMdd (e.g., 23NOV25) -> 7 chars.
        # We can try suffix matching.
        pass

    valid_aliases_map = {} # alias -> code
    for code, aliases in data.items():
        for a in aliases:
            valid_aliases_map[a] = code
            
    # Strategy 1: Suffix Split (Exact Match)
    # Most reliable. Try to split the LAST 6-8 chars.
    # e.g. "25DEC08SACIND" -> try splitting "SACIND"
    
    # Try all possible split points for the *whole string* first (legacy support)
    for i in range(1, len(text_upper)):
        left = text_upper[:i]
        right = text_upper[i:]
        if valid_aliases_map.get(left) and valid_aliases_map.get(right):
            return [valid_aliases_map[left], valid_aliases_map[right]]

    # Strategy 2: Suffix Split (Smart)
    # If the string ends with TEAM1TEAM2, we can find it.
    # We iterate i from len-8 to len-1
    start_search = max(0, len(text_upper) - 10)
    for i in range(start_search, len(text_upper)):
        # We need to find a split where:
        # 1. right part is a valid team
        # 2. the part immediately preceding it is a valid team
        
        right_part = text_upper[i:]
        code_right = valid_aliases_map.get(right_part)
        
        if code_right:
            # Now check the left neighbor
            # We try different lengths for the left team (2, 3, 4 chars)
            for length in [2, 3, 4]:
                if i - length < 0: continue
                left_part = text_upper[i-length:i]
                code_left = valid_aliases_map.get(left_part)
                if code_left:
                     # Found it!
                     return [code_left, code_right]

    # Strategy 3: Substring Search (Fallback - strict)
    # Only match if the alias is NOT a substring of another found alias
    matches = set()
    found_aliases = []
    
    for code, aliases in data.items():
        # Filter for short aliases (<= 4 chars) to match ticker codes
        ticker_aliases = [a for a in aliases if len(a) <= 4]
        for alias in ticker_aliases:
            if alias in text_upper:
                found_aliases.append((alias, code))

    # Sort by length (longest first) to prevent 'SA' matching 'SAC'
    found_aliases.sort(key=lambda x: len(x[0]), reverse=True)
    
    final_teams = []
    used_indices = set()
    
    for alias, code in found_aliases:
        # Check where this alias occurs
        start_idx = text_upper.find(alias)
        end_idx = start_idx + len(alias)
        
        # If this region overlaps with an already found longer alias, skip it
        # (e.g. if we found SAC at 0-3, skip SA at 0-2)
        is_overlapping = any(idx in used_indices for idx in range(start_idx, end_idx))
        
        if not is_overlapping:
            final_teams.append(code)
            for idx in range(start_idx, end_idx):
                used_indices.add(idx)
                
    if len(final_teams) >= 2:
        return final_teams[:2]
        
    return None
