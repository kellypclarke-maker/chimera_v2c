from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

from chimera_v2c.lib import team_mapper


def compute_four_factors(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for g_id, group in df.groupby("GAME_ID"):
        if len(group) != 2:
            continue
        # Identify home via matchup string
        row1, row2 = group.iloc[0], group.iloc[1]
        if "vs." in row1["MATCHUP"]:
            home = row1
            away = row2
        else:
            home = row2
            away = row1

        def ff(row, opp):
            fga = row["FGA"] or 1
            tov = row["TOV"] or 0
            fta = row["FTA"] or 0
            oreb = row["OREB"] or 0
            opp_dreb = opp["DREB"] or 0
            efg = (row["FGM"] + 0.5 * row["FG3M"]) / fga
            tov_pct = tov / (fga + 0.44 * fta + tov)
            orb_pct = oreb / (oreb + opp_dreb) if (oreb + opp_dreb) else 0
            ftr = fta / fga
            return efg, tov_pct, orb_pct, ftr

        h_efg, h_tov, h_orb, h_ftr = ff(home, away)
        a_efg, a_tov, a_orb, a_ftr = ff(away, home)
        records.append(
            {
                "home": home["TEAM_ABBREVIATION"],
                "away": away["TEAM_ABBREVIATION"],
                "h_efg": h_efg,
                "h_tov": h_tov,
                "h_orb": h_orb,
                "h_ftr": h_ftr,
                "a_efg": a_efg,
                "a_tov": a_tov,
                "a_orb": a_orb,
                "a_ftr": a_ftr,
            }
        )
    return pd.DataFrame(records)


def aggregate_team_factors(ff_df: pd.DataFrame) -> dict:
    rows = []
    for _, r in ff_df.iterrows():
        rows.append({"team": r["home"], "efg": r["h_efg"], "tov": r["h_tov"], "orb": r["h_orb"], "ftr": r["h_ftr"]})
        rows.append({"team": r["away"], "efg": r["a_efg"], "tov": r["a_tov"], "orb": r["a_orb"], "ftr": r["a_ftr"]})
    df = pd.DataFrame(rows)
    grouped = df.groupby("team").mean()
    out = {}
    for team, row in grouped.iterrows():
        code = team_mapper.normalize_team_code(team, "nba") or team
        out[code] = {
            "efg": float(row["efg"]),
            "tov": float(row["tov"]),
            "orb": float(row["orb"]),
            "ftr": float(row["ftr"]),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare Four Factors and ratings data for v2c.")
    parser.add_argument("--season", default="2024-25", help="NBA season string (e.g., 2024-25)")
    parser.add_argument("--out_factors", default="chimera_v2c/data/team_four_factors.json")
    parser.add_argument("--out_ratings", default="chimera_v2c/data/team_ratings.json")
    args = parser.parse_args()

    print(f"Fetching league game logs for {args.season} ...")
    gl = leaguegamelog.LeagueGameLog(season=args.season, player_or_team_abbreviation="T").get_data_frames()[0]
    ff_df = compute_four_factors(gl)
    factors = aggregate_team_factors(ff_df)

    Path(args.out_factors).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_factors, "w", encoding="utf-8") as f:
        json.dump(factors, f, indent=2)
    print(f"Wrote Four Factors to {args.out_factors}")

    # Simple rating proxy: use net points per game as rating baseline
    gl["DIFF"] = gl["PTS"] - gl["PLUS_MINUS"]
    rating_rows = []
    for team, g in gl.groupby("TEAM_ABBREVIATION"):
        code = team_mapper.normalize_team_code(team, "nba") or team
        rating_rows.append({"team": code, "rating": float(g["PLUS_MINUS"].mean()) + 1500})
    ratings = {r["team"]: r["rating"] for r in rating_rows}
    with open(args.out_ratings, "w", encoding="utf-8") as f:
        json.dump(ratings, f, indent=2)
    print(f"Wrote ratings to {args.out_ratings}")


if __name__ == "__main__":
    main()
