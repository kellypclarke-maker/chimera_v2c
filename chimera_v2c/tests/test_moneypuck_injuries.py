from chimera_v2c.lib.moneypuck_injuries import (
    canonical_rows,
    diff_by_player_id,
    parse_current_injuries_csv,
    render_team_digest,
    sha256_of_rows,
)


def test_parse_and_hash_is_stable():
    csv_text = "\n".join(
        [
            "playerId,playerName,teamCode,position,dateOfReturn,daysUntilReturn,gamesStillToMiss,gamesMissedSoFar,lastGameDate,yahooInjuryDescription,playerInjuryStatus",
            "1,Test Goalie,ANA,G,2025-12-20,1,2,3,2025-12-10,Lower Body,IR",
            "2,Test Skater,BOS,C,2099-12-31,-999,-999,0,2025-12-01,Upper Body,O",
            "",
        ]
    )
    rows = parse_current_injuries_csv(csv_text)
    canon = canonical_rows(rows)
    h1 = sha256_of_rows(canon)
    h2 = sha256_of_rows(canonical_rows(parse_current_injuries_csv(csv_text)))
    assert h1 == h2
    assert len(canon) == 2
    assert canon[0]["team"] in {"ANA", "BOS"}


def test_diff_by_player_id_detects_changes():
    old = [
        {"player_id": "1", "player_name": "A", "team": "ANA", "position": "G", "injury_status": "IR", "injury_description": "X", "date_of_return": "2025-12-20", "last_game_date": "2025-12-10", "games_still_to_miss": "2", "games_missed_so_far": "3"},
    ]
    new = [
        {"player_id": "1", "player_name": "A", "team": "ANA", "position": "G", "injury_status": "O", "injury_description": "X", "date_of_return": "2025-12-20", "last_game_date": "2025-12-10", "games_still_to_miss": "2", "games_missed_so_far": "3"},
        {"player_id": "2", "player_name": "B", "team": "BOS", "position": "C", "injury_status": "IR", "injury_description": "Y", "date_of_return": "2025-12-21", "last_game_date": "2025-12-11", "games_still_to_miss": "1", "games_missed_so_far": "1"},
    ]
    diff = diff_by_player_id(old_rows=old, new_rows=new)
    assert len(diff["changed"]) == 1
    assert len(diff["added"]) == 1
    assert len(diff["removed"]) == 0


def test_render_team_digest_filters_teams():
    rows = [
        {"player_id": "1", "player_name": "A", "team": "ANA", "position": "G", "injury_status": "IR", "injury_description": "Lower Body", "date_of_return": "2025-12-20", "last_game_date": "2025-12-10", "games_still_to_miss": "2", "games_missed_so_far": "3"},
        {"player_id": "2", "player_name": "B", "team": "BOS", "position": "C", "injury_status": "O", "injury_description": "Upper Body", "date_of_return": "2099-12-31", "last_game_date": "2025-12-11", "games_still_to_miss": "-999", "games_missed_so_far": "1"},
    ]
    txt = render_team_digest(date_iso="2025-12-13", rows=rows, teams=["ANA"])
    assert "ANA:" in txt
    assert "BOS:" not in txt

