import datetime


def test_market_linker_parses_event_core_with_two_letter_codes():
    from chimera_v2c.src.market_linker import _market_matchup_from_event_core

    away, home, date_iso = _market_matchup_from_event_core("25DEC14VANNJ", "nhl")  # NJ is 2-letter in Kalshi tickers
    assert date_iso == "2025-12-14"
    assert away == "VAN"
    assert home == "NJD"


def test_market_linker_parses_event_core_with_two_letter_nfl_codes():
    from chimera_v2c.src.market_linker import _market_matchup_from_event_core

    away, home, date_iso = _market_matchup_from_event_core("25DEC21NEBAL", "nfl")  # NE is 2-letter in Kalshi tickers
    assert date_iso == "2025-12-21"
    assert away == "NE"
    assert home == "BAL"


def test_match_markets_to_games_matches_by_venue_not_pair():
    from chimera_v2c.src.market_linker import MarketQuote, match_markets_to_games

    matchups = [{"away": "BOS", "home": "NYK"}]
    markets = [
        # BOS@NYK (target)
        MarketQuote(ticker="T1", yes_bid=49, yes_ask=51, home="NYK", away="BOS", yes_team="NYK"),
        MarketQuote(ticker="T2", yes_bid=49, yes_ask=51, home="NYK", away="BOS", yes_team="BOS"),
        # NYK@BOS (different venue; should not be matched to BOS@NYK)
        MarketQuote(ticker="T3", yes_bid=49, yes_ask=51, home="BOS", away="NYK", yes_team="BOS"),
        MarketQuote(ticker="T4", yes_bid=49, yes_ask=51, home="BOS", away="NYK", yes_team="NYK"),
    ]
    market_map = match_markets_to_games(matchups, markets)
    assert set(market_map.keys()) == {"BOS@NYK"}
    assert set(market_map["BOS@NYK"].keys()) == {"BOS", "NYK"}


def test_fetch_markets_filters_to_target_date(monkeypatch):
    from chimera_v2c.src.market_linker import fetch_markets

    def fake_list_public_markets(*_args, **_kwargs):
        return {
            "markets": [
                {
                    "ticker": "KXNHLGAME-25DEC14VANNJ-NJ",
                    "event_ticker": "KXNHLGAME-25DEC14VANNJ",
                    "yes_bid": 50,
                    "yes_ask": 52,
                    "no_bid": 48,
                    "no_ask": 50,
                },
                {
                    "ticker": "KXNHLGAME-25DEC15VANBOS-VAN",
                    "event_ticker": "KXNHLGAME-25DEC15VANBOS",
                    "yes_bid": 50,
                    "yes_ask": 52,
                    "no_bid": 48,
                    "no_ask": 50,
                },
            ]
        }

    import chimera_v2c.lib.kalshi_utils as kalshi_utils

    monkeypatch.setattr(kalshi_utils, "list_public_markets", fake_list_public_markets)

    mkts = fetch_markets(
        "nhl",
        "KXNHLGAME",
        use_private=False,
        status="open",
        target_date=datetime.date(2025, 12, 14),
    )
    assert len(mkts) == 1
    assert mkts[0].away == "VAN"
    assert mkts[0].home == "NJD"
    assert mkts[0].event_date == "2025-12-14"
