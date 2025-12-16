import datetime
from pathlib import Path

from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src import pipeline


def test_ws_cache_overrides_mid(monkeypatch, tmp_path):
    cache_path = tmp_path / "ws.json"
    cache_path.write_text(
        '{"KXNBAGAME-TEST": {"yes_bid": 60, "yes_ask": 62, "ts": "t"}}',
        encoding="utf-8",
    )

    def fake_fetch_matchups(league, date):
        return [{"home": "BOS", "away": "NYK", "event_id": "EVT1"}]

    class MQ:
        def __init__(self):
            self.ticker = "KXNBAGAME-TEST"
            self.yes_bid = 10
            self.yes_ask = 90

        @property
        def mid(self):
            return ((self.yes_bid + self.yes_ask) / 2.0) / 100.0

    def fake_fetch_markets(league, series_ticker, use_private=False, **kwargs):
        return []

    def fake_match_markets_to_games(matchups, markets):
        mq = MQ()
        mq.yes_team = "BOS"
        return {f"{matchups[0]['away']}@{matchups[0]['home']}": {"BOS": mq}}

    monkeypatch.setattr(pipeline, "fetch_matchups", fake_fetch_matchups)
    monkeypatch.setattr(pipeline, "fetch_markets", fake_fetch_markets)
    monkeypatch.setattr(pipeline, "match_markets_to_games", fake_match_markets_to_games)

    cfg = V2CConfig.load("chimera_v2c/config/defaults.yaml")
    cfg.paths["ws_cache"] = str(cache_path)
    cfg.use_sharp_prior = False

    plans = pipeline.build_daily_plan(cfg, datetime.date.today())
    side_market = plans[0].sides[0].market
    assert side_market is not None
    assert side_market.yes_bid == 60
    assert side_market.yes_ask == 62
