import datetime

from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.pipeline import build_daily_plan


def test_build_daily_plan_with_stubbed_markets(monkeypatch, tmp_path):
    # Stub ESPN + Kalshi to avoid network
    import chimera_v2c.src.pipeline as pipeline

    def fake_fetch_matchups(league, date):
        return [{"home": "BOS", "away": "NYK", "event_id": "EVT1"}]

    class MQ:
        def __init__(self):
            self.mid = 0.5
            self.ticker = "KXNBAGAME-TEST"
            self.yes_bid = 49
            self.yes_ask = 51

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
    cfg.use_sharp_prior = False
    plans = build_daily_plan(cfg, datetime.date.today())
    assert len(plans) == 1
    p = plans[0]
    assert p.market is not None
    assert p.yes_team == "BOS"
    assert any(s.yes_team == "BOS" for s in p.sides)
