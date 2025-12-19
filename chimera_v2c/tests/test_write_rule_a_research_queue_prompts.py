from __future__ import annotations

from chimera_v2c.tools.write_rule_a_research_queue import QueueRow, build_prompt_pack_markdown


def test_prompt_pack_contains_json_template() -> None:
    md = build_prompt_pack_markdown(
        rows=[
            QueueRow(
                date="2025-12-17",
                league="nhl",
                matchup="LAK@FLA",
                away="LAK",
                home="FLA",
                event_ticker="KXNHLGAME-25DEC17LAFLA",
                market_ticker_home="KXNHLGAME-25DEC17LAFLA-FLA",
                market_ticker_away="KXNHLGAME-25DEC17LAFLA-LA",
                start_time_utc="2025-12-18T00:00:00Z",
                minutes_to_start="30.00",
                home_yes_bid_cents="58",
                home_yes_ask_cents="59",
                mid_home="0.585000",
                away_yes_bid_cents="41",
                away_yes_ask_cents="42",
                away_yes_ask_dollars="0.420000",
                slippage_cents="1",
                price_away_taker_assumed="0.430000",
            )
        ]
    )
    assert "Return ONLY this JSON" in md
    assert '{"matchup":"LAK@FLA"' in md

