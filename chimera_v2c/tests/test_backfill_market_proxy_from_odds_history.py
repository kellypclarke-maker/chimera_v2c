from __future__ import annotations

from pathlib import Path

import pandas as pd

import chimera_v2c.tools.backfill_market_proxy_from_odds_history as tool


def _write_ledger(path: Path, market_proxy: str) -> None:
    pd.DataFrame(
        [
            {
                "date": "2099-01-01",
                "league": "nba",
                "matchup": "DET@BOS",
                "v2c": "NR",
                "grok": "NR",
                "gemini": "NR",
                "gpt": "NR",
                "kalshi_mid": "NR",
                "market_proxy": market_proxy,
                "moneypuck": "NR",
                "actual_outcome": "",
            }
        ]
    ).to_csv(path, index=False)


def test_apply_market_proxy_fills_nr_without_overwrite(tmp_path: Path, monkeypatch) -> None:
    ledger_path = tmp_path / "20990101_daily_game_ledger.csv"
    _write_ledger(ledger_path, market_proxy="NR")

    monkeypatch.setattr(tool, "LEDGER_SNAPSHOT_DIR", tmp_path / "snapshots")

    updated, _ = tool.apply_market_proxy(
        ledger_path,
        league="nba",
        books_map={"DET@BOS": {"market_proxy": 0.5}},
        allow_overwrite=False,
        apply=True,
    )
    assert updated is True
    df = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    assert df.loc[0, "market_proxy"] == ".50"


def test_apply_market_proxy_respects_no_overwrite(tmp_path: Path, monkeypatch) -> None:
    ledger_path = tmp_path / "20990101_daily_game_ledger.csv"
    _write_ledger(ledger_path, market_proxy=".40")

    monkeypatch.setattr(tool, "LEDGER_SNAPSHOT_DIR", tmp_path / "snapshots")

    updated, _ = tool.apply_market_proxy(
        ledger_path,
        league="nba",
        books_map={"DET@BOS": {"market_proxy": 0.5}},
        allow_overwrite=False,
        apply=True,
    )
    assert updated is False
    df = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    assert df.loc[0, "market_proxy"] == ".40"


def test_apply_market_proxy_allows_overwrite(tmp_path: Path, monkeypatch) -> None:
    ledger_path = tmp_path / "20990101_daily_game_ledger.csv"
    _write_ledger(ledger_path, market_proxy=".40")

    monkeypatch.setattr(tool, "LEDGER_SNAPSHOT_DIR", tmp_path / "snapshots")

    updated, _ = tool.apply_market_proxy(
        ledger_path,
        league="nba",
        books_map={"DET@BOS": {"market_proxy": 0.5}},
        allow_overwrite=True,
        apply=True,
    )
    assert updated is True
    df = pd.read_csv(ledger_path, dtype=str, keep_default_na=False)
    assert df.loc[0, "market_proxy"] == ".50"

