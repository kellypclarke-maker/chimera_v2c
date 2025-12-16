import json
from pathlib import Path

from chimera_v2c.src.doctrine import bucket_for_p
from chimera_v2c.tools.build_roi_by_bucket_guardrails import build_roi_by_bucket


def test_bucket_for_p_is_005_buckets():
    assert bucket_for_p(0.52) == "[0.50,0.55)"
    assert bucket_for_p(0.49) == "[0.45,0.50)"
    assert bucket_for_p(0.0) == "[0.00,0.05)"
    assert bucket_for_p(1.0).startswith("[0.95,1.00")


def test_build_roi_by_bucket_guardrails_selected_rows_only(tmp_path: Path):
    daily_dir = tmp_path / "daily_ledgers"
    daily_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = daily_dir / "20251204_daily_game_ledger.csv"
    ledger_path.write_text(
        "\n".join(
            [
                "date,league,matchup,kalshi_mid,actual_outcome",
                "2025-12-04,nhl,STL@BOS,0.55,STL 2-5 BOS",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plan_log_path = tmp_path / "v2c_plan_log.json"
    plan_log_path.write_text(
        json.dumps(
            [
                {
                    "date": "2025-12-04",
                    "league": "nhl",
                    "matchup": "STL@BOS",
                    "home": "BOS",
                    "away": "STL",
                    "yes_team": "BOS",
                    "mid": 0.60,
                    "p_yes": 0.65,
                    "selected": True,
                    "stake_fraction": 0.01,
                    "ticker": "KXNHLGAME-TEST",
                },
                {
                    "date": "2025-12-04",
                    "league": "nhl",
                    "matchup": "STL@BOS",
                    "home": "BOS",
                    "away": "STL",
                    "yes_team": "STL",
                    "mid": 0.40,
                    "p_yes": 0.35,
                    "selected": False,
                    "stake_fraction": None,
                    "ticker": "KXNHLGAME-TEST",
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = build_roi_by_bucket(
        league="nhl",
        start_date="2025-12-04",
        end_date="2025-12-04",
        plan_log_path=plan_log_path,
        daily_dir=daily_dir,
        bucket_width=0.05,
        min_bets=1,
        require_selected=True,
    )
    # Only the selected BOS side should be included; BOS wins at price 0.60 => ROI=(1-0.60)/0.60=0.666...
    row = next(r for r in rows if r["bucket"] == "[0.65,0.70)")
    assert row["n_bets"] == 1
    assert row["eligible"] is True
    assert abs(float(row["roi_estimate"]) - (1.0 - 0.60) / 0.60) < 1e-9
