from __future__ import annotations

from pathlib import Path


def test_latest_plan_csv_picks_lexicographic_last(tmp_path: Path, monkeypatch) -> None:
    # Arrange a fake repo root with the expected folder structure.
    root = tmp_path
    ymd = "20251217"
    d = root / "reports" / "execution_logs" / "rule_a_votes" / ymd
    d.mkdir(parents=True, exist_ok=True)
    (d / "rule_a_votes_plan_nhl_20251217_010000Z.csv").write_text("x", encoding="utf-8")
    (d / "rule_a_votes_plan_nhl_20251217_020000Z.csv").write_text("x", encoding="utf-8")
    (d / "rule_a_votes_plan_nhl_20251217_015959Z.csv").write_text("x", encoding="utf-8")
    (d / "rule_a_votes_plan_nhl_20251217_020000Z_results.csv").write_text("x", encoding="utf-8")

    monkeypatch.chdir(root)

    from chimera_v2c.tools.run_rule_a_daily import _latest_plan_csv

    latest = _latest_plan_csv(date_iso="2025-12-17", league="nhl")
    assert latest is not None
    assert latest.name == "rule_a_votes_plan_nhl_20251217_020000Z.csv"
