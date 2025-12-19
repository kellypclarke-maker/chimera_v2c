from __future__ import annotations

from pathlib import Path

from chimera_v2c.tools.log_rule_a_votes_plan import _plan_fieldnames, _write_csv


def test_log_rule_a_votes_plan_allow_empty_writes_header_only(tmp_path: Path) -> None:
    out = tmp_path / "plan.csv"
    fieldnames = _plan_fieldnames(["v2c", "gpt"])
    _write_csv(out, [], fieldnames=fieldnames, allow_empty=True)

    text = out.read_text(encoding="utf-8").strip()
    assert text
    assert "\n" not in text  # header-only
    assert "event_ticker" in text
    assert "p_home_v2c" in text
    assert "edge_net_gpt" in text
    assert "vote_gpt" in text
