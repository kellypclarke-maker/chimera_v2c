from __future__ import annotations

import json
from pathlib import Path


def test_rule_a_vote_calibration_nhl_has_base_units_zero() -> None:
    path = Path("chimera_v2c/data/rule_a_vote_calibration_nhl.json")
    d = json.loads(path.read_text(encoding="utf-8"))
    assert int(d.get("base_units")) == 0
    models = [str(x) for x in (d.get("models") or [])]
    # NHL calibration should include at least the core sources; additional candidates may be present
    # and are controlled via primary/secondary eligibility.
    for required in ["gemini", "grok", "market_proxy"]:
        assert required in set(models)


def test_vote_delta_calibration_load_json_preserves_base_units_zero() -> None:
    from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration

    cfg = VoteDeltaCalibration.load_json(Path("chimera_v2c/data/rule_a_vote_calibration_nhl.json"))
    assert cfg.base_units == 0
