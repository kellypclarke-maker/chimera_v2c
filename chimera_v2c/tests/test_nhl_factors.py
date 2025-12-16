import datetime
from pathlib import Path

from chimera_v2c.src.probability import ProbabilityEngine


def test_nhl_factor_component(tmp_path):
    factors_path = tmp_path / "factors.json"
    factors_path.write_text(
        """
        {
          "NHL": {
            "TOR": {"xgf_pct": 0.55, "hdcf_pct": 0.56, "pp_index": 1.0, "pk_index": 0.8, "goalie_rating": 0.6, "oi_sh_pct": 0.65},
            "MTL": {"xgf_pct": 0.45, "hdcf_pct": 0.44, "pp_index": -0.5, "pk_index": -0.4, "goalie_rating": 0.4, "oi_sh_pct": 0.45}
          }
        }
        """,
        encoding="utf-8",
    )
    ratings_path = tmp_path / "ratings.json"
    ratings_path.write_text('{"TOR": 1500, "MTL": 1500}', encoding="utf-8")
    injury_path = tmp_path / "injury.json"
    injury_path.write_text("{}", encoding="utf-8")

    eng = ProbabilityEngine(
        league="nhl",
        weights={"elo": 0.0, "four_factors": 1.0, "market": 0.0, "goalie": 0.0},
        elo_cfg={"base_rating": 1500.0, "home_bonus": 0.0, "k_scale": 0.18},
        ff_cfg={
            "xgf_weight": 0.35,
            "hdcf_weight": 0.20,
            "pp_weight": 0.20,
            "pk_weight": 0.15,
            "goalie_weight": 0.10,
            "intercept": 0.0,
            "steepness": 1.2,
        },
        ratings_path=ratings_path,
        factors_path=factors_path,
        injury_path=injury_path,
        goalie_cfg={},
        goalie_path=None,
    )

    p_final, comps = eng.ensemble_prob("TOR", "MTL", datetime.date.today().isoformat(), market_mid=0.5, sharp_prior=None)
    assert comps["four_factors"] > 0.5
    assert 0.5 < p_final < 1.0
