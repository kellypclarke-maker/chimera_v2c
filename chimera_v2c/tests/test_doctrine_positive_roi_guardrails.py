from pathlib import Path

from chimera_v2c.src.doctrine import DoctrineConfig, doctrine_decide_trade


def test_doctrine_require_positive_roi_buckets_blocks_unknown_and_nonpositive(tmp_path: Path) -> None:
    csv_path = tmp_path / "roi_by_bucket.csv"
    csv_path.write_text(
        "\n".join(
            [
                "bucket,roi_estimate",
                "\"[0.55,0.60)\",-0.10",
                "\"[0.60,0.65)\",0.10",
                "\"[0.65,0.70)\",0.00",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = DoctrineConfig(
        max_fraction=0.01,
        enable_bucket_guardrails=True,
        require_positive_roi_buckets=True,
        bucket_guardrails_path=str(csv_path),
        league="nhl",
    )

    # Unknown bucket -> blocked.
    stake, _, reason = doctrine_decide_trade(
        p_model=0.77,
        p_market=0.60,
        cfg=cfg,
        used_fraction=0.0,
        daily_cap=1.0,
    )
    assert stake is None
    assert reason.startswith("bucket_roi_unknown(")

    # ROI == 0 bucket -> blocked.
    stake, _, reason = doctrine_decide_trade(
        p_model=0.67,
        p_market=0.50,
        cfg=cfg,
        used_fraction=0.0,
        daily_cap=1.0,
    )
    assert stake is None
    assert reason.startswith("bucket_roi_not_positive(")

    # ROI < 0 bucket -> blocked by negative ROI guardrail.
    stake, _, reason = doctrine_decide_trade(
        p_model=0.58,
        p_market=0.40,
        cfg=cfg,
        used_fraction=0.0,
        daily_cap=1.0,
    )
    assert stake is None
    assert reason.startswith("bucket_negative_roi_blocked(")

    # ROI > 0 bucket -> allowed (passes delta_min test as well).
    stake, _, reason = doctrine_decide_trade(
        p_model=0.62,
        p_market=0.40,
        cfg=cfg,
        used_fraction=0.0,
        daily_cap=1.0,
    )
    assert stake is not None
    assert reason.startswith("ok")
