# chimera_v2c – Engine Overview

This directory contains the active v2c maker-only engine: stats + market + injuries + calibration → doctrine gates → planned orders. There is no v1 “ladder” workflow; outputs are plans and optional maker executions logged to `reports/execution_logs/`.

## Entry Points
- Plan a slate: `PYTHONPATH=. python chimera_v2c/tools/run_daily.py --config <league_config> --date YYYY-MM-DD [--skip-preflight] [--llm-injuries]`
  - `--llm-injuries` runs an OpenAI-based injury impact merge from the ESPN digest into `chimera_v2c/data/injury_adjustments.json` (requires `OPENAI_API_KEY`).
- Log plan: `PYTHONPATH=. python chimera_v2c/tools/log_plan.py --config <league_config> --date YYYY-MM-DD`
- Optional execute: `PYTHONPATH=. python chimera_v2c/tools/execute_plan.py --config <league_config> --date YYYY-MM-DD --dry-run`
- Rule A (taker) track (separate from maker-only v2c): `PYTHONPATH=. python chimera_v2c/tools/run_rule_a_daily.py plan|reconcile --date YYYY-MM-DD` (see `chimera_v2c/tools/README.md`).
- Refresh factors/ratings: ETLs under `chimera_v2c/tools/` (e.g., `etl_nba_history.py`, `etl_nhl_factors.py`, `elo_builder*.py`, `train_ff_model.py`).
- Calibration/backtest: `fit_calibration.py`, `nba_backtest.py`, `nhl_backtest.py`.
  - For v2c per-league calibration from daily ledgers (e.g., NBA), use
    `fit_calibration_from_daily.py` to produce
    `chimera_v2c/data/calibration_params_<league>.json` and reference it
    from the league’s `calibration.path` in the config.
- One-command multi-league: `chimera_v2c/tools/run_pipeline.py --leagues all --date YYYY-MM-DD --refresh-factors --fit-calibration --backtest-days 7`.

## Data Inputs
- Ratings: `chimera_v2c/data/team_ratings*.json`
- Factors: `chimera_v2c/data/team_four_factors*.json`
- Injuries: `chimera_v2c/data/injury_adjustments.json`
- Calibration: `chimera_v2c/data/calibration_params*.json`

## Outputs
- Plans/exec/results: `reports/execution_logs/` (`v2c_plan_log.json`, `v2c_execution_log.csv`, `v2c_results.csv`)
- Ledgers (append-only, see `OPERATOR_INSTRUCTIONS.md`): canonical per‑day ledgers in `reports/daily_ledgers/`; no live master ledger is maintained (any archived master files are historical reference only).

## Doctrine Highlights
- Maker-only, one side per game, quarter-Kelly with per-game and daily caps.
- Confluence gap/edge thresholds live in `chimera_v2c/config/*.yaml`.
- Calibration applied via `calibration_params*.json`; injuries/news refresh is mandatory preflight (enforced unless you explicitly pass `--skip-preflight`).

## Deprecated / Legacy
- No “betting ladders” or multi-rung traps are part of v2c. Any legacy naming is historical only.
