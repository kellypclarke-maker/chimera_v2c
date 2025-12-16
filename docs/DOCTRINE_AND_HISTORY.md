# Chimera v2c Doctrine & History (v8.2)

## What v2c Is (Active)
- **Purpose:** Generate conservative, maker-only pre-game plans for Kalshi game markets using a stats + market ensemble (Elo, Four Factors/process metrics, market mids/sharps, optional injuries, calibration).
- **Doctrine:** Edge thresholds + confluence gap + fee buffer + quarter-Kelly with per-game and daily caps; one side per game; maker pricing only.
- **Outputs:** Plans and optional executions logged to `reports/execution_logs/` (`v2c_plan_log.json`, `v2c_execution_log.csv`, `v2c_results.csv`). Legacy names like `betting_ladders` are historical only.
- **Ledgers:** Canonical per-day ledgers live in `reports/daily_ledgers/` (append-only, one file per date). There is no live master ledger; any archived master files are historical reference only. v2c does not write to v1 manual ledgers.

## Active Runbook (Phase 1 Pre-Game)
1) **Refresh data (as needed)**  
   - Ratings/FF: `etl_nba_history.py`, `etl_nhl_factors.py`, `elo_builder*.py`, `train_ff_model.py` (league-specific configs in `chimera_v2c/config/`).
   - Calibration (recommended): `fit_calibration_from_daily.py` writes `chimera_v2c/data/calibration_params_<league>.json` (daily ledgers do not store `v2c_raw`).
2) **Injuries/news (optional but recommended; hybrid mode)**  
   - `refresh_slate_updates.py` auto-fetches ESPN injuries + team-filtered news and writes the digest:
     - `chimera_v2c/data/news_<date>_<league>.txt`
   - Apply LLM-sized injury deltas (writes `chimera_v2c/data/injury_adjustments.json`):
     - `apply_llm_injuries_v2.py` (NBA/NFL) or `apply_llm_nhl_injuries.py` (NHL)
     - Tools also emit a “what changed” JSON + file snapshots under `reports/injury_snapshots/`.
3) **Plan**  
   - `PYTHONPATH=. python chimera_v2c/tools/run_daily.py --config <league_config> --date YYYY-MM-DD [--llm-injuries] [--skip-preflight]`
4) **Log**  
   - `PYTHONPATH=. python chimera_v2c/tools/log_plan.py --config <league_config> --date YYYY-MM-DD [--llm-injuries]`
5) **Optional execute**  
   - `PYTHONPATH=. python chimera_v2c/tools/execute_plan.py --config <league_config> --date YYYY-MM-DD --dry-run [--llm-injuries]` (add `--skip-halt` for real orders).
6) **Doctrine rulebooks (optional)**  
   - Build per-league ROI-by-bucket guardrails from plan logs + daily ledger outcomes:
     - `PYTHONPATH=. python chimera_v2c/tools/build_roi_by_bucket_guardrails.py --league <league> --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
     - Default bucket width is `0.05` (bucket labels like `[0.55,0.60)`), matching `doctrine.negative_roi_buckets`.
   - The per-league CSV is referenced by `doctrine.bucket_guardrails_path` (defaults point to `reports/roi_by_bucket_<league>.csv`).
   - Guardrail modes:
     - **Negative-only blocks (default behavior):** set `doctrine.enable_bucket_guardrails: true` to block buckets where `roi_estimate < 0` (and/or any buckets listed in `doctrine.negative_roi_buckets`).
     - **Positive-only allowlist (strict):** set both `doctrine.enable_bucket_guardrails: true` and `doctrine.require_positive_roi_buckets: true` to trade *only* when the bucket has `roi_estimate > 0` (unknown buckets are blocked).
7) **Backtest / calibration**  
   - `nba_backtest.py`, `nhl_backtest.py`, `fit_calibration_from_daily.py` (store params in `chimera_v2c/data/calibration_params*.json`).
8) **Multi-league one-shot**  
   - `chimera_v2c/tools/run_pipeline.py --leagues all --date YYYY-MM-DD --refresh-factors --fit-calibration --backtest-days 7 --skip-preflight`

## Data & Outputs
- **Inputs:** `chimera_v2c/data/team_ratings*.json`, `team_four_factors*.json`, `injury_adjustments.json`, `calibration_params*.json`.
- **Market data:** Kalshi markets are fetched from the **public** API by default (no private key needed) for planning and ledger-building. Private Kalshi credentials are only required for trading/portfolio calls (e.g., `execute_plan.py`, portfolio helpers).
- **Outputs:** Plan/execution/result logs in `reports/execution_logs/`; backtests in `execution_logs/` or tool-specified paths.
- **Ledger Safety:** Per-day ledgers in `reports/daily_ledgers/` are append-only and are the canonical record; only sanctioned builders may write them (see `OPERATOR_INSTRUCTIONS.md`). Any archived master files are read-only historical artifacts.

## Not Part of v2c
- v1/v7 “betting ladder”, “value-trap ladder”, or multi-rung trap doctrine.
- Legacy ladder scripts/tools. Treat any ladder language as historical context only.
- Auto-grading of v1 specialist ledgers; v2c operates on its own JSON inputs and plan logs.

## History Snapshot
- v1 (“Project Chimera”): manual ledger + ladder execution (deprecated).
- v2 (“Aeternus v2c”): current stats+market engine with maker-only plans, doctrine gating, calibration, and league-specific factors (NBA/NHL; NFL Elo-only stub).
