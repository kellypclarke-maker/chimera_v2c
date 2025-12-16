# Chimera v2c — Maker-Only Pre-Game Engine

> LLM/agent note: start with `bootstrap.md` and `AGENTS.md` for a full doc map and safety rules.

Chimera v2c is a stats + market ensemble that produces conservative, maker-biased plans for Kalshi game markets. It blends Elo, Four Factors/process metrics, market mids (or sharps when available), injuries, and calibration, then applies doctrine gates (edge thresholds, confluence checks, fee buffer, quarter-Kelly caps). Outputs are **plans** and optional executions—no v1 “betting ladders” or multi-rung traps.

## What’s Active
- Code: `chimera_v2c/src/` (probability, doctrine, risk, logging) and `chimera_v2c/tools/` (ETL, planning, logging, execution, backtests).
- Data: JSON inputs in `chimera_v2c/data/` (ratings, four_factors, injuries, calibration params).
- Logs: `reports/execution_logs/` (plans, executions, results). The legacy name `reports/betting_ladders` is not used.
- Ledger safety: append‑only **daily ledgers** in `reports/daily_ledgers/` (see `reports/daily_ledgers/README.md` and `OPERATOR_INSTRUCTIONS.md` for rules). There is no live master ledger; any archived master files are reference-only.

## Pipeline (Phase 1 pre-game)
1) Refresh data (as needed): ETLs for ratings/factors (`etl_nba_history.py`, `etl_nhl_factors.py`, `elo_builder*.py`, `train_ff_model.py`, `fit_calibration.py`).
2) Injuries/news (mandatory preflight): run `refresh_slate_updates.py` to pull ESPN injuries + team-filtered news for the slate, then optionally merge operator notes via LLM into `injury_adjustments.json`. Planning/logging/execution will auto-run the refresh step unless you explicitly pass `--skip-preflight`.
   - To auto-run the LLM merge from the ESPN digest, pass `--llm-injuries` to `run_daily.py`, `run_pipeline.py`, or `scheduled_injury_plan.py` (requires `OPENAI_API_KEY`).
3) Plan: `chimera_v2c/tools/run_daily.py --config <league_config> --date YYYY-MM-DD` (maker-only, one side per game).
4) Log: `chimera_v2c/tools/log_plan.py --config <league_config> --date YYYY-MM-DD` → `reports/execution_logs/`.
5) Optional execute: `chimera_v2c/tools/execute_plan.py --config <league_config> --date YYYY-MM-DD --dry-run` (add `--skip-halt` for real orders).
6) Backtest & calibration: `chimera_v2c/tools/nba_backtest.py`, `chimera_v2c/tools/nhl_backtest.py`, `chimera_v2c/tools/fit_calibration.py`.
   - For v2c calibration from daily ledgers (per league, NBA-first), use
     `chimera_v2c/tools/fit_calibration_from_daily.py` to write
     `chimera_v2c/data/calibration_params_<league>.json` and point the
     league config `calibration.path` at the resulting file.

## Quickstart (single league)
```bash
# 0) Install deps
pip install -r requirements.txt

# 1) Refresh data if stale (examples)
PYTHONPATH=. python chimera_v2c/tools/etl_nba_history.py
PYTHONPATH=. python chimera_v2c/tools/etl_nhl_factors.py --teams-csv <teams.csv> --goalies-csv <goalies.csv>
PYTHONPATH=. python chimera_v2c/tools/fit_calibration.py --league nhl --out chimera_v2c/data/calibration_params_nhl.json

# 2) Injuries/news (mandatory preflight)
# ESPN pull (injuries + team news)
PYTHONPATH=. python chimera_v2c/tools/refresh_slate_updates.py --league nhl --date YYYY-MM-DD

# Optional: LLM impact sizing from the ESPN digest (or operator notes)
PYTHONPATH=. OPENAI_API_KEY=... python chimera_v2c/tools/apply_llm_injuries_v2.py --league nba --date YYYY-MM-DD --input news.txt
PYTHONPATH=. OPENAI_API_KEY=... python chimera_v2c/tools/apply_llm_nhl_injuries.py --date YYYY-MM-DD --input news.txt

# One-step variant: refresh + LLM merge + plan (no manual packet)
PYTHONPATH=. OPENAI_API_KEY=... python chimera_v2c/tools/run_daily.py --config chimera_v2c/config/nhl_defaults.yaml --date YYYY-MM-DD --llm-injuries
 
# 3) Plan the slate
PYTHONPATH=. python chimera_v2c/tools/run_daily.py --config chimera_v2c/config/nhl_defaults.yaml --date YYYY-MM-DD
 
# 4) Log (and optionally execute)
PYTHONPATH=. python chimera_v2c/tools/log_plan.py --config chimera_v2c/config/nhl_defaults.yaml --date YYYY-MM-DD
PYTHONPATH=. python chimera_v2c/tools/execute_plan.py --config chimera_v2c/config/nhl_defaults.yaml --date YYYY-MM-DD --dry-run
```

## One-Command Runner
For a cross-league refresh + plan + backtest:  
`PYTHONPATH=. python chimera_v2c/tools/run_pipeline.py --leagues all --date YYYY-MM-DD --refresh-factors --fit-calibration --backtest-days 7`

## Files & Folders
- Configs: `chimera_v2c/config/defaults.yaml` (NBA), `nhl_defaults.yaml`, `nfl_defaults.yaml`.
- Data inputs: `chimera_v2c/data/team_ratings*.json`, `team_four_factors*.json`, `calibration_params*.json`, `injury_adjustments.json`.
- Logs: `reports/execution_logs/v2c_plan_log.json`, `v2c_execution_log.csv`, `v2c_results.csv`.
- Ledgers (append-only): canonical per‑day ledgers in `reports/daily_ledgers/` (managed via `ensure_daily_ledger.py`). No live master table exists; any archived master files are historical reference only.

## Not Included / Deprecated
- v1/v7 “betting ladder” or “value-trap ladder” workflows are **not part of v2c**.
- Legacy ladder docs/tools are out of scope for v2c operations.
- If you encounter legacy names (e.g., `betting_ladders` in comments), treat them as historical; the active system produces plans and maker-only orders via the v2c tools above.

## Safety Reminders
- Never overwrite ledgers; follow `OPERATOR_INSTRUCTIONS.md`.
- Maker-only by default; real orders require explicit `--skip-halt`.
- Keep JSON inputs fresh before planning; stale injuries/factors reduce accuracy.
