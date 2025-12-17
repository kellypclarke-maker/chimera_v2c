# Operator Instructions (Critical Safety)

## Canonical Ledgers
- **Daily ledgers (single source of truth):**
  - `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv` are the canonical records for model probabilities, Kalshi mids, and outcomes.
  - Treat each file as append‑only: you may add new rows or fill blank cells, but never bulk‑overwrite existing rows or probabilities.
  - Daily ledgers must follow the canonical schema documented in `reports/daily_ledgers/README.md`
    (no sportsbook moneyline columns and no MoneyPuck provenance columns in the daily ledgers).
  - Timestamp/provenance for external baselines should live in `reports/market_snapshots/` (not in the daily ledgers).
- **Lock finished days:**
  - When a daily ledger is finalized, create `reports/daily_ledgers/locked/YYYYMMDD.lock` to prevent accidental edits. Do not modify locked dates unless a human passes an explicit override.
- **No live master ledger:**
  - The legacy, historical game-level master `reports/specialist_performance/game_level_ml_master.csv` is read-only reference; do not mutate or rebuild it in normal workflows.
- **Master ledger (derived view):**
  - The analysis-ready master lives at `reports/master_ledger/master_game_ledger.csv` and is built via `PYTHONPATH=. python chimera_v2c/tools/build_master_ledger.py`. It snapshots the prior master and only appends/fills blanks; never edit it by hand.

## Daily Ledgers (per-date)
- Create the day’s ledger once via `PYTHONPATH=. python chimera_v2c/tools/ensure_daily_ledger.py` (writes `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv`). This will skip if the file exists.
- `build_daily_game_ledgers.py` is for **historical backfill only** (e.g., seeding 2025‑11‑19..2025‑12‑03 from the archived master). Do not use it to regenerate live days in bulk.
- Do not use shell redirection or ad‑hoc scripts to rewrite daily ledgers. Append new rows only (never delete/replace existing rows).
- If a day’s ledger has drifted from the canonical schema/formatting, use:
  - `PYTHONPATH=. python chimera_v2c/tools/format_daily_ledger.py --date YYYY-MM-DD --apply`
- Outcomes: leave `actual_outcome` blank until a game is final; do not use placeholders like `0-0 (push)`.
- The master ledger is derived; if it needs refresh, rebuild it with `build_master_ledger.py` instead of manual edits.

## Never Do
- Do not create “recover/clean/dedupe” scripts that write directly to the ledgers.
- Do not delete rows, truncate, or rewrite columns in `reports/daily_ledgers/` (the only approved in-place reformatter is `format_daily_ledger.py` for a single date).
- Do not regenerate all daily ledgers from scratch or overwrite multiple dates in one shot.
