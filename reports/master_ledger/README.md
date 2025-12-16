Master Ledger
=============

- Path: `reports/master_ledger/master_game_ledger.csv`
- Shape: one row per game with columns (intentionally identical to daily ledgers)  
  `date, league, matchup, v2c, grok, gemini, gpt, kalshi_mid, market_proxy, moneypuck, actual_outcome`
- Source of truth: derived from locked daily ledgers plus archived specialist/canonical reports; never edited by hand.
  - `market_proxy` is optional and comes from daily ledgers (e.g., populated via `backfill_market_proxy_from_odds_history.py`).
  - Probabilities are stored as rounded hundredths without a leading zero (e.g., `.85`) and use `NR` for missing values.
  - `moneypuck` is NHL-only; non-NHL rows use `NR`.

By-League Views (Derived)
------------------------
`build_master_ledger.py` also writes league-specific, analysis-friendly ledgers under:
- `reports/master_ledger/by_league/nba_game_ledger.csv` (no `moneypuck` column)
- `reports/master_ledger/by_league/nhl_game_ledger.csv` (includes `moneypuck`)
- `reports/master_ledger/by_league/nfl_game_ledger.csv` (no `moneypuck` column)

Workflow
--------
- Build/update via `PYTHONPATH=. python chimera_v2c/tools/build_master_ledger.py [--start-date YYYY-MM-DD --end-date YYYY-MM-DD]`.
- The tool snapshots the prior master to `reports/master_ledger/snapshots/` before writing.
- Safety semantics:
  - Probabilities are append-only (only fills blanks / `NR`).
  - If a daily ledger is explicitly corrected (human-approved), rebuild the master with `--allow-overwrite-locked` so the master matches the canonical per-day files.
  - `actual_outcome` may be blank until a game is final; it may be corrected when daily ledgers are corrected.
  - `actual_outcome` is allowed to be cleared only when the previous value was an obvious non-final placeholder (e.g., `0-0 (push)`).
  - Daily ledgers are the canonical writers; the master is a convenience table for analysis and calibration.

Safety
------
- Do not edit the master CSV manually; use the tool so append-only and snapshot guards stay in effect.
- Lock daily ledgers when finished and treat them as immutable; rebuild the master from those locks if recovery is needed.
