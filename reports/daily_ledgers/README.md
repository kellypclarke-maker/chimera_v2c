Daily Game Ledgers (Per-Date, Immutable)
========================================

This directory holds the **canonical per-day ledger** CSVs used for calibration,
backtests, and model grading.

Location and Naming
-------------------
- Directory: `reports/daily_ledgers/`
- File pattern (canonical): `YYYYMMDD_daily_game_ledger.csv`

Canonical Schema (Required)
---------------------------
Each daily ledger is intended to be **human-readable** and must use a fixed,
minimal schema (one game per row) with **no blank cells in the probability-like columns**
(outcomes may be blank until final).

Columns (in order)
------------------
`date, league, matchup, v2c, grok, gemini, gpt, kalshi_mid, market_proxy, moneypuck, actual_outcome`

Formatting rules
----------------
- **Probability-like columns** (`v2c|grok|gemini|gpt|kalshi_mid|market_proxy|moneypuck`):
  - Round to the **hundredths** place.
  - Drop the leading zero (`0.85` → `.85`).
  - Use the sentinel `NR` (not blank) when a value is missing.
- **Final score (outcomes)**:
  - `actual_outcome` is the final score string when the game is final:
    `AWAY <away_score>-<home_score> HOME` (example: `CHA 119-111 CLE`).
  - If the game is not final yet, `actual_outcome` must be **blank** (not `NR`).
  - Do not store placeholders like `0-0 (push)` or `(pending)` in canonical ledgers.
- **NHL-only MoneyPuck**:
  - `moneypuck` is MoneyPuck’s *pregame* home win probability (NHL only).
  - For non-NHL rows, `moneypuck` must be `NR`.
- **No noisy metadata columns in daily ledgers**:
  - Do not store sportsbook moneylines (`books_*`) or MoneyPuck provenance fields
    (`moneypuck_ts`, `moneypuck_game_id`, `moneypuck_starting_goalie`) in daily ledgers.
  - Put rich/timestamped provenance in `reports/market_snapshots/` instead.

Safe Workflow (Do This)
-----------------------
- **One file per date**:
  - For a new date, create the ledger via:
    - `PYTHONPATH=. python chimera_v2c/tools/ensure_daily_ledger.py --date YYYY-MM-DD`
      (or let it default to today).
  - This writes `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv` (header-only if no rows exist yet for that date).
  - To seed today’s slate from the v2c planner (and capture Kalshi mids used by the plan), run:
    - `PYTHONPATH=. python chimera_v2c/tools/fill_daily_ledger_from_plan.py --date YYYY-MM-DD --config <league_config> --apply`
      - This fills blank `v2c` + `kalshi_mid` cells (append-only).
- **Append-only mindset**:
  - Once a daily ledger exists for a date, treat it as append-only:
    - You may *add* new rows (e.g., new games) if needed.
    - You may *fill blank cells* in existing rows (e.g., add `gpt` or outcomes).
    - You should **not** delete rows or rewrite probabilities that are already set.
    - Exception (rare): if a probability was recorded incorrectly (e.g., side flipped / bad market link), make a small, surgical correction only with explicit human approval and snapshot the ledger first (tools like `format_daily_ledger.py` and `build_master_ledger.py` rely on snapshots for auditability).
- **Lock finished days**:
  - When a day is finalized, create `reports/daily_ledgers/locked/YYYYMMDD.lock` to prevent accidental edits. Only modify a locked date with an explicit, human-approved override.
- **Small, surgical edits only**:
  - For corrections or new model columns, edit the specific cells in the specific
    file for that date.
  - Do not run bulk scripts that regenerate multiple days in one shot.
- If a ledger has drifted from the canonical schema/formatting, use:
  - `PYTHONPATH=. python chimera_v2c/tools/format_daily_ledger.py --date YYYY-MM-DD --apply`
    - This tool snapshots the prior file and rewrites **only** that day to match the canonical format.

Things That Are Not Allowed
---------------------------
- **No bulk overwrites**:
  - Do not call `build_daily_game_ledgers.py` with a wide date range and `--overwrite`
    pointed at this directory.
  - Do not use shell redirection or ad-hoc scripts to rewrite an entire
    `daily_game_ledger_<date>.csv` file from scratch.
- **No ledger cleaning/deduping here**:
  - Do not create scripts that “clean”, “dedupe”, or “rebuild” all daily ledgers.
  - Any one-off transformations should work on a *copy* in a separate directory.
  - The only approved in-place reformatter is `chimera_v2c/tools/format_daily_ledger.py` for a single date.
- **No moving the canonical files out of this directory**:
  - Historical / archived ledgers should be copied into `archive/...` as needed,
    not removed from here.
- **Master ledger is derived**:
  - The analysis-friendly master table is built via
    `PYTHONPATH=. python chimera_v2c/tools/build_master_ledger.py` and lives at
    `reports/master_ledger/master_game_ledger.csv`. Do not hand-edit it; rebuild
    it from the daily ledgers if needed.

How This Interacts With Other Tools
-----------------------------------
- Source of truth for probabilities:
  - The **per‑day files in this directory are canonical**. All grading, calibration,
    and analysis should read from `reports/daily_ledgers/*.csv`.
  - A legacy game‑level master lives at
    `archive/specialist_performance/archive_old_ledgers/game_level_ml_master.csv`;
    it is archived and only used to backfill historical days before 2025‑12‑04.
- Writers:
  - `chimera_v2c/tools/ensure_daily_ledger.py` is the recommended entry point for
    daily use; it will create today’s file if missing and otherwise leave it alone.
  - `chimera_v2c/tools/build_daily_game_ledgers.py` is reserved for **historical
    backfill** (e.g., 2025‑11‑19..2025‑12‑03) and must be pointed at the archived
    master explicitly. Do not use it to rewrite live days in bulk.
  - `chimera_v2c/tools/backfill_market_proxy_from_odds_history.py` can add the optional
    `market_proxy` column and fill blank cells only using sportsbook-implied probabilities
    from Odds API history. It respects lockfiles unless `--allow-locked` (or `--force`) is
    supplied and never touches `kalshi_mid`. Overwriting existing `market_proxy` values requires
    explicit `--overwrite-existing` (or `--force`).
  - `chimera_v2c/tools/backfill_moneypuck_pregame.py` can fill the optional NHL `moneypuck` column
    (append-only). Rich MoneyPuck metadata belongs in `reports/market_snapshots/`.
  - `chimera_v2c/tools/external_snapshot.py` can capture timestamped external baselines (Kalshi + books + MoneyPuck)
    into `reports/market_snapshots/` and optionally fill blank daily-ledger market cells (append-only).
  - `chimera_v2c/tools/backfill_kalshi_mid_from_candlesticks.py` can backfill historical `kalshi_mid` values
    at/just before ESPN game start (or T-30m) using Kalshi candlesticks (append-only; fills only blank/`NR`).
  - `chimera_v2c/tools/build_game_level_ml_table.py` and the legacy master ledger
    are for archival maintenance only and are not part of the day‑to‑day workflow.
- Readers:
  - Tools like `rolling_calibration.py`, `nba_backtest.py`, and `nhl_backtest.py`
    expect to read from `reports/daily_ledgers/*.csv` and must not write here.

Human Checklist Before Editing
------------------------------
1. Confirm you are working on the **correct date file**:
   - `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv`
2. Confirm that any changes are:
   - Adding new rows, or
   - Filling blank cells (e.g., `gpt`, `actual_outcome`), not changing existing
     probabilities that are already set.
3. Only in rare historical backfill cases should you regenerate from the archived
   master, and never for 2025‑12‑04 or later:
   - Work on a copy of the daily ledger in a separate directory.
   - Use `build_daily_game_ledgers.py` for that specific date with explicit intent.

Market Proxy Column
-------------------
- An optional `market_proxy` column may appear in daily ledgers to store sportsbook-implied
  home win probabilities when Kalshi mids are missing.
- Only fill blank `market_proxy` cells (append-only); do not overwrite existing values unless
  a human explicitly approves and you pass `--overwrite-existing` (or `--force`).
- Use `chimera_v2c/tools/backfill_market_proxy_from_odds_history.py` to add this column (if
  absent) and fill it safely from the Odds API history snapshot.

Market Timestamp Policy (Kalshi vs Proxy)
----------------------------------------
- For EV/Brier comparisons, `kalshi_mid` and `market_proxy` should represent the **same anchor timestamp**
  relative to ESPN’s scheduled start time (e.g., **T-30 minutes**).
- The column names remain `kalshi_mid` and `market_proxy` (canonical schema); the timestamp convention is a workflow rule.
- Tools:
  - Kalshi candlestick backfill supports `--minutes-before-start` (default T-30).
  - Odds proxy alignment supports `--minutes-before-start` to match the same anchor (e.g., 30 for T-30).

MoneyPuck Column (NHL Optional)
------------------------------
- An optional `moneypuck` column may appear in NHL daily ledgers to store MoneyPuck’s published
  pregame home win probability (`preGameMoneyPuckHomeWinPrediction`).
- Use `chimera_v2c/tools/backfill_moneypuck_pregame.py` to fill blank `moneypuck` cells (append-only).
- Store MoneyPuck provenance fields (gameID, goalie indicator, fetch timestamp) in `reports/market_snapshots/`
  via `chimera_v2c/tools/external_snapshot.py`.

Explicit "No Report" Marker (NR)
--------------------------------
- Sometimes a game will legitimately have no specialist report for one of `gemini|grok|gpt`
  (e.g., the report was never produced or cannot be recovered).
- After you have **manually confirmed** the report truly does not exist, you may fill the
  corresponding blank model cell with the sentinel value `NR` ("No Report") so future audits
  do not treat it as an unreviewed blank.
- If a specialist report exists but was generated **after the game started/finished** (post-start
  leakage / hindsight bias), treat it as non-actionable and mark the model cell as `NR` (take a
  snapshot first, and keep the canonical report file for auditability).
- Do **not** use `NA`, `NaN`, or `N/A` for this purpose; many tools parse those as missing/blank.
- Prefer not to mark `market_proxy` or `actual_outcome` as `NR` since those should be backfilled.
- `actual_outcome` should be blank until final and then backfilled to a final score string (never `NR`).
- For `kalshi_mid`, prefer backfilling when possible; if you confirm historical mids were not
  captured and cannot be recovered, marking `kalshi_mid` as `NR` is acceptable.
- Use `PYTHONPATH=. python chimera_v2c/tools/audit_ledger_completeness.py` to generate a
  per-game/per-cell blank report under `reports/thesis_summaries/`. After confirming a missing
  report, create a mark-list CSV (subset of `ledger_blanks_cells_*.csv`) and apply with
  `--mark-from ... --apply` (respects lockfiles unless `--force`).

Outcome Backfill (Final Games Only)
-----------------------------------
- Use `PYTHONPATH=. python chimera_v2c/tools/fill_missing_daily_outcomes.py` to fill blank
  `actual_outcome` cells **only when ESPN marks the game as final** (`status.state == post`).
- If a day’s `actual_outcome` cells were accidentally filled too early (e.g., `0.0-0.0 (push)` or
  otherwise incorrect scores), rerun with:
  - `PYTHONPATH=. python chimera_v2c/tools/fill_missing_daily_outcomes.py --date YYYY-MM-DD --overwrite-existing`
  - This snapshots before writing and only allows overwriting `actual_outcome` (no probabilities).

LLM / Agent Guidance
--------------------
- On startup, **read this file** before touching anything in `reports/daily_ledgers`.
- Never write scripts that:
  - Iterate over all dates and regenerate every daily ledger, or
  - Overwrite existing daily ledgers in bulk.
- When asked to “update today’s ledger”, operate on the single
  `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv` file and perform only the minimal, targeted edits
  requested (e.g., filling `gpt` cells for games that already exist).
