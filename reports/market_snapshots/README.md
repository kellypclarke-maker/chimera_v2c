Market Snapshots (External Baselines, Timestamped)
==================================================

This directory holds **timestamped, per-slate external baseline snapshots** used
to evaluate and calibrate Chimera v2c against:

- Kalshi mids (home-implied)
- Sportsbook consensus (moneylines + no‑vig implied probability)
- NHL baselines like MoneyPuck

These files are meant to keep the daily ledgers readable while preserving full
provenance for “what did we know at time T?” analysis.

Location and Naming
-------------------
- Directory: `reports/market_snapshots/`
- File pattern:
  - `YYYYMMDD_<league>_external_snapshot_<HHMMSSZ>.csv`
    - Example: `20251213_nhl_external_snapshot_193000Z.csv`

Recommended Workflow
--------------------
- Create a snapshot for the slate and (optionally) fill blank daily-ledger cells:
  - `PYTHONPATH=. python chimera_v2c/tools/external_snapshot.py --league nhl --date YYYY-MM-DD --apply`

Daily Ledger Relationship
-------------------------
- Daily ledgers remain the canonical per-day records under `reports/daily_ledgers/`.
- Snapshot files store richer market metadata (timestamps, IDs, raw lines) and can
  be joined to daily ledgers by `(date, league, matchup)`.
- Snapshot tools must never overwrite existing daily-ledger probabilities; they
  may only add rows or fill blank cells.

Note: keep the canonical daily ledgers minimal/readable — do **not** store
MoneyPuck provenance fields in `reports/daily_ledgers/`. Store provenance
(`moneypuck_game_id`, `moneypuck_starting_goalie`, fetch timestamp) in these
snapshot CSVs only.

Schema (Current)
----------------
Columns may evolve, but the intent is:
- Keys:
  - `date` (YYYY-MM-DD)
  - `league` (nba|nhl|nfl)
  - `matchup` (AWAY@HOME)
  - `snapshot_ts` (UTC ISO timestamp, `...Z`)
- Kalshi:
  - `kalshi_mid` (home-implied mid prob)
  - `kalshi_ticker_home_yes` (optional)
  - `kalshi_yes_bid_home` / `kalshi_yes_ask_home` (optional, cents)
- Sportsbooks (Odds API history):
  - `books_home_ml` / `books_away_ml` (median US moneylines across books)
  - `market_proxy` (median no‑vig implied home win probability)
- MoneyPuck (NHL):
  - `moneypuck` (home win prob)
  - `moneypuck_game_id` (MoneyPuck gameID)
  - `moneypuck_starting_goalie` (MoneyPuck goalie flag)
