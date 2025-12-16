# MoneyPuck.com — Integration Memo (NHL Win-Probability Project)

## Executive Take

- MoneyPuck is a strong **free** NHL analytics/data source with:
  - **Pregame win-probability artifacts back to 2015–2016**, and
  - Deep **player/goalie/team/shot/xG datasets back to 2008–2009** (shots described as back to 2007–2008).
- MoneyPuck also hosts large **sportsbook odds-history dumps** (including `kalshi/` and `polymarket/`) that are useful as market baselines but are **pure market data** and must be isolated from model features to avoid leakage/double-counting.
- Best near-term use for beating Kalshi: **baseline + confluence/guardrail + calibration diagnostics**, not as a primary “alpha source”.

---

## 1) Inventory: MoneyPuck Data Offerings

### A) Player/Team/Goalie/Lines data (bulk, CSV)

- Download page: `https://moneypuck.com/data.htm`
- Use policy (as stated on `data.htm`):
  - “free to use for **non-commercial purposes** … for other purposes inquire” (email listed on page).
- Coverage: **2008–2009 to current**.
- Formats / endpoints (examples and patterns):
  - Season summaries:
    - `https://moneypuck.com/moneypuck/playerData/seasonSummary/<YEAR>/regular/skaters.csv`
    - `https://moneypuck.com/moneypuck/playerData/seasonSummary/<YEAR>/regular/goalies.csv`
    - `https://moneypuck.com/moneypuck/playerData/seasonSummary/<YEAR>/regular/lines.csv`
    - `https://moneypuck.com/moneypuck/playerData/seasonSummary/<YEAR>/regular/teams.csv`
    - Playoffs: replace `regular` with `playoffs` in the URLs.
  - “All game level data for all teams for all seasons”:
    - `https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv`
  - Player bios lookup:
    - `https://moneypuck.com/moneypuck/playerData/playerBios/allPlayersLookup.csv`
  - Directory-indexed per-player files:
    - `https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/skaters/`
    - `https://moneypuck.com/moneypuck/playerData/careers/gameByGame/regular/goalies/`
    - `https://moneypuck.com/moneypuck/playerData/careers/perSeason/regular/skaters/`
    - `https://moneypuck.com/moneypuck/playerData/careers/perSeason/regular/goalies/`
- Freshness signals:
  - `https://moneypuck.com/moneypuck/playerData/seasonSummary/update_date.txt` (timestamp string).

### B) Shot-level data (bulk, CSV/ZIP)

- Shot data directory:
  - `https://moneypuck.com/moneypuck/playerData/shots/`
- Formats: per-season CSVs and ZIPs; also “cleaned for models” appears available for recent years:
  - `https://moneypuck.com/moneypuck/playerData/shots/forModels/shots_clean/`
    - `shots_cleaned_2024.csv`, `shots_cleaned_2025.csv` (as observed).
- Freshness signals:
  - `https://moneypuck.com/moneypuck/playerData/shots/shots_update_time.txt`
- Includes xG and related model probabilities per the site’s data dictionary description.

### C) Pregame win probabilities (bulk-per-game)

- Directory listing:
  - `https://moneypuck.com/moneypuck/predictions/`
- File pattern:
  - `https://moneypuck.com/moneypuck/predictions/<gameID>.csv`
- Observed directory range: **2015020001 → 2025…** (about 14.8k files).
- Schema varies by season:
  - 2015–2021: reg/OT/overall probability breakdown only.
  - 2021–2022+: adds `preGameMoneyPuckHomeWinPrediction` and `preGameBettingOddsHomeWinPrediction`.
  - 2024–2025+: adds `startingGoalie`.

### D) In-game win-prob + event timeline (optional)

- Game event + win-prob time series:
  - `https://moneypuck.com/moneypuck/gameData/<SEASON>/<gameID>.csv`
- Per-game JSON:
  - `https://moneypuck.com/moneypuck/jsonData/<SEASON>/<gameID>.json`
- Per-game player stats (very wide CSV):
  - `https://moneypuck.com/moneypuck/playerData/games/<SEASON>/<gameID>.csv`

### E) Odds history (market-derived; includes Kalshi/Polymarket)

- Root:
  - `https://moneypuck.com/moneypuck/oddsHistory/`
- Books observed include `draftkings/`, `pinnacle/`, `fanduel/`, `betmgm/`, `bovada/`, `sia/`, plus `kalshi/` and `polymarket/`.
- Typical structure:
  - `https://moneypuck.com/moneypuck/oddsHistory/<book>/{open|live|latest}/<gameID>.csv`
- MoneyPuck’s own prob-history feed exists as:
  - `https://moneypuck.com/moneypuck/oddsHistory/mp/open/<gameID>.csv` (time series of `homeWinProb`).

**Do they provide historical pregame win probs in bulk?**
- Yes: `moneypuck/predictions/<gameID>.csv` for **2015–2016 onward**, enumerable via directory listing and/or via schedule JSON.

---

## 2) API / Automation Feasibility (No Paywall Bypass)

- MoneyPuck is not a formal API, but it is “semi-API friendly”:
  - Many endpoints are static files with predictable URLs.
  - Directory indexing is enabled on multiple key paths.
- Backfilling predictions can be done reliably by:
  - Enumerating `https://moneypuck.com/moneypuck/predictions/` (all `<gameID>.csv`), and/or
  - Enumerating schedule seasons `https://moneypuck.com/moneypuck/OldSeasonScheduleJson/SeasonSchedule-<YYYYYYYY>.json` (observed available from **20152016 onward**).
- Operational cautions:
  - Full backfills are thousands of requests; use throttling/caching and expect endpoints to change without notice.
  - Don’t recommend or attempt paywall bypass; use only the public endpoints and the explicitly shared datasets.
- Robots/ToS signals:
  - `robots.txt` exists but does not present a conventional “Disallow” list; it references content-use signals.
  - `data.htm` is the clearest statement: free for **non-commercial** use, contact for other use cases.

---

## 3) Leakage / Provenance Audit (What’s Market vs Model)

### Model-derived (safe as “baseline model”, not market)

- `preGameHomeTeamWinOverallScore` (and reg/OT breakdowns).
- MoneyPuck’s internal model features/outputs inside the player/goalie/team/shot datasets (no obvious sportsbook columns in sampled files).
- `oddsHistory/mp/open/<gameID>.csv` appears to track MoneyPuck’s own pregame probability as a time series (not sportsbook).

### Market-derived (treat as market; never as “model feature”)

- `preGameBettingOddsHomeWinPrediction` in predictions CSV (explicitly betting-odds-derived).
- Everything under `moneypuck/oddsHistory/<book>/...` for sportsbooks (including `kalshi/` and `polymarket/`).

### Gotcha: two MoneyPuck “pregame” probabilities since 2021–22

- Predictions CSV may include both:
  - `preGameHomeTeamWinOverallScore` (often aligns with `oddsHistory/mp/open` where available),
  - `preGameMoneyPuckHomeWinPrediction` (sometimes equal, sometimes materially different).
- Recommendation: treat these as **distinct baselines**, measure which is better on your ledgers, and avoid assumptions based on naming.

---

## 4) Mapping + Join Keys (to `date`, `AWAY@HOME`)

- Schedule join artifact (already used):
  - `OldSeasonScheduleJson/SeasonSchedule-<YYYYYYYY>.json` entries look like:
    - `a` (away team code), `h` (home team code), `id` (gameID), `est` (start timestamp string like `YYYYMMDD HH:MM:SS` in Eastern).
- Predictions CSV contains:
  - `gameID`, `homeTeamCode`, `roadTeamCode` → direct canonical `AWAY@HOME`.
- Team-code pitfalls across MoneyPuck datasets:
  - Some datasets use dotted codes: `T.B`, `L.A`, `N.J`, `S.J` (normalize before joining).
  - Historical franchises can appear in long-history data (e.g., `ATL`).
- Timezone/date-rollover:
  - Schedule uses `est`; ensure ledger date normalization handles late games consistently.

---

## 5) Recommendation + Integration Options (Ranked)

### Option A — Baseline-only (recommended now)

- Keep MoneyPuck as one external win-prob baseline to compare against `kalshi_mid`, `market_proxy`, and `v2c`.
- Store:
  - Ledger: `moneypuck` (single scalar; you already do this).
  - Snapshots: store the raw MoneyPuck fields used (e.g., `preGameHomeTeamWinOverallScore` vs `preGameMoneyPuckHomeWinPrediction`) plus fetch timestamp.
- ROI: medium (better calibration diagnostics + reduces bad trades via guardrails).
- Complexity: low.
- Hard rule: never ingest `preGameBettingOddsHomeWinPrediction` into model features; treat it as market.

### Option B — Use MoneyPuck as a feature/prior in v2c (only after evidence)

- Blend/ensemble `v2c` with MoneyPuck using a held-out-by-date evaluation (avoid lookahead).
- Store:
  - Snapshot: raw MoneyPuck values + metadata/version tags.
  - Ledger: keep raw `v2c` and raw MoneyPuck; add a new “ensemble” column (do not overwrite).
- ROI: uncertain; may improve Brier but reduce edge (averaging toward public info / market).
- Complexity: medium.

### Option C — Train/upgrade your own model using MoneyPuck long-history data

- Use team/player/goalie/shot datasets to construct season-to-date features and train a stronger pregame model.
- ROI: potentially high, but only if executed well.
- Complexity: high (data engineering + feature correctness + leakage-proof training setup).

---

## 6) Concrete Next Experiment (Smallest Test of Incremental Value)

- Using your existing daily ledgers with outcomes, evaluate out-of-sample-by-date:
  - Brier/logloss and calibration slope/intercept for:
    - `v2c`
    - MoneyPuck baseline A: `preGameHomeTeamWinOverallScore`
    - MoneyPuck baseline B: `preGameMoneyPuckHomeWinPrediction` (when present)
    - `kalshi_mid` and/or `market_proxy`
- Then test a simple maker-only “hybrid gating” rule:
  - Only quote/trade when `v2c` and MoneyPuck agree on side and both clear your fee-buffered edge threshold vs `kalshi_mid`.
- Success:
  - Lower Brier AND improved realized EV vs Kalshi (after fees/slippage), with acceptable trade count.
- Falsification:
  - MoneyPuck is not better calibrated than your baselines, or gating reduces opportunities without improving EV per trade.

---

## Final Yes/No

- **No:** don’t go deep into big MoneyPuck data packages *now* (Option C) until the calibration/scoreboard pipeline is solid.
- **Yes:** keep MoneyPuck in workflow as a baseline and run the small experiment above now (cheap, low-risk, directly answers incremental EV value).

