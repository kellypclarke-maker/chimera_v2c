# Tools Overview (Active Only)

All tools must be documented here. Anything not listed is treated as legacy and should live in archive.

## Core Daily Flow
- `refresh_slate_updates.py` — Mandatory preflight: pulls ESPN injuries (player-level) + team-filtered news for the slate (nba|nhl|nfl), writes `chimera_v2c/data/raw_injuries.json`, `chimera_v2c/data/raw_news.json`, and a digest `chimera_v2c/data/news_<date>_<league>.txt` (also mirrored into `reports/llm_packets/<league>/<YYYYMMDD>/news.txt`). Ensures `chimera_v2c/data/injury_adjustments.json` has per-team entries for the slate date without overwriting existing non-zero deltas. `run_daily.py`, `log_plan.py`, and `execute_plan.py` auto-run this step unless `--skip-preflight` is passed.
- `run_daily.py` — Plan the slate (stats + market ensemble) for a given date/league. Uses Kalshi **public** markets by default (`execution.use_private: false`); only uses private/trading API when `use_private: true` and private creds are present. Supports `--llm-injuries` to apply LLM-derived injury deltas (from the ESPN digest) before planning.
- `log_plan.py` — Persist plan log rows (per market side) to `reports/execution_logs/v2c_plan_log.json` for analysis/backtests. Includes `league`, `selected`, and `stake_fraction` fields; supports `--llm-injuries` to apply LLM-derived injury deltas (from the ESPN digest) before logging.
- `execute_plan.py` — Execute maker-only orders (dry-run by default). Supports `--llm-injuries` to apply LLM-derived injury deltas (from the ESPN digest) before execution.
- `kalshi_ws_listener.py` — Stream Kalshi mids to `data/ws_mids.json` (overlay in planner).
- `fetch_injuries.py` — ESPN injuries fetcher; writes `data/injury_adjustments.json` and `data/raw_injuries.json`.
- `fetch_injuries_nhl.py` — ESPN HTML scrape fallback for NHL injuries (seed text; prefer LLM merge for impact sizing).
- `news_watcher.py` — Add manual notes/halts and optional injury deltas.
- `live_guardrails.py` — Thesis-breaker halts using live scoreboard context.
- `ingest_results.py` — Grade results for a date/league into reports.

## Ratings / Data Prep
- `etl_nba_history.py` — Build `data/chimera.db` with NBA games + team stats.
- `elo_builder.py` — Derive Elo ratings into `data/team_ratings.json` from DB (NBA).
- `elo_builder_nhl.py` — Derive Elo ratings into `data/team_ratings_nhl.json` from ESPN scoreboard (with decay).
- `train_ff_model.py` — Train Four Factors logistic model -> `data/ff_model.json`.
- `export_db_to_json.py` — Hydrate JSON bridge (`team_four_factors.json`, etc.) from DB.
- `prepare_data.py` — Quick NBA ratings refresh (season-based).
- `fetch_sharp_odds.py` — Ingest sharp odds into the DB.

## Ledgers (Append-Safe Only)
- `ensure_daily_ledger.py` — Step #1 each day; create daily ledger for today (header-only if empty). Never overwrites unless `--overwrite --force`.
- `fill_daily_ledger_from_plan.py` — Append-safe seeder: runs the v2c planner for a league/date and fills blank/`NR` `v2c` + `kalshi_mid` cells (adds missing rows with other probability-like columns initialized to `NR`; respects `reports/daily_ledgers/locked/` unless `--force`).
- `format_daily_ledger.py` — Canonical formatter: rewrites a *single date* ledger into the minimal, operator-readable schema (drops noisy columns, formats probabilities to `.85`, fills blank probability cells with `NR`, leaves `actual_outcome` blank until final), snapshots before writing; use `--force` to operate on locked days only with explicit human intent.
- `backfill_v2c_raw_from_plan_json.py` — Deprecated: v2c_raw is no longer part of the canonical daily-ledger schema; do not use in normal workflows.
- `build_daily_game_ledgers.py` — Historical backfill: write per-day ledger(s) from the archived master (e.g., 2025-11-19..2025-12-03). Uses lockfiles; overwrite requires `--overwrite --force` and snapshots the prior file.
- `build_model_wr_by_league.py` — Read-only accuracy/Brier summary by league and overall from `reports/daily_ledgers/*_daily_game_ledger.csv` -> `reports/daily_ledgers/model_wr_by_league.csv` (models: `v2c|gemini|grok|gpt|kalshi_mid|market_proxy|moneypuck`).
- `build_model_reliability_by_p_bucket.py` — Read-only reliability/Brier/EV summary by predicted p buckets for each model vs `kalshi_mid` from `reports/daily_ledgers/*_daily_game_ledger.csv` -> `reports/daily_ledgers/model_reliability_by_p_bucket.csv` (models: `v2c|gemini|grok|gpt|kalshi_mid|market_proxy|moneypuck`).
- `build_ev_summary_by_league.py` — Read-only EV vs Kalshi mid + Brier summary by league from daily ledgers -> `reports/daily_ledgers/ev_brier_by_league.csv`.
- `fill_missing_daily_outcomes.py` — Use ESPN scoreboards to fill blank `actual_outcome` cells in daily ledgers (final games only; does not touch probabilities). Supports `--overwrite-existing` to correct non-blank outcomes to ESPN finals and clear non-final placeholders back to blank, respects lockfiles unless `--force`, and snapshots before writing.
- `build_master_ledger.py` — Derive the master ledger (`reports/master_ledger/master_game_ledger.csv`) from daily ledgers, snapshots, archived specialist/canonical CSVs, and plan logs. Daily ledgers are authoritative for their keys; snapshots the prior master before writing. By default it refuses to overwrite existing probability cells; if you intentionally corrected a daily ledger and need the master to reflect it, rerun with `--allow-overwrite-locked` (explicit human intent only). Also writes by-league views under `reports/master_ledger/by_league/` (NBA/NFL omit `moneypuck`).
- `build_game_level_ml_table.py` — Legacy: append/fill the archived master `archive/specialist_performance/archive_old_ledgers/game_level_ml_master.csv` for historical analysis only.
- `build_complete_game_ledger.py` — Legacy: filter the master to rows with v2c, Gemini, Grok, and market mids -> `game_level_ml_complete.csv`.
- `snapshot_game_level_ledger.py` — Legacy: snapshot the master ledger to `reports/specialist_performance/snapshots/`.
- `preflight_check.py` — Guardrail: ensure injuries are fresh and game start times are within 30 minutes before planning/execution.
- `rolling_calibration.py` — Non-destructive accuracy/Brier summary over the last N days of daily ledgers.
- `shadow_no_edge_log.py` — Shadow log of per-game diagnostics/reasons when no orders are placed (no quotes, low edge, guardrails).
- `apply_llm_injuries_v2.py` — LLM-based injury delta generator (safe merge into injury_adjustments.json); audits raw response to `reports/specialist_reports/raw/` and writes change logs/snapshots under `reports/injury_snapshots/`. Team codes are normalized via `team_mapper`.
- `apply_llm_nhl_injuries.py` — NHL-specific LLM injury deltas (goalie/role aware) into injury_adjustments.json (safe merge); audits raw response to `reports/specialist_reports/raw/` and writes change logs/snapshots under `reports/injury_snapshots/`. Team codes are normalized via `team_mapper`.
- `etl_nhl_factors.py` — Build NHL process factors (xGF%, HDCF%, PP/PK z-scores, goalie rating, oi_sh%) from local CSVs into `team_four_factors_nhl.json`.
- `fit_calibration.py` — Fit Platt calibration parameters from ledger outcomes into calibration_params_*.json.
- `fit_calibration_from_daily.py` — Fit Platt calibration parameters from daily ledgers for a model probability column (e.g., `v2c`, `grok`, `gemini`, `gpt`) and write JSON for pipeline/offline use.
- `fit_anchor_offset_calibration.py` — Fit additive-bias (“handicap”) calibration stats from daily ledgers for a probability column (default: `grok`) and break them out by rulebook quadrants (A/B/C/D) vs `kalshi_mid`; writes `chimera_v2c/data/anchor_offset_calibration_<league>_<model>.json` and snapshots prior outputs under `reports/calibration_snapshots/`.
- `nhl_backtest.py` — Read-only ROI/backtest by p bucket from `reports/execution_logs/v2c_plan_log.json` + daily ledgers (one side per game by default; `--all-sides` to include both).
- `nba_backtest.py` — Read-only ROI/backtest by p bucket from `reports/execution_logs/v2c_plan_log.json` + daily ledgers (one side per game by default; `--all-sides` to include both).
- `build_roi_by_bucket_guardrails.py` — Read-only doctrine helper: build per-league `reports/roi_by_bucket_<league>.csv` guardrails from plan logs + daily ledgers (writes `roi_estimate` only when bucket has enough samples; can be used for negative-only bucket blocks or strict positive-only allowlists via `doctrine.require_positive_roi_buckets`).
- `llm_tune_nhl_weights.py` — LLM-assisted suggestions for NHL factor weights (log-only; no auto-write).
- `run_pipeline.py` — Unified runner to refresh/fit/plan/log/backtest across leagues (nba|nhl|nfl). Supports `--llm-injuries` with `--llm-injuries-source espn|moneypuck` (NHL) to apply LLM-derived injury deltas before planning; `--moneypuck-injuries` maintains a canonical MoneyPuck injury snapshot + diff logs.
- `scheduled_injury_plan.py` — Sleep-until-start scheduler (ESPN slate refresh + optional `--llm-injuries` merge; NHL can add `--moneypuck-injuries --llm-injuries-source moneypuck` to run LLM deltas only when the MoneyPuck snapshot changes; can also call run_daily/log_plan/ensure_daily_ledger).
- `backfill_market_proxy_from_odds_history.py` — Append-safe sportsbook backfill using Odds API history. Supports either a fixed per-date snapshot time (`--snapshot-time`) or per-game T-minus (`--minutes-before-start`, e.g. 30 for T-30). Fills blank/`NR` `market_proxy` (no‑vig implied home prob); `kalshi_mid` untouched. Respects locks; use `--allow-locked` to include locked ledgers, and `--overwrite-existing`/`--force` only with explicit intent.
- `align_market_proxy_to_game_start.py` — Align `market_proxy` to each game’s ESPN start timestamp (Odds API history), so comparisons vs `kalshi_mid` captured at the same anchor are time-aligned. Supports `--minutes-before-start` (e.g., 30 for T-30m). Dry-run by default; snapshots before writing; respects lockfiles unless `--force`. Overwriting existing `market_proxy` requires `--overwrite-existing` (explicit human intent only).
- `external_snapshot.py` — Capture timestamped external baselines (Kalshi + books + MoneyPuck) into `reports/market_snapshots/` and optionally fill blank daily-ledger market cells (`kalshi_mid`, `market_proxy`, and NHL `moneypuck`) append-only; snapshots the daily ledger before writing. By default it snapshots **open** Kalshi markets; use `--kalshi-status settled` (or legacy `closed`) or `--kalshi-status all` (open+settled) when running late.
- `backfill_kalshi_mid_from_candlesticks.py` — Append-safe historical Kalshi mid backfill for games where we did not capture live markets. Uses Kalshi `/markets/candlesticks` to compute `kalshi_mid` at/just before ESPN game start (or `--minutes-before-start`, default T-30m; short lookback fallback) and fills only blank/`NR` cells; retries 429/5xx with backoff; respects lockfiles unless `--force`, snapshots before writing.
- `backfill_moneypuck_pregame.py` — Append-safe NHL backfill of MoneyPuck `preGameMoneyPuckHomeWinPrediction` into daily ledgers as `moneypuck` (fills blanks only; snapshots the daily ledger before writing). Rich MoneyPuck metadata is stored via `external_snapshot.py` under `reports/market_snapshots/`.
- `ingest_raw_specialist_reports_v2c.py` — Preview-first specialist ingester for raw Gemini/Grok/GPT reports. Parses HELIOS blocks (and legacy `HELIOS_PREDICTION_HEADER` blocks), writes canonical per-game files (`YYYYMMDD_league_away@home_model.txt`), and fills blank/`NR` daily-ledger model cells (append-only; snapshots the daily ledger before writing; respects locks unless `--force`). If a parsed game date doesn’t match any existing daily-ledger row but the filename-inferred `MM-DD` date does (timezone/rollover), it will use the filename date for that matchup. Moves raw files to `archive/raw_processed` or `archive/raw_unparsed` on apply.
- `refresh_model_probs_from_canonicals.py` — Append-safe ledger hydrator: scans canonical specialist reports and fills blank `gemini|grok|gpt` cells using HELIOS `p_home` (preferred) or `p_true`+Winner fallback. Respects lockfiles unless `--force`.
- `add_helios_headers_to_canonical.py` — Canonical report helper: prepends missing HELIOS headers to canonical specialist reports by deriving `Winner` + `p_home` from daily ledgers when safely available (does not touch daily ledgers).
- `normalize_canonical_headers.py` — Canonical report normalizer: canonicalizes team codes in `Game:`/`Winner:` and inserts `p_home` when safely derivable (writes canonical files in place; dry-run by default, `--apply` to write).
- `validate_team_codes.py` — Read-only checker to ensure daily ledgers and canonical specialist reports use normalized team codes via `team_mapper`; exits non-zero on mismatches.

## Dossiers / Research
- `build_dossier.py` — Build per-game dossiers.
- `deep_research.py` — Deep research harness for a given game/date/league.
- `build_llm_packets.py` — Export small, slate-level CSV packets (standings/form, injuries, schedule/fatigue, odds) for a given league/date into `reports/llm_packets/` so LLM specialists can reason from structured tables instead of hallucinating records/injuries/odds. Standings packets include overall/home/road records, last-10 (W/L), and current streak; injuries packets surface per-team deltas from `injury_adjustments.json` or simple status-based heuristics, restricted to teams on the slate; use `--no-odds` to skip generating the odds/markets CSV when you don’t want Kalshi mids in the upload.
- `eval_nba_context_features.py` — R&D: read-only NBA evaluation of whether packet features (form, rest/fatigue, injuries, H2H) improve v2c accuracy. Auto-builds missing packets for each date, fits a logistic adjustment vs baseline, and writes datasets/summaries to `reports/thesis_summaries/`.
- `learn_hybrid_weights.py` — Learn simplex hybrid weights across calibrated models (v2c/grok/gemini/gpt) from the master ledger with date-grouped CV; writes `hybrid_weights_<league>.json` and summaries under `reports/thesis_summaries/`.

## Analysis / Debug
- `analyze_master_ledger.py` — Read-only accuracy/Brier summary off the master ledger.
- `build_scoreboard.py` — Read-only evaluation scoreboard from daily ledgers: writes summary + daily trend + reliability bucket CSVs (and a small Markdown preview) under `reports/thesis_summaries/`.
- `audit_ledger_parity.py` — Read-only parity audit across schedule CSVs (`reports/thesis_summaries/*_schedule_*.csv`), canonical specialist reports (`reports/specialist_reports/*`), daily ledgers, and the derived master ledger; flags phantom/misdated games, score/outcome mismatches, and canonical p_home drift. Use `--skip-master` while repairing schedule↔canonical↔daily parity.
- `audit_ledger_completeness.py` — Read-only audit of blank cells in daily ledgers (v2c/gemini/grok/gpt/kalshi_mid/market_proxy/moneypuck) with CSV outputs under `reports/thesis_summaries/`; can optionally mark *confirmed* missing cells with sentinel `NR` (append-only, respects locks unless `--force`).
- `probe_markets.py` — Quick Kalshi market probe. By default uses the **public** Kalshi API; for live GAME markets you can either:
  - Set `KALSHI_PUBLIC_BASE=https://api.elections.kalshi.com/trade-api/v2` and filter by `series_ticker` (e.g., `KXNBAGAME`), or
  - Use the higher-level helpers in `chimera_v2c/src/market_linker.py` to fetch ESPN matchups and match them to Kalshi tickers, then read home‑implied mids from the resulting `MarketQuote.mid`.
- `analyze_ev_vs_kalshi.py` — Read-only EV vs Kalshi mid and Brier analysis from daily ledgers by date window/league.
- `analyze_scheme_d.py` — Read-only Scheme D analysis (I/J-gated home-favorite fades) from daily ledgers; writes derived rule stats and backtests under `reports/ev_rulebooks/`.
- `analyze_grok_quadrant_consensus.py` — Read-only Grok-vs-Kalshi quadrant breakdown (A/B/C/D) with optional consensus sizing from v2c/gemini/gpt; supports `--confirmer-positive-bucket-only` gating; writes a CSV under `reports/thesis_summaries/`.
- `analyze_rulebook_quadrants.py` — Read-only A/B/C/D + sub-bucket (I/J/K/L/M/N/O/P) EV scan vs Kalshi mid for symmetric “fade/follow” regimes; supports edge-threshold sweeps and can emit a selected rulebook under `reports/ev_rulebooks/` (writes `quadrants_rule_stats.csv` and, in sweep mode, `quadrants_rule_stats_sweep.csv` + `quadrants_rulebook_selected.json`).
- `learn_threshold_rulebook_kalshi_mid.py` — Read-only threshold learner: sweep edge thresholds vs `kalshi_mid` for A/B/C/D (optional sub-buckets) and write a selected threshold rulebook JSON to `chimera_v2c/data/threshold_rulebook_kalshi_mid.json` (plus sweep/selected CSVs under `reports/ev_rulebooks/`). Can optionally apply offset calibration from `anchor_offset_calibration_*` files.
- `learn_grok_mid_hybrid.py` — Read-only Grok↔Kalshi-mid hybrid: fits Platt calibration for Grok and learns a shrinkage scalar `alpha` by leave-one-ledger-date-out CV to minimize Brier, using `p_hybrid = p_mid + alpha*(Platt(p_grok)-p_mid)`; writes `chimera_v2c/data/grok_mid_hybrid_<league>.json` and a CV curve CSV under `reports/thesis_summaries/`.
- `walkforward_grok_mid_hybrid.py` — Read-only walk-forward (train→test) evaluator for the Grok↔Kalshi-mid hybrid (Brier-focused): for each ledger date, trains Platt + picks `alpha` on prior days, then evaluates next-day Brier deltas vs `kalshi_mid`; writes per-league daily + summary CSVs under `reports/thesis_summaries/`.
- `walkforward_grok_mid_hybrid_backtest.py` — Read-only walk-forward (train→test) backtest: per ledger date, learns Grok Platt + `alpha` on prior days (default `--alpha-objective brier_cv`, optional `pnl_cv` / `pnl_train`), learns per-bucket edge thresholds `t*` on prior days (A/B/C/D), then evaluates next-day out-of-sample PnL vs `kalshi_mid`. Output includes side-by-side results for `grok_raw`, `grok_platt`, and the `grok_mid_hybrid`; writes a per-day CSV under `reports/thesis_summaries/`.
- `plan_grok_mid_hybrid_trades.py` — Read-only paper trade sheet for a target date: trains Grok Platt + `alpha` and learns A/B/C/D thresholds on dates < target date, then outputs per-game trade recommendations (bucket/side/EV at mid) to `reports/trade_sheets/`.
- `online_grok_mid_hybrid_walkthrough.py` — Read-only online learning curve (game-by-game): re-fits Grok Platt + selects `alpha` after each graded game and reports whether hybrid Brier/PnL improves as n grows; writes a walkthrough CSV under `reports/thesis_summaries/`.
- `update_moneypuck_injuries_nhl.py` — Fetch MoneyPuck’s public `current_injuries.csv`, write canonical `chimera_v2c/data/moneypuck_injuries_nhl.{json,csv}`, and emit a slate-filtered LLM digest; writes diffs under `reports/alerts/moneypuck_injuries/` when changed.
- `watch_moneypuck_injuries_nhl.py` — Log-only poller that checks MoneyPuck injuries every N minutes and prints an alert + writes diffs on change (no auto-trigger).
- `analyze_home_rich_sweet_spots.py` — Read-only sweep over edge thresholds for “Kalshi home favorite, model says home is rich → fade home” and writes per-model curves + sweet-spot summary under `reports/thesis_summaries/`.
- `eval_positive_bucket_policy.py` — Read-only simulator: build per-model “positive p_yes buckets” from a training window and simulate a target slate trading only those buckets (compares v2c/grok/gemini/gpt/market_proxy/moneypuck vs Kalshi mid).

## Archived / Legacy
The following are moved to `archive/tools_legacy/` and should not be used for production:
- `analyze_recent_performance.py`
- `apply_llm_injuries.py`
- `backfill_daily_ledger_from_plan.py`
- `build_game_master_ledger.py`
- `fill_master_scores.py`
- `inspect_game_master_ledger.py`
- `organize_specialist_reports.py`
- `parse_specialist_batch.py`
- `process_raw_specialist_reports.py`
- `rebuild_specialist_ledger.py`
- `select_canonical_specialist_versions.py`
- `split_specialist_ledger_by_league.py`
- `prepare_nhl_ratings.py`

If a legacy tool is needed, copy it back intentionally and document it here with clear overwrite safeguards.
