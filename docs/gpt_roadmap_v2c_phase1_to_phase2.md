Last updated: 2025-12-13
Scope: Chimera v2c (Aeternus v2), Phase 1 pre-game → Phase 2 live (gated)

CHIMERA_v2c_OFFICIAL_ROADMAP (Phase 1 → Phase 2)

GOAL
Build an elite, streamlined, profitable pre-game (Phase 1) Kalshi engine for NBA/NHL/NFL, then extend to live trading (Phase 2) only after Phase 1 shows stable, fee-adjusted positive EV.

NON-NEGOTIABLES

* v2c is maker-first and NO-LADDERS. Any ladder references remain strictly legacy/deprecated.
* Append-only ledgers.
* Calibrated probabilities beat raw confidence.
* Micro-bankroll realism: prioritize capital turnover + fill quality over theoretical price perfection.

SUCCESS METRICS (TRACK DAILY)

* Fill rate per qualified signal.
* Capital utilization (how much bankroll is actually working).
* Realized edge vs mid at time of fill (fee-adjusted).
* Slippage/adverse selection rate.
* Calibration: Brier score + reliability buckets for p_true.
* Net EV per game day (after fees).
* Survival metrics: max drawdown, time-to-recovery.

NEXT 5 (EXECUTION ORDER — NHL-FIRST)

1. **Time-aligned external baselines (first-class columns)**
   * Persist baseline **values** into per‑day ledgers (append-only): `kalshi_mid`, sportsbook consensus (`market_proxy`, moneylines), and NHL baselines like `moneypuck`.
   * Persist baseline **provenance/timestamps** into `reports/market_snapshots/` so ledgers stay readable.
   * Goal: a clean, timestamped substrate for evaluation that doesn’t depend on “what we remember later”.

2. **Rebuild the evaluation scoreboard**
   * Daily / rolling Brier + reliability buckets + EV vs mid for each model/baseline.
   * Make it easy to answer: “Are we improving?” and “Which buckets actually beat Kalshi after costs?”

3. **Calibrate before optimizing doctrine**
   * Fit and version calibrators (per league) for v2c and any useful baselines; compare against raw.
   * Only then decide which predictors/buckets are eligible for trading.

4. **Automate injuries/news impact sizing (hybrid-by-default)**
   * Use ESPN digest automation + LLM sizing to produce per-team injury deltas (and log what changed).
   * Optional extension: LLM-assisted web search (Tavily) for late-breaking injury notes when needed.

5. **Doctrine + rulebook iteration on timestamped data**
   * Tune edge thresholds, confluence gates, and sizing using the timestamped ledger data (fees included).
   * Keep it simple: one side per game, maker-only, “no bet by default,” and only trade the buckets that prove positive.

PHASE 0 — DOCUMENTATION + SOURCE-OF-TRUTH HYGIENE

1. ~~Confirm “single source of truth” docs~~ (root README, chimera_v2c/README, doctrine overview aligned; ladders marked deprecated)

   * Root README: only the active v2c Phase 1 flow.
   * chimera_v2c/README: engine-specific guide.
   * DOCTRINE_AND_HISTORY (or replacement): concise v2c doctrine overview.
   * Explicit “Legacy/Archived” section that names ladders as deprecated.

2. ~~Normalize naming in code + paths~~ (logging now points to reports/execution_logs; legacy ladder names labeled historical)

3. ~~Add a one-command “v2c daily run” entrypoint~~ (run_pipeline.py chains refresh→fit→preflight→plan→log→backtest)

DELIVERABLES

* Clean docs.
* A “v2c-only” quickstart that cannot accidentally reference archived workflows.

PHASE 1A — EXECUTION ENGINE REBUILD FOR MICRO-BANKROLL
Objective: convert model edge into realized P&L with minimal capital lock-up.

1. ~~Remove/disable any residual ladder generation in v2c planning path~~ (plan-only, maker-only)

2. **(partial)** Active maker order management

   * Maker placement with reprice attempts and mid/spread audit logging added.
   * TTL/cancel loop still minimal; no auto-taker conversion.

3. **(partial)** Edge-tiered execution

   * Maker-only remains; edge-tiered taker not enabled (deferred).

4. Pre-game cutoff rules

   * Not implemented; defer to future.

5. Adverse selection guardrails

   * Basic reprice on mid snapshot; full adverse-selection re-eval deferred.

DELIVERABLES

* Execution audit fields now include anchor/placed price, mid/spread, attempt (placed/error/dry-run).

PHASE 1B — PROBABILITY CALIBRATION + UNCERTAINTY BANDS
Objective: stop overconfident bets and formalize “confidence as error bars.”

1. Build the calibration dataset from v2c ledger

   * Standardize fields needed for training:
     p_true_raw, market_implied, outcome, league, model_version.

2. Implement a calibration layer

   * Start simple:

     * logistic calibration or isotonic regression.
   * Output:
     p_true_calibrated.

3. Add uncertainty margin ε(p)

   * Define:
     q_min = p_true_calibrated − ε(p)
     q_max = p_true_calibrated + ε(p)
   * Use q_min for bet inclusion.

4. Integrate calibration into PTCS/ECS gating

   * Confidence should be tied to:
     sample size, model agreement, injury volatility.

DELIVERABLES

* Calibration script + stored calibrator artifact.
* Automated weekly refresh.

PHASE 1C — EDGE GATING THAT RESPECTS KALSHI FEES
Objective: only trade when the edge survives realistic cost + error.

1. ~~Formalize fee-aware inclusion criteria~~ (q_min vs market + fee_buffer applied in doctrine)

2. Edge bands by archetype

   * Not yet differentiated by archetype; current bands are global.

3. Add “no-bet by default” discipline

   * Present (missing data/guardrails → skip).

DELIVERABLES

* Doctrine-aligned inclusion rules encoded in planner (archetype bands deferred).

PHASE 1D — SPECIALIST/LLM AUTOMATION (NO MANUAL HANDOFFS)
Objective: preserve your multi-model edge without human bottlenecks.

1. Create a unified specialist schema

   * p_true, confidence/PTCS, key factors, injury assumptions, timestamp, model_id.

2. Automated ingestion pipeline

   * Raw LLM reports → structured JSON rows → append to master ledger.

3. Ensemble logic

   * Weighted blend of:
     Elo + Four Factors + calibrated specialist outputs + market sanity checks.
   * Track disagreement metrics:
     “model spread” becomes a risk input.

DELIVERABLES

* ingest_specialist_report_v2c.py (or equivalent) fully automated.
* Daily “ensemble health” summary.
* DONE (v8.4 grounding): `build_llm_packets.py` generates slate-level CSV packets (standings/form with L10/streaks, injuries with team_deltas, schedule/fatigue, optional odds, plus H2H) under `reports/llm_packets/<league>/<YYYYMMDD>/` so specialists can reason from structured tables instead of hallucinating records/injuries/odds.

PHASE 1E — BACKTESTS THAT MATCH REAL EXECUTION
Objective: stop optimizing on fantasy fills.

1. **(partial)** Execution-aware simulator

   * Simple fill-probability model added to NBA/NHL backtests (edge-distance heuristic).

2. A/B test strategies

   * Not implemented; maker-only simulated.

3. Report outputs

   * EV per trade with fill_prob now reported; capital turnover not yet modeled.

DELIVERABLES

* Backtest reports now include avg_fill_prob; full A/B and turnover deferred.

PHASE 1F — RISK TUNING FOR SURVIVAL + SIGNAL QUALITY
Objective: survive long enough for edge to compound.

1. Default to smaller fractions for micro-bankroll

   * Consider 1/8-Kelly baseline with caps.

2. Daily stop rules

   * max daily loss,
   * max open exposure,
   * max correlated exposure (same league/time window).

3. Dynamic de-risking on volatility spikes

   * Injury uncertainty or sudden market dislocations lower size automatically.

DELIVERABLES

* Risk config file + clear operator overrides.

PHASE 1G — OPERATOR UX + RELIABILITY
Objective: make the system frictionless and hard to misuse.

1. Add a “pre-flight” checklist command

   * Verifies data freshness, injuries updated, calibrator present, markets open.

2. CI + type checks around core v2c modules

   * Prevent regressions in planner/executor/ledger.

3. Container or reproducible env bootstrap

   * One-command install/run.

DELIVERABLES

* Stable daily operation with minimal manual poking.

PHASE 1 EXIT CRITERIA (MUST HIT BEFORE PHASE 2)

* Sustained positive fee-adjusted EV in live paper or micro-real runs.
* Demonstrated calibration improvement over baseline.
* Healthy fill rate without excessive adverse selection.
* Clear evidence that execution changes improved realized vs theoretical edge.

PHASE 2 — LIVE TRADING (ONLY AFTER PHASE 1 IS SOLID)
Objective: add in-game buy/sell to monetize information velocity.

1. Event-driven live architecture

   * Game state updates → live model update → risk gate → execution action.

2. Live p_true deltas

   * Simple first:

     * score/time/possession style win-prob updates per league.
   * Then enrich with:
     lineup changes, foul trouble, goalie pulls, etc.

3. Live kill-switch + thesis integrity

   * If live signals contradict pre-game thesis strongly:
     hedge/reduce/exit.

4. Latency-aware execution

   * Tighten thresholds further for live due to faster information decay.

DELIVERABLES

* live_watch module integrated with the same calibration + gating logic.

PHASE 3 — SCALE + “ELITE” POLISH

1. Multi-league abstraction cleanup

   * Unified interfaces for schedules, injuries, markets, results.

2. Model governance

   * Versioned artifacts, changelogs, automatic rollbacks if metrics degrade.

3. Expanded market selection (optional)

   * Only after core ML winner markets are robust.

NEAR-TERM OPERATOR PLAN (MAKE WHAT WE HAVE SHINY)
Applies to Phase 1 work (pre-game). Goal: reduce manual work, tighten doctrine around *measured* edge, and improve calibration without a massive new data pipeline.

1. Metrics heartbeat (weekly, read-only)

   * Standardize the weekly loop: EV vs Kalshi mid + Brier + reliability buckets by league/date window.
   * Use the existing analysis tools (`analyze_ev_vs_kalshi.py`, `build_model_reliability_by_p_bucket.py`, `rolling_calibration.py`) as the source of truth for whether changes helped.

2. Fee/spread-aware doctrine tuning (turn edge into realized PnL)

   * Convert the currently winning “fade rich favorite” doctrine into a cost-aware rule-set (fees + maker spread + adverse selection), with league- and bucket-specific thresholds.
   * Gate on model agreement counts and only trade buckets that stay positive under realistic costs.

3. Automate injury/news ingestion (replace manual web-search packets)

   * Prefer deterministic sources first (ESPN APIs / existing injury fetchers), then optionally layer in web search (e.g., Tavily) as a supplement.
   * Produce an auditable “injury/news packet” with timestamps + sources; have an LLM convert that into `injury_adjustments.json` deltas (append-safe) for the pipeline.
   * Keep this human-overridable (preflight should warn/stop if injuries are stale).

4. Hybridization as a bridge to “v2c-only” (analysis-first, then default)

   * Learn and evaluate simple hybrids across calibrated model columns (v2c/grok/gemini/gpt) using date-grouped CV.
   * Treat hybrid as an *offline* candidate until it beats baselines out-of-sample; only then consider wiring a `p_home_hybrid` into the daily flow.

5. Improve v2c calibration/uncertainty before adding heavy new data

   * Focus on calibration + uncertainty bands and the regimes where v2c is overconfident (favorites/high-p buckets).
   * Only after the above is stable: consider integrating richer player-level or advanced-stat feeds, and prove impact via the same weekly metrics loop.

FINAL ORDER OF OPERATIONS (THE SHORT PRIORITY LIST)

1. Purge/lock out ladder logic in v2c runtime paths.
2. **(partial)** Maker order manager: maker-only with reprice logging; TTL/cancel and taker conversion deferred.
3. ~~Execution-aware logging and metrics~~ (anchor/placed price, mid/spread, attempt logged).
4. Calibration + uncertainty bands (ε(q) still global; archetype bands deferred).
5. ~~Fee-aware edge gating tied to q_min~~ (doctrine fee_buffer applied).
6. Automated specialist ingestion + ensemble disagreement risk (deferred).
7. **(partial)** Execution-realistic backtesting (simple fill heuristic; A/B and turnover deferred).
8. Micro-bankroll risk tuning: per-game/daily caps present; correlated exposure caps deferred.
9. NEW (grounding):
   * DONE: Per-slate LLM data packets (standings_form + L10/streaks, injuries with team_deltas, schedule_fatigue, optional odds, H2H) via `build_llm_packets.py` for NBA/NHL/NFL.
   * TODO: Refine injury `team_delta` heuristics toward player-specific impact (e.g., star NRtg-based weights instead of uniform -5.0).
   * TODO: Add roster-aware QC to the injuries packet (drop or flag player/team mismatches by cross-checking against ESPN scoreboard/roster data rather than trusting upstream hallucinated mappings).
10. ~~One-command daily orchestrator + pre-flight checks~~ (run_pipeline with preflight).
11. CI/type safety + reproducible environment (deferred).
12. Phase 2 live architecture once Phase 1 metrics prove out.

FILE NAME SUGGESTION FOR YOUR REPO ROOT
gpt_roadmap_v2c_phase1_to_phase2.md
