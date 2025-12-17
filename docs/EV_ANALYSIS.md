EV vs Kalshi Analysis (Daily Ledgers)
=====================================

This document describes how to use the daily game ledgers to analyze:

- Realized EV vs the Kalshi mid for each model.
- Accuracy and Brier scores vs actual outcomes.
- EV by absolute edge buckets (|p_model − p_kalshi|) and over rolling windows.

All workflows here are **read-only** on the canonical daily ledgers.

Data Source
-----------
- Canonical ledgers: `reports/daily_ledgers/`
- File pattern: `YYYYMMDD_daily_game_ledger.csv`
- Expected columns (at minimum):
  - `date`, `league`, `matchup`, `actual_outcome`
  - Model columns: `v2c`, `gemini`, `grok`, `gpt`
  - Market column: `kalshi_mid` (workflow convention: captured at **T‑30 minutes** vs scheduled start; see `reports/daily_ledgers/README.md`)
  - Optional market baseline: `market_proxy` (Odds API, same T‑30 anchor when present)

The ledgers are append-only; do **not** rewrite or bulk-regenerate them. See
`reports/daily_ledgers/README.md` and `OPERATOR_INSTRUCTIONS.md` for full rules.

Core Concepts
-------------

**1. Realized EV vs Kalshi mid**

For each model and game, we compare the model’s home-win probability `p_model`
to the Kalshi mid `p_kalshi`:

- If `p_model > p_kalshi`: treat this as buying home at price `p_kalshi`.
- If `p_model < p_kalshi`: treat this as being effectively long away
  (short home) at price `p_kalshi`.
- One contract per game; PnL in probability/contract units:
  - Long home win:  `(1 − p_kalshi)`
  - Long home loss: `−p_kalshi`
  - Long away win:  `p_kalshi`
  - Long away loss: `−(1 − p_kalshi)`

Aggregating this over games gives:

- `bets`: number of traded games.
- `total_pnl`: total realized edge vs the Kalshi mid.
- `avg_pnl_per_bet`: average edge per game (our primary EV metric here).

**2. Brier score (calibration)**

For each model and game with a resolved (non-push) outcome `y ∈ {0,1}` and
probability `p`:

- Per-game Brier: `(p − y)²`
- Mean Brier: average over all such games.

Lower Brier means the model’s probabilities are, on average, closer to the
true win chances. Comparing Brier vs `kalshi_mid` tells us whether we are
more or less calibrated than the market.

**3. Edge buckets**

We often care about *where* we have edge, not just overall:

- Define `edge = p_model − p_kalshi`.
- Group games by absolute edge `|edge|` into buckets (e.g., 0–2.5%, 2.5–5%, …).
- Compute PnL metrics per bucket.

This shows, for each model, which edge ranges historically produce positive
EV vs the Kalshi mid and which should be avoided or down-weighted.

Tools
-----

### 1) `analyze_ev_vs_kalshi.py` (new)

Location: `chimera_v2c/tools/analyze_ev_vs_kalshi.py`

Usage examples (from repo root):

```bash
# Last 30 ledger days, all leagues, core models vs Kalshi mid
PYTHONPATH=. python chimera_v2c/tools/analyze_ev_vs_kalshi.py --days 30

# Explicit date range and league filter
PYTHONPATH=. python chimera_v2c/tools/analyze_ev_vs_kalshi.py \
    --start-date 2025-12-04 --end-date 2025-12-09 --league nhl
```

Key options:

- `--days N`  
  Include only the most recent N ledger days by filename date
  (default: 30). Use `--days 0` (or a negative value) to include all.
- `--start-date YYYY-MM-DD`, `--end-date YYYY-MM-DD`  
  Optional explicit date range. If provided, overrides `--days`.
- `--league nba|nhl|nfl`  
  Optional league filter; default is all leagues.
- `--models v2c gemini grok gpt`  
  Model columns to analyze. `kalshi_mid` is always used as the market baseline.
- `--bucket-width FLOAT`  
  Absolute edge bucket width for EV breakdown (default: 0.025).

Outputs (printed to stdout):

- **Overall EV vs Kalshi mid** per model:
  - `bets`, `total_pnl`, `avg_pnl_per_bet`.
- **Brier vs outcomes** for each model and `kalshi_mid`:
  - mean Brier and number of games.
- **Bucketed EV vs Kalshi mid** by `|p_model − p_kalshi|`:
  - For each model and bucket: `bets`, `total_pnl`, `avg_pnl_per_bet`.

This tool is read-only and safe with respect to daily ledgers.

### 1b) `analyze_scheme_d.py` (Scheme D backtests)

Location: `chimera_v2c/tools/analyze_scheme_d.py`

Purpose: recompute Scheme D’s per‑(league,model) I/J gates and backtest the
1/3/5 consensus sizing rule against the canonical daily ledgers.

Example (from repo root):

```bash
PYTHONPATH=. python chimera_v2c/tools/analyze_scheme_d.py \
    --start-date 2025-11-19 --end-date 2025-12-12
```

Key options:

- `--edge-threshold 0.05`  
  Defines “home rich” as `p_model − p_mid ≤ −edge_threshold` (default 5c).
- `--ev-threshold 0.10`  
  Gate threshold for allowing a (league,model,rule) bucket (default 10c).
- `--min-bets 1`  
  Minimum sample size for a bucket to be eligible for gating (default 1; raise
  this to reduce small‑sample noise).

Outputs:

- Prints I/J bucket EV per model/league and the consensus 1/3/5 backtest.
- Writes derived CSVs under `reports/ev_rulebooks/`:
  - `scheme_d_rule_stats.csv`
  - `scheme_d_backtest_consensus_135.csv`
  - `scheme_d_backtest_baseline_1u.csv`

### 1c) `analyze_rulebook_quadrants.py` (A/B/C/D + sub-buckets)

Location: `chimera_v2c/tools/analyze_rulebook_quadrants.py`

Purpose: scan symmetric edge regimes vs Kalshi mid, covering both:
- Market **home-favorite** and **away-favorite** days, and
- “fade” and “follow” directions, with optional sub-buckets based on whether
  the model itself crosses 0.5.

Bucket definitions (home-win probabilities; trade at `kalshi_mid`):
- `A` (home-fav + fade home): `p_mid >= 0.5` and `p_model <= p_mid - t` → buy **away**
  - `I`: `p_model < 0.5`
  - `J`: `p_model >= 0.5`
- `B` (home-fav + follow home): `p_mid >= 0.5` and `p_model >= p_mid + t` → buy **home**
  - `M`: `p_model >= 0.5`
  - `N`: `p_model < 0.5`
- `C` (away-fav + follow away): `p_mid < 0.5` and `p_model <= p_mid - t` → buy **away**
  - `O`: `p_model < 0.5`
  - `P`: `p_model >= 0.5`
- `D` (away-fav + fade away): `p_mid < 0.5` and `p_model >= p_mid + t` → buy **home**
  - `K`: `p_model >= 0.5`
  - `L`: `p_model < 0.5`

Example (NHL, matching the Scheme D window):

```bash
PYTHONPATH=. python chimera_v2c/tools/analyze_rulebook_quadrants.py \
  --start-date 2025-11-19 --end-date 2025-12-12 --league nhl \
  --models v2c gemini grok gpt market_proxy moneypuck \
  --edge-threshold 0.05
```

Writes: `reports/ev_rulebooks/quadrants_rule_stats.csv`

### 1d) `walkforward_grok_mid_hybrid_backtest.py` (walk-forward hybrid EV)

Location: `chimera_v2c/tools/walkforward_grok_mid_hybrid_backtest.py`

Purpose: walk-forward (train→test) evaluation for a Grok↔Kalshi-mid hybrid:
- Learns Grok Platt + shrinkage `alpha` on prior days (date-grouped CV),
- Learns per-bucket edge thresholds `t*` (A/B/C/D) on prior days, and
- Applies them to the next day to estimate out-of-sample PnL vs `kalshi_mid`.

Example:

```bash
PYTHONPATH=. python chimera_v2c/tools/walkforward_grok_mid_hybrid_backtest.py \
  --start-date 2025-11-30 --end-date 2025-12-14 --league nhl
```

Writes: `reports/thesis_summaries/walkforward_grok_mid_hybrid_<league>_<start>_<end>.csv`

Sweep mode (for per-model threshold calibration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To scan multiple edge thresholds and emit a "selected rulebook" (one threshold
per `(league, model, bucket)` based on allow-gates), use:

```bash
PYTHONPATH=. python chimera_v2c/tools/analyze_rulebook_quadrants.py \
  --start-date 2025-11-19 --end-date 2025-12-12 --league nhl \
  --models v2c gemini grok gpt market_proxy moneypuck \
  --preset-thresholds \
  --min-bets 10 --ev-threshold 0.10 \
  --select-threshold-mode min_edge \
  --write-selected-rulebook
```

Sweep outputs:
- `reports/ev_rulebooks/quadrants_rule_stats_sweep.csv` (full grid)
- `reports/ev_rulebooks/quadrants_rule_sweet_spots.csv` (selected thresholds)
- `reports/ev_rulebooks/quadrants_rulebook_selected.json` (machine-readable rulebook)

### 2) `fit_calibration_from_daily.py` (league-specific Platt scaling)

Location: `chimera_v2c/tools/fit_calibration_from_daily.py`

Purpose: fit Platt calibration parameters for **v2c** using the canonical
daily ledgers, per league.

Usage (NBA example, from repo root):

```bash
PYTHONPATH=. python chimera_v2c/tools/fit_calibration_from_daily.py \
    --league nba \
    --model-col v2c \
    --out chimera_v2c/data/calibration_params_nba.json
```

Details:

- Reads: `reports/daily_ledgers/*_daily_game_ledger.csv`
- Uses rows where:
  - `league` matches the `--league` argument, and
  - the requested `--model-col` is present and numeric, and
  - `actual_outcome` encodes a resolved home win/loss.
- Builds `(p_v2c, y_true)` pairs where `y_true` is 1 for home win and
  0 for home loss.
- Fits a Platt scaler (`fit_platt`) when there are at least
  `--min-samples` pairs (default 30); otherwise writes an identity
  calibration `{a: 1.0, b: 0.0}`.
- Writes JSON to the path given by `--out`, typically
  `chimera_v2c/data/calibration_params_<league>.json`.

In the NBA pipeline, `chimera_v2c/config/defaults.yaml` points the
`calibration.path` at `chimera_v2c/data/calibration_params_nba.json`
so that `build_daily_plan` applies this calibration to `p_final`
before risk sizing.

Important note on calibration
-----------------------------
- Daily ledgers intentionally keep a minimal schema and do not store a separate `v2c_raw` column.
- If you want to fit a pipeline calibrator, calibrate the probability source you actually plan to
  trade/evaluate (typically the daily-ledger `v2c`), and avoid introducing extra columns into the
  canonical daily ledgers just for analysis.

### 3) `rolling_calibration.py`

Location: `chimera_v2c/tools/rolling_calibration.py`

Purpose: rolling accuracy/Brier summary from daily ledgers over the last N
days. It does not compute EV vs the Kalshi mid, but is useful for tracking
calibration drift over time.

Example:

```bash
PYTHONPATH=. python chimera_v2c/tools/rolling_calibration.py --days 14 --league nba
```

### 4) `build_model_wr_by_league.py`

Location: `chimera_v2c/tools/build_model_wr_by_league.py`

Purpose: compute per-league and overall accuracy/Brier summaries for
`v2c`, `gemini`, `grok`, `gpt`, `kalshi_mid`, and `market_proxy` from all daily
ledgers and write `reports/daily_ledgers/model_wr_by_league.csv`.

Example:

```bash
PYTHONPATH=. python chimera_v2c/tools/build_model_wr_by_league.py
```

### 5) `build_ev_summary_by_league.py` (EV + Brier “EV file”)

Location: `chimera_v2c/tools/build_ev_summary_by_league.py`

Purpose: write a single CSV “EV file” that combines:

- Realized EV vs Kalshi mid (1 unit per game) for each model.
- Brier score vs outcomes for each model and for `kalshi_mid`.

Output: `reports/daily_ledgers/ev_brier_by_league.csv`

Example:

```bash
PYTHONPATH=. python chimera_v2c/tools/build_ev_summary_by_league.py \
    --start-date 2025-11-19 --end-date 2025-12-12
```

How to Use This to Improve EV
-----------------------------

1. **Identify where each model has edge vs Kalshi**
   - Run `analyze_ev_vs_kalshi.py` over a rolling window (e.g., last 30–60 days).
   - Look at `avg_pnl_per_bet` by absolute edge bucket:
     - Buckets with consistently positive EV are candidates for trading.
     - Buckets with flat or negative EV should be avoided or down-weighted.

2. **Tune entry thresholds and gating**
   - Use the bucketed EV tables to set league- and model-specific edge
     thresholds:
     - Example: “In NHL, only trade v2c where `|p_v2c − p_kalshi| ≥ 0.06` and
       historical bucketed EV is positive.”
   - Combine this with existing doctrine/guardrails in `chimera_v2c/config/*.yaml`.

3. **Improve calibration**
   - Use Brier outputs from `analyze_ev_vs_kalshi.py`, `rolling_calibration.py`,
     and `build_model_wr_by_league.py` to check:
     - Are we more or less calibrated than `kalshi_mid` overall?
     - Are specific leagues or regimes poorly calibrated?
   - Re-fit calibration maps (e.g., via `fit_calibration.py`) and feed calibrated
     probabilities into staking rules instead of raw probabilities.

4. **Consider meta-probabilities (future work)**
   - Treat each ledger row as a supervised example with features:
     `p_v2c`, `p_gemini`, `p_grok`, `p_gpt`, `p_kalshi`, league, etc.
   - Fit an offline meta-model to predict a combined probability `p_meta`.
   - Backtest `p_meta` vs Kalshi using the same EV logic as
     `analyze_ev_vs_kalshi.py` before wiring into live configs.

LLM / Agent Notes
-----------------
- Before using these tools, read:
  - `reports/daily_ledgers/README.md`
  - `OPERATOR_INSTRUCTIONS.md`
- Do **not** write new scripts that modify or bulk-regenerate daily ledgers.
  All analysis should be read-only and operate on the existing CSV files.

Performance Tracking Playbook
-----------------------------

This is the standard way to track v2c performance over time and decide
calibration/gating changes:

- **Daily artifacts**
  - Per-day ledgers: `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv`
    (canonical probabilities, market mids, outcomes).
  - Plan/execution logs: `reports/execution_logs/` (what v2c wanted to do vs what
    actually executed).

- **Core metrics**
  - Realized EV vs Kalshi mid:
    - `avg_pnl_per_bet` per model/league/date window from `analyze_ev_vs_kalshi.py`.
  - Calibration:
    - Brier vs actual outcomes for each model and `kalshi_mid` (overall and by
      league) from `analyze_ev_vs_kalshi.py`, `rolling_calibration.py`, and
      `build_model_wr_by_league.py`.
  - Execution-aware ROI (for leagues with backtests):
    - `nba_backtest.py` / `nhl_backtest.py` for bucketed ROI with simple fill
      heuristics.

- **Recommended loop**
  - On a regular cadence (e.g., weekly):
    1. Run `analyze_ev_vs_kalshi.py` for each active league (e.g., last 30–60
       days) to see which models have positive realized EV vs Kalshi and in
       which edge buckets.
    2. Run `rolling_calibration.py` and/or `build_model_wr_by_league.py` to
       track calibration vs `kalshi_mid` and truth.
    3. For NBA/NHL, run the league backtests to cross-check ROI by probability
       bucket and fill assumptions.
    4. Use these outputs to:
       - Tighten or relax edge thresholds and doctrine guardrails in
         `chimera_v2c/config/*.yaml`.
       - Decide where v2c needs calibration updates (via
         `fit_calibration_from_daily.py`) or different weighting vs sharp odds
         or Kalshi.

Grok‑Centric Edge Tiers (Scheme A)
----------------------------------

This subsection documents an exploratory **tiered edge system** built on the
daily ledgers for the specialist models (Grok, Gemini, GPT) and v2c, using
Kalshi mids as the reference. It is **analysis‑only** and not wired into the
live v2c doctrine; think of it as a structured way to reason about how
multiple models agree or disagree with the market.

Setup and Notation
------------------

For each graded game in the daily ledgers:

- `p_mid`  = `kalshi_mid` (home win probability implied by the mid).
- `p_grok` = Grok home‑win probability.
- `p_v2c`, `p_gemini`, `p_gpt` = v2c / Gemini / GPT home‑win probabilities.
- `edge_g = p_grok − p_mid` (Grok’s edge vs the Kalshi mid).
- A model’s **favorite** is:
  - Home if `p_model ≥ 0.5`, else away.
- The **market favorite** is:
  - Home if `p_mid ≥ 0.5`, else away.

All tiers below use the same baseline **Grok edge gate**:

- Require `|edge_g| ≥ 0.05` (Grok differs from the mid by at least 5 points).
- Direction for PnL:
  - If `edge_g > 0`: treat as a long **home YES** at price `p_mid`.
  - If `edge_g < 0`: treat as a long **away YES** (short home) at price `p_mid`.

We then look at how other models line up relative to Grok and the market.

Alignment (Scheme A)
--------------------

In **Scheme A**, Grok is the only model that must clear the ±5% edge gate.
Other models are used as **sign‑only filters**:

- A model `m ∈ {v2c, gemini, gpt}` is considered **aligned** with Grok on a
  given game if:
  - `p_m` is present and numeric, and
  - `(p_m − p_mid)` has the **same sign** as `edge_g`
    (i.e., both lean home over the mid or both lean away).
- We do **not** require `|p_m − p_mid| ≥ 0.05` in Scheme A; only direction
  matters for alignment.

We also define:

- **Grok opposite favorite:** Grok’s favorite side (home/away) differs from
  the market favorite.
- **Combo favorite:** For a set of models (e.g., Grok + aligned models), the
  combo favorite is home if the **mean** of their `p_home` is ≥ 0.5, else away.
- **Combo opposite favorite:** combo favorite differs from market favorite.

Tier Definitions (Scheme A)
---------------------------

Given the above, we define 8 conceptual tiers:

1. **Tier 1 – Grok ±5% edge (baseline)**
   - Condition: `|edge_g| ≥ 0.05`.
   - Direction: sign of `edge_g` (YES home if `edge_g > 0`, NO home if `< 0`).

2. **Tier 2 – Grok ±5% + any 1 aligned model**
   - Start from Tier 1.
   - At least **one** of `{v2c, gemini, gpt}` is aligned with Grok
     (same sign vs `p_mid`).

3. **Tier 3 – Grok ±5% + any 2 aligned models**
   - Start from Tier 1.
   - At least **two** of `{v2c, gemini, gpt}` are aligned with Grok.

4. **Tier 4 – Grok ±5% + Grok fades market favorite**
   - Start from Tier 1.
   - Grok’s favorite side (home/away) is **different** from the market
     favorite (i.e., Grok is backing the underdog vs Kalshi).

5. **Tier 5 – Tier 2 + combo fades market favorite**
   - Start from Tier 2 (Grok ±5% + ≥1 aligned model).
   - Define `p_combo` as the mean of `p_grok` and the aligned models’ `p_home`.
   - Combo favorite (from `p_combo`) is **different** from the market favorite.

6. **Tier 6 – Tier 3 + combo fades market favorite**
   - Start from Tier 3 (Grok ±5% + ≥2 aligned models).
   - Combo favorite is different from the market favorite.

7. **Tier 7 – All four models agree with Grok vs Kalshi (hypothetical)**
   - Start from Tier 1.
   - All four models `{v2c, gemini, grok, gpt}` have `(p_model − p_mid)`
     with the **same sign**.
   - This is a **reserved** tier: in current data, this configuration has not
     yet occurred under the Grok ±5% gate.

8. **Tier 8 – Tier 7 + all four fade market favorite (hypothetical)**
   - Start from Tier 7.
   - Let `p_all` be the mean of `{p_v2c, p_gemini, p_grok, p_gpt}`.
   - The all‑model favorite (from `p_all`) differs from the market favorite.
   - Also reserved; we use it as a conceptual “maximum confluence” tier.

Empirical Snapshot (All Leagues, All Days)
------------------------------------------

Using Scheme A and the daily ledgers **as of 2025‑12‑10**, the aggregate EV
per tier looks like:

| Tier | Description                                | bets | avg_pnl_per_bet |
|------|--------------------------------------------|------|------------------|
| 1    | Grok ±5% vs mid                            | 47   | ≈ 0.157          |
| 2    | Tier 1 + ≥1 aligned model                  | 38   | ≈ 0.182          |
| 3    | Tier 1 + ≥2 aligned models                 | 16   | ≈ 0.241          |
| 4    | Tier 1 + Grok opposite market favorite     | 17   | ≈ 0.326          |
| 5    | Tier 2 + combo opposite market favorite    | 15   | ≈ 0.303          |
| 6    | Tier 3 + combo opposite market favorite    | 7    | ≈ 0.403          |
| 7    | All four models same sign vs mid (Tier 1)  | 0    | n/a              |
| 8    | Tier 7 + all‑four opposite market favorite | 0    | n/a              |

Important caveats:

- These EV numbers are **purely historical** and based on a modest number of
  bets per tier; they should be treated as suggestive, not as guarantees.
- Tiers 7 and 8 have never fired in the current sample and are kept as
  “shadow” tiers for future monitoring only.
- The analysis assumes fills at the mid with no fees or spread.

How to Use the Tier System
--------------------------

For now, this tier system is intended as a **diagnostic and planning tool**:

- Tier 1 defines a **minimum edge gate** (Grok must differ from `p_mid` by at
  least 5 points).
- Higher tiers represent increasingly strong conditions:
  - More models aligned with Grok vs Kalshi.
  - Grok and/or the combo explicitly fading Kalshi’s favorite.
- In historical data, higher tiers systematically show **higher average EV per
  bet** but fewer total bets, which is what we would expect if confluence and
  underdog fades are meaningful signals.

Operationally, you can:

- Use Tier 1 as the baseline “is there any edge here?” filter when studying
  Grok vs Kalshi.
- Treat Tiers 2–6 as **confidence buckets**:
  - e.g., 1× stake for Tier 1, more weight for Tier 3–6 in hypothetical
    simulations.
- Log and highlight any future Tier 7/8 events as **extreme confluence**
  scenarios and review them manually before acting.

Nothing in this section changes the live v2c doctrine; it is a structured way
to interpret the joint behavior of Grok, Gemini, GPT, v2c, and the Kalshi mid
within the existing EV analysis framework.

Alternative Schemes (Experimental)
----------------------------------

The Grok-anchored tiers above (Scheme A) treat positive and negative edges
symmetrically: any `|edge_g| ≥ 0.05` (home or away) qualifies as a candidate
signal, and other models are used as alignment filters.

Empirically, early data suggests a *directional* asymmetry:

- Historical EV is much stronger when **Kalshi appears rich on home** and we
  fade home (bet away) than when Kalshi appears rich on away and we buy home.
- Multi‑model agreement on that “home‑rich → fade home” condition tends to
  increase realized EV per bet.

The following two schemes are **experimental** analysis views built on top of
the same daily ledgers. They do not change live doctrine; we are still in a
testing phase and collecting more data before preferring any single scheme.

### Scheme B – Multi‑Model Home‑Rich Fade (Any Model)

Gate and direction:

- For each model `m ∈ {v2c, gemini, grok, gpt}` define
  `edge_m = p_home_m − p_mid`.
- A game is in the **home‑rich** set if **any** model satisfies
  `edge_m ≤ −0.05` (Kalshi home too high by ≥ 5 points).
- Direction for PnL: always treat this as a **long away** (short home) at
  price `p_mid`, regardless of which model(s) triggered it.

This scheme is **model‑agnostic**: each model–game pair with `edge_m ≤ −0.05`
is counted as a separate 1‑unit bet.

Current ledger snapshot (all leagues, all days in the master ledger):

- v2c home‑rich fades: `avg_pnl_per_bet ≈ 0.10`
- Gemini home‑rich fades: `avg_pnl_per_bet ≈ 0.12`
- Grok home‑rich fades: `avg_pnl_per_bet ≈ 0.19`
- GPT home‑rich fades: `avg_pnl_per_bet ≈ 0.11`

Aggregated across all models, home‑rich fades have mean EV comparable to the
Scheme A Tier‑1 baseline, but they concentrate EV on the **negative‑edge
(home‑rich) direction** rather than using ±edge symmetrically.

### Scheme C – Grok‑Negative, Multi‑Model Fade (Directional Tiering)

Scheme C combines the Grok‑anchored gate from Scheme A with the directional
asymmetry from Scheme B.

Gate:

- Grok must have a **negative** edge of at least 5 points:
  - `edge_g = p_grok − p_mid ≤ −0.05` (Kalshi rich on home, away cheap).
- Alignment: treat `{v2c, gemini, gpt}` as aligned when they share the same
  sign vs the mid:
  - `edge_m = p_home_m − p_mid < 0` (also prefer away at this mid).

Tiering (Scheme C):

- **C1 – Grok‑negative baseline**
  - Condition: `edge_g ≤ −0.05`; direction: bet away at `p_mid`.
- **C2 – Grok‑negative + any 1 aligned model**
  - C1, plus at least **one** of `{v2c, gemini, gpt}` has `edge_m < 0`.
- **C3 – Grok‑negative + any 2 aligned models** (**Schema 3**)  
  - C1, plus at least **two** of `{v2c, gemini, gpt}` have `edge_m < 0`.

(You can optionally define a C4 “combo opposite favorite” layer analogous to
Scheme A’s Tiers 5–6 by checking whether the mean of `p_home` over Grok and
aligned models has a favorite opposite the market favorite.)

Current ledger snapshot (all leagues, current master ledger, following Grok’s
direction and betting away):

- C1 (Grok‑negative, ±alignment): directional subset of Scheme A Tier 1.
- **C3 (Schema 3: Grok‑negative + ≥2 aligned models)**:
  - `bets ≈ 18`
  - `avg_pnl_per_bet ≈ 0.27`

These values are **preliminary** and based on the current ledger only. They
should be treated as suggestive rather than definitive. In particular,
Schemes B and C are still in testing; we have not yet selected a single
canonical trading scheme, and future data may favor one of these or a hybrid,
or the original symmetric Scheme A, as more data accumulates.

Scheme D – I/J‑Gated Consensus Home‑Favorite Fades (Current Default)
--------------------------------------------------------------------

As more data accumulated across NBA/NFL/NHL daily ledgers, a clearer signal
emerged specifically in **home‑favorite mispricing** regimes:

- The strongest, most consistent edge vs Kalshi is when:
  - Kalshi favors **home** (`p_mid ≥ 0.5`), and
  - One or more specialist models view home as **rich by ≥ 5 points**.
- Within that, **multi‑model agreement** (v2c, Gemini, Grok, and in some
  NHL pockets GPT) is a much better predictor of realized EV than any single
  model in isolation.

Scheme D formalizes a **directional, data‑gated home‑favorite fade** and is
the current preferred analysis schema. It is still experimental and should be
re‑evaluated as the ledgers grow.

Definitions
~~~~~~~~~~~

For each graded game:

- `p_mid` = Kalshi **home** probability (`kalshi_mid`).
- `p_m`   = model `m`’s home probability (for `m ∈ {v2c, gemini, grok, gpt}`).
- We restrict to **home‑favorite** markets: `p_mid ≥ 0.5`.
- Define:
  - **Home‑rich** vs model: `edge_m = p_m − p_mid ≤ −0.05`.
  - **Rule I (hard flip):**  
    - `p_mid ≥ 0.5`,  
    - `edge_m ≤ −0.05`, and  
    - `p_m < 0.5` (model actually prefers away).
  - **Rule J (soft disagreement):**  
    - `p_mid ≥ 0.5`,  
    - `edge_m ≤ −0.05`, and  
    - `p_m ≥ 0.5` and `p_m ≤ p_mid − 0.05` (model still prefers home, but is ≥5 points less bullish than the market).

Every I or J event is a **Rule‑A home‑rich fade** (`p_mid ≥ 0.5`, `edge_m ≤ −0.05`), but
we treat I/J separately because their realized EV varies by model and league.

Per‑Model, Per‑League I/J Gating
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Empirically, not every (model, league, rule) bucket is equally strong. For
each of v2c, Gemini, Grok, and GPT, we:

1. Compute realized EV vs Kalshi for **Rule I** and **Rule J** separately,
   per league (NBA/NFL/NHL), using the canonical daily ledgers.
2. Keep only **“strong” buckets** where the historical average
   `avg_pnl_per_bet ≥ +0.10` (10 cents of realized edge per contract).
3. Drop weak or negative buckets (e.g., Gemini Rule I in NBA, Grok Rule J in
   NHL, GPT Rule J in all leagues).

As of the 2025‑12‑10 ledgers, this yields:

- Rule I (hard flips) allowed for:
  - NBA: Grok
  - NFL: v2c, Grok
  - NHL: Gemini, Grok, GPT
- Rule J (soft disagreement) allowed for:
  - NBA: v2c, Gemini, Grok
  - NFL: v2c, Gemini, Grok
  - NHL: v2c only

These sets are **data‑driven** and should be recomputed as more games are
added. The intent is to only size into I/J buckets that have earned their
keep historically, while continuing to track weaker buckets for future
re‑evaluation.

Consensus‑Count Sizing (1/3/5 Units)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the above gating, Scheme D ignores the difference between I and J for
position sizing and focuses instead on **how many strong models agree** that
the home favorite is rich:

1. For a given game, count:
   - `n` = number of models in `{v2c, gemini, grok, gpt}` that:
     - Are in an allowed (model, league, I/J) bucket, and
     - Fire Rule I or Rule J (home rich by ≥5 points) for that game.
2. Always bet **AWAY** (fade the home favorite) in these cases.
3. Map `n` to **units** as:

   - `n = 0` → **0 units** (no bet).
   - `n = 1` → **1 unit** (weakest tier; single strong model).
   - `n = 2` → **3 units** (strong; two strong models agree).
   - `n ≥ 3` → **5 units** (very strong; three or more strong models agree).

Here, “unit” is a stake scalar (e.g., 1 contract or 1× bankroll fraction).
All units are sized at the Kalshi mid; fees and spread are omitted in this
analysis.

Historical Snapshot (as of 2025‑12‑10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the canonical daily ledgers through 2025‑12‑10, with the per‑model
I/J ≥ 10‑cent gating above:

- **Baseline (1 unit per strong I/J signal)**  
  - Total units: ~58  
  - Total PnL: ≈ +20.1  
  - `avg_pnl_per_unit ≈ +0.346`

- **Scheme D 1/3/5 consensus sizing**  
  - Total units: ~77  
  - Total PnL: ≈ +28.5  
  - `avg_pnl_per_unit ≈ +0.370`

- **By tier (approximate)**  
  - 1‑unit tier (`n = 1` strong model):  
    - ≈ 25 units, `avg_pnl_per_unit ≈ +0.22`
  - 3‑unit tier (`n = 2` strong models):  
    - ≈ 27 units, `avg_pnl_per_unit ≈ +0.43`
  - 5‑unit tier (`n ≥ 3` strong models):  
    - ≈ 25 units, `avg_pnl_per_unit ≈ +0.46`

This produces the desired **monotone pattern** by tier: higher consensus
generally corresponds to higher realized EV per unit, and the overall
avg_pnl_per_unit improves relative to a flat 1‑unit‑per‑model scheme using
the same gated signals.

Status and Usage
~~~~~~~~~~~~~~~~

Scheme D is currently the **preferred experimental schema** for analyzing
home‑favorite mispricing fades across models:

- It respects per‑model, per‑league performance by gating I/J buckets on
  historical EV.
- It uses a simple, interpretable sizing rule based on the number of strong
  models in agreement.
- It deliberately ignores away‑favorite cases and non‑home‑rich scenarios,
  which have weaker or inconsistent signals in the current data.

This schema is **not yet wired into live v2c risk sizing**; it is intended
for backtesting, dashboarding, and operator review. As the ledgers grow, the
gates (e.g., 10‑cent threshold) and tier mapping (1/3/5 units, or more
granular tiers) should be revisited and tuned. If a future schema clearly
dominates Scheme D, update this section and the rulebook CSVs under
`reports/ev_rulebooks/` to keep the documentation and analysis in sync.
