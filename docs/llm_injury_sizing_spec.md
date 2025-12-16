# LLM Injury/News Impact Sizing — v2c Step 4 Design Spec (Doc Only)

This memo specifies a **safe, auditable, reproducible** way to size and apply LLM-derived injury/news impacts (per team, per slate date) for Chimera v2c.

**Scope**
- Primary target: **NHL-first**, but schema is league-agnostic (`nba|nhl|nfl`).
- This spec is **doc-only** (no code changes here). It is written to align with:
  - `chimera_v2c/tools/refresh_slate_updates.py` (ESPN slate digest + stubs)
  - `chimera_v2c/tools/apply_llm_nhl_injuries.py`, `chimera_v2c/tools/apply_llm_injuries_v2.py` (current LLM merge)
  - `chimera_v2c/src/probability.py` (injury deltas applied to Elo component)
  - `docs/EV_ANALYSIS.md` (Brier + EV vs Kalshi mid)
  - `reports/daily_ledgers/README.md` + `OPERATOR_INSTRUCTIONS.md` (append-only + provenance rules)

---

## Current State (What Exists Today)

- `refresh_slate_updates.py`:
  - Pulls ESPN injuries + team-filtered news for the slate.
  - Writes a digest: `chimera_v2c/data/news_<date>_<league>.txt` and `reports/llm_packets/<league>/<YYYYMMDD>/news.txt`.
  - Ensures `chimera_v2c/data/injury_adjustments.json` has **0.0 stubs** for all slate teams **without overwriting existing values**.
- LLM appliers:
  - `apply_llm_injuries_v2.py` and `apply_llm_nhl_injuries.py` ask the LLM for per-team Elo-like deltas and merge into `injury_adjustments.json`.
  - They **overwrite team/date values** on merge (not “fill blanks only”).
  - They do not carry confidence/ptcs/evidence, and audit is just the raw response JSON.
- Engine behavior:
  - `ProbabilityEngine._p_elo()` applies injury deltas as rating offsets:
    - `home_rating += home_bonus + inj_delta(home)`
    - `away_rating += inj_delta(away)`
  - So injury deltas affect only the Elo component unless/until explicitly expanded.
- Safety rules:
  - Ledgers are append-only; provenance belongs in `reports/market_snapshots/`, not in ledgers.

This spec upgrades the injury sizing workflow to be **versioned, explainable, and merge-safe** while preserving the existing runtime contract (`injury_adjustments.json` is still the simple float map used by `InjuryAdjustmentConfig`).

---

## 1) Canonical LLM Output: Strict JSON Schema (Per-Team Deltas + Confidence/PTCS + Evidence)

### Design Principles

- **Strict, machine-validated** schema: reject malformed output early.
- **One packet = one league + one date**, with team-level entries.
- **Two separate concepts**:
  - `confidence` = LLM’s subjective confidence in the *magnitude/direction* of the delta.
  - `ptcs` = *data-quality score* (PTCS rubric) used for **shrinkage/gating**, not “vibes”.
- **Evidence lines** are optional but strongly recommended: short quotes/snippets from the ESPN digest that justify the delta.
- Do **not** store these rich fields inside `injury_adjustments.json` yet; store them in a timestamped snapshot file (see Section 2/3).

### JSON Schema (Draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://chimera.local/schemas/llm_injury_sizing_packet_v1.json",
  "title": "LLM Injury Sizing Packet v1",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "schema_version",
    "run_id",
    "league",
    "date",
    "generated_at",
    "model",
    "source",
    "teams"
  ],
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "llm_injury_sizing_v1"
    },
    "run_id": {
      "type": "string",
      "description": "Unique run identifier for traceability (UUID recommended).",
      "pattern": "^[0-9a-fA-F-]{16,64}$"
    },
    "league": {
      "type": "string",
      "enum": ["NHL", "NBA", "NFL"]
    },
    "date": {
      "type": "string",
      "format": "date"
    },
    "generated_at": {
      "type": "string",
      "description": "UTC ISO timestamp, Z-normalized (e.g., 2025-12-13T19:30:00Z).",
      "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$"
    },
    "model": {
      "type": "object",
      "additionalProperties": false,
      "required": ["provider", "name"],
      "properties": {
        "provider": { "type": "string", "enum": ["openai"] },
        "name": { "type": "string", "minLength": 1 },
        "temperature": { "type": "number", "minimum": 0, "maximum": 2 }
      }
    },
    "source": {
      "type": "object",
      "additionalProperties": false,
      "required": [
        "input_digest_path",
        "input_digest_sha256",
        "input_digest_generated_at"
      ],
      "properties": {
        "input_digest_path": { "type": "string", "minLength": 1 },
        "input_digest_sha256": {
          "type": "string",
          "pattern": "^[0-9a-f]{64}$"
        },
        "input_digest_generated_at": {
          "type": "string",
          "pattern": "^\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}Z$"
        },
        "notes": { "type": "string", "maxLength": 2000 }
      }
    },
    "teams": {
      "type": "object",
      "description": "Per-team deltas and quality scores.",
      "minProperties": 1,
      "additionalProperties": false,
      "patternProperties": {
        "^[A-Z]{2,4}$": {
          "type": "object",
          "additionalProperties": false,
          "required": ["delta_elo", "confidence", "ptcs"],
          "properties": {
            "delta_elo": {
              "type": "number",
              "description": "Elo-like delta applied to team rating (negative = penalty).",
              "minimum": -40.0,
              "maximum": 0.0
            },
            "confidence": {
              "type": "number",
              "description": "LLM subjective confidence in delta magnitude/direction.",
              "minimum": 0.0,
              "maximum": 1.0
            },
            "ptcs": {
              "type": "number",
              "description": "PTCS data-quality score used for shrinkage/gating (not 'vibes').",
              "minimum": 0.0,
              "maximum": 1.0
            },
            "evidence_lines": {
              "type": "array",
              "description": "Optional short quotes/snippets from the digest supporting the delta.",
              "maxItems": 12,
              "items": { "type": "string", "minLength": 1, "maxLength": 280 }
            },
            "flags": {
              "type": "array",
              "description": "Optional machine tags (e.g., GOALIE_UNCONFIRMED, MULTIPLE_Q, LOW_SOURCE_TRUST).",
              "maxItems": 10,
              "items": {
                "type": "string",
                "pattern": "^[A-Z0-9_]{2,40}$"
              }
            }
          }
        }
      }
    }
  }
}
```

### PTCS Guidance (Injury Sizing Context)

PTCS should be treated as **data-quality** and should drive **shrinkage** and **edge thresholds** (consistent with the PTCS rubric used in the NHL directive).

Recommended simplified injury PTCS rubric (per team, deterministic-friendly):
- Start at `ptcs = 1.0`.
- Subtract quality penalties (floored at 0.0):
  - `-0.50` if starting goalie status is unknown/conflicting for that team (NHL).
  - `-0.25` if one or more key players are **questionable** / game-time decision.
  - `-0.25` if the evidence is stale (digest older than X hours) or lacks clear status tags.
  - `-0.25` if evidence lines are missing for a non-zero delta.

Even if PTCS is estimated by the LLM initially, the pipeline should be able to recompute/override it deterministically later.

---

## 2) Guardrails for Safe Merge into `injury_adjustments.json`

### Goals

- Preserve the safety property already present in `refresh_slate_updates.py`:
  - **Never overwrite an existing non-zero delta by default.**
- Make every change auditable:
  - store raw packet, merge decision log, timestamps, and checksums.

### Merge Rules (Required)

1. **Clamp deltas** at merge time:
   - `delta_elo = clamp(delta_elo, -40.0, 0.0)`
2. **Default behavior fills blanks only**:
   - If existing `(league,date,team)` is missing or `0.0`, allow filling from LLM.
   - If existing is non-zero, **skip** unless an explicit override flag is passed.
3. **No silent overwrites**:
   - Overwriting non-zero values requires an explicit operator flag, e.g.:
     - `--overwrite-nonzero` (or `--force`).
4. **Diff + audit must be written even on dry-run**:
   - For each team in packet: record `old`, `new`, `action={filled|skipped_nonzero|clamped|ignored_invalid}`, and why.
5. **Timestamping and snapshots**:
   - Before writing `injury_adjustments.json`, create a timestamped backup + sha256 (similar to `snapshot_file()` behavior used for ledgers).
   - Write a timestamped injury snapshot artifact (see below).

### Snapshot Artifacts (Where the Rich Data Lives)

Store the LLM packet and merge audit under `reports/market_snapshots/` so that:
- daily ledgers remain readable,
- we can reconstruct “what did we know at time T?” for evaluation/backtests.

Recommended file patterns:
- Raw packet (exact LLM output after schema validation):
  - `reports/market_snapshots/YYYYMMDD_<league>_injury_llm_packet_<HHMMSSZ>.json`
- Merge audit (includes before/after diff and merge stats):
  - `reports/market_snapshots/YYYYMMDD_<league>_injury_llm_merge_<HHMMSSZ>.json`

Minimum contents for the merge audit:
- `snapshot_ts` (UTC)
- `run_id`, `league`, `date`, `model.name`
- `input_digest_path`, `input_digest_sha256`
- `changes`: list of `{team, old_delta, new_delta, action, reason, confidence, ptcs}`
- `summary`: `{teams_seen, filled, skipped_nonzero, clamped, ignored_invalid}`

### Optional but Recommended Guardrail

If the daily ledger date is locked (`reports/daily_ledgers/locked/YYYYMMDD.lock`), refuse to write new injury deltas for that date unless `--force`. This prevents retroactively changing what “the model knew” for closed days (critical for clean eval).

---

## 3) Wiring + Logging (run_daily/run_pipeline + plan logs + market snapshots)

### Where to Wire

Current wiring already exists:
- `run_daily.py --llm-injuries` runs `refresh_slate_updates` then calls the LLM applier.
- `run_pipeline.py --llm-injuries` does the same per league.
- `scheduled_injury_plan.py --llm-injuries` does the same near tip.

Spec changes (behavioral intent):
- Keep the flags, but change the underlying injury tool to:
  1) produce a **versioned packet** (schema above),
  2) write **timestamped snapshots** under `reports/market_snapshots/`,
  3) perform a **safe merge** into `chimera_v2c/data/injury_adjustments.json`.

Recommended additional flags (for the future implementation):
- `--llm-injuries` (existing): generate packet + attempt safe merge (fill blanks only).
- `--llm-overwrite-nonzero` (new): allow overwriting existing non-zero deltas.
- `--llm-dry-run` (new): write packet + merge audit, but do not modify `injury_adjustments.json`.
- `--llm-ptcs-min <0..1>` (new): if team `ptcs` below threshold, auto-zero the delta or force shrinkage.

### What to Log to Plans

Today, `build_daily_plan()` already logs in `GamePlan.diagnostics`:
- `injury_mtime`, `injury_home_delta`, `injury_away_delta`.

Spec additions (recommended) so plans are self-auditing:
- `injury_llm_used`: boolean
- `injury_llm_run_id`: string
- `injury_llm_packet_path`: string (relative path in `reports/market_snapshots/`)
- `injury_llm_packet_ts`: string (`...Z`)
- `injury_llm_model`: string
- `injury_digest_path`: string
- `injury_digest_sha256`: string
- (optional per-game) `injury_home_ptcs`, `injury_away_ptcs`, and/or `injury_ptcs_min` used for shrinkage

Also update `log_plan.py` output fields to include the above (or at least the packet path + injury deltas) so plan logs can be joined with outcomes without re-running the planner.

### What to Log to `reports/market_snapshots/`

In addition to the existing external market snapshots (`*_external_snapshot_*.csv`):
- Write injury LLM packet + merge audit as JSON (see Section 2).
- Keep these files immutable (append-only by filename time token); never overwrite existing snapshot files.

---

## 4) A/B Evaluation Plan (v2c_raw vs v2c_raw + LLM Injury Deltas)

### Hypothesis

Adding LLM-sized injury deltas improves:
- **Probability quality**: lower Brier and log loss vs actual outcomes.
- **Trading outcomes**: higher realized EV vs Kalshi mid, after applying the same doctrine gates.

### Experimental Definitions

Two model variants for the same games:
- **A (baseline):** `v2c_raw_base` — v2c raw probability computed **without** applying the LLM injury deltas (inj deltas forced to 0.0 for the slate, or pre-LLM state).
- **B (treatment):** `v2c_raw_inj` — v2c raw probability computed with the **LLM injury deltas** applied (possibly PTCS-shrunk).

Notes:
- Keep market data constant between A and B for clean attribution (same Kalshi mids, same sharp prior, same factors).
- For NHL, deltas flow into the Elo component; the treatment effect should be measurable but may be damped by market weighting.

### Data Requirements (for clean offline grading)

To avoid “retroactive edits” and to enable reproducible grading:
- Persist the injury packet + merge audit under `reports/market_snapshots/` (timestamped).
- Persist the baseline and treatment probabilities at plan time (plan JSON/log) so you don’t have to reconstruct later from mutable config files.

### Cross-Validation Setup (Date-Grouped)

Use **date-grouped CV** (group = ledger day) to avoid within-day correlation leakage:

- Split by date, not by game.
- Suggested folds:
  - Rolling/blocked CV (preferred): train on past dates, test on the next block.
  - Or k-fold by date buckets (e.g., 5 folds).

### Metrics (Per Fold and Overall)

Use the definitions in `docs/EV_ANALYSIS.md`:

1) **Brier** (lower is better)
- Compute on all resolved games in the test fold.

2) **Log loss** (lower is better)
- Same population as Brier.

3) **EV vs Kalshi mid** (higher is better)
- Treat each game as 1 unit “long” the side implied by the model vs `kalshi_mid`.
- Compute `avg_pnl_per_bet` and bucket by `|p_model - p_kalshi|` (same as `analyze_ev_vs_kalshi.py`).

4) **Doctrinal trading simulation (optional but recommended)**
- Run the same planner doctrine gates for both A and B (same thresholds), differing only in the input probabilities.
- Compare:
  - number of trades,
  - realized EV per trade,
  - drawdown/variance proxies.

### Calibration Handling

Because calibration can hide or amplify improvements:
- Evaluate **raw** (`v2c_raw_base` vs `v2c_raw_inj`) and also **calibrated** versions.
- Fit calibration parameters **only on training dates**, separately for A and B (avoid leakage).

### Decision Criteria (What “Works”)

Recommend “go live” with LLM injury sizing only if:
- Brier/logloss improves out-of-sample (date-grouped), AND
- EV vs Kalshi improves on the same held-out dates (or at least doesn’t degrade), AND
- Performance doesn’t rely exclusively on a tiny number of “big injury” days (check per-date contribution).

If metrics conflict:
- Prefer **EV vs Kalshi** for trading, but require that calibration doesn’t materially worsen (avoid phantom edge).

---

## Implementation Notes (Non-Binding)

This spec intentionally does not mandate a specific code refactor, but it strongly suggests:
- A single shared injury sizing/merge module used by `run_daily`, `run_pipeline`, and `scheduled_injury_plan`.
- Snapshot-first provenance (market_snapshots JSON) plus safe merge into the simple runtime `injury_adjustments.json`.

