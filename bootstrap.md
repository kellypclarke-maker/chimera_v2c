# Chimera v2c — LLM / Agent Bootstrap

This file (`bootstrap.md`) is the **single entrypoint** for any LLM/agent (Codex, Gemini, GPT, etc.) working in this repository.

- If you are an LLM: **read this file first**.
- For coding style and day‑to‑day rules, also follow `AGENTS.md` (canonical contributor guide).
- For safety‑critical topics (ledgers, specialist reports), follow the specific docs linked here.

---

## 0. LLM Behavior Protocol

When a user says anything like **“read the onboarding guide”**, **“read bootstrap”**, **“read bootstrap.md”**, or **“read the onboarding txt”**, do the following **automatically**, without asking for extra confirmation:

1. **Silently read this file (`bootstrap.md`)** from the repo root.  
   - Do **not** dump the full file contents back to the user unless they explicitly ask to see them.
2. **Follow the Startup Checklist (Section 4) for the minimal docs needed for the task**:  
   - Always read `AGENTS.md`.  
   - Skim `README.md` (do not re‑phrase it at length).  
   - Read `chimera_v2c/README.md` and `chimera_v2c/tools/README.md` only if you may touch tools.  
   - Read `OPERATOR_INSTRUCTIONS.md` and `reports/daily_ledgers/README.md` only before ledger work.  
   - Read `reports/market_snapshots/README.md` only before market-snapshot work.  
   - Read `reports/specialist_reports/README.md` only before specialist‑report work.  
   - Keep your notes internal; in chat, summarize each doc in at most a few short bullets, only when relevant.
3. **Then reply with a very short status and a small menu of next actions**, for example:
   - One or two bullets on:
     - What you read and understood (v2c scope, key safety rules).
     - That you are ready to operate within those constraints.
   - A short menu of options, e.g.:
     - `A)` Analyze EV vs Kalshi for a date/league.  
     - `B)` Plan and log today’s slate (paper‑mode).  
     - `C)` Work on calibration or configs.  
     - `D)` Explain the current pipeline / docs.  
     - `E)` **Something else (custom task)** — user describes.  
   - Ask the user to choose a letter (and any needed date/league) **before** running long or multi‑step workflows.

Treat this protocol as the default behavior for any new shell/session in this repo.

---

## 1. System Overview (One Screen)

- **Name / scope:** Chimera v2c (Aeternus v2) — maker‑only, pre‑game engine for Kalshi game markets (NBA, NHL, NFL).
- **Related workflow (separate track):** Rule A (taker) “home-favorite fade” research + OOS logging lives under `chimera_v2c/tools/` (does not place orders; default outputs go to `reports/execution_logs/`).
- **Goal:** Turn calibrated win probabilities + market context into conservative, maker‑only plans and optional executions.
- **Core doctrine:**
  - Stats + market ensemble (Elo, Four Factors/process metrics, market mids/sharps, injuries/news refresh as mandatory preflight, calibration).
  - Edge thresholds + confluence gaps + fee buffer.
  - Quarter‑Kelly with per‑game and daily caps.
  - One side per game, **maker‑only** pricing (no ladders, no multi‑rung traps).
- **Key directories:**
  - Code: `chimera_v2c/src/` (pipelines, doctrine, risk, logging, utilities).
  - Tools / CLIs: `chimera_v2c/tools/` (ETL, planning, execution, backtests, research).
  - Data: `chimera_v2c/data/` (ratings, factors, injuries, calibration).
  - Config: `chimera_v2c/config/` (YAML configs per league).
  - Logs: `reports/execution_logs/` (plans, executions, results).
  - Ledgers (canonical): `reports/daily_ledgers/` (per‑day, append‑only game ledgers).

For more doctrine/background, open `docs/DOCTRINE_AND_HISTORY.md`.

---

## 2. Absolute Safety Rules (Read Before Writing)

These rules are **non‑negotiable** for any LLM or automation.

### 2.1 Ledgers (Canonical, Append‑Only)

- Canonical game records are the **per-day ledgers**:
  - Location: `reports/daily_ledgers/`
  - Pattern: `YYYYMMDD_daily_game_ledger.csv`
- Treat each per-day ledger as **append-only**:
  - You may **add new rows** (e.g., new games).
  - You may **fill blank cells** in existing rows (e.g., `gpt`, outcomes).
  - You **must not** delete rows or bulk-overwrite existing probabilities.
- Lock finished days: drop a marker in `reports/daily_ledgers/locked/YYYYMMDD.lock` when a day is complete; tools must refuse changes to locked dates unless explicitly forced.
- Master ledger (derived): `reports/master_ledger/master_game_ledger.csv` is generated via `PYTHONPATH=. python chimera_v2c/tools/build_master_ledger.py` and is never hand-edited. The tool snapshots the prior master and only appends or fills blanks.
- Daily creation:
  - Use `chimera_v2c/tools/ensure_daily_ledger.py` (preferred) or sanctioned backfill tools.
  - Never auto‑overwrite daily ledgers unless a human explicitly passes `--overwrite` for a specific date.
- Historical master:
  - Any game‑level master tables (e.g., under `archive/` or `reports/specialist_performance/`) are **derived artifacts** and **not** the canonical source of truth.

Before touching anything under `reports/daily_ledgers/`, also read:

- `reports/daily_ledgers/README.md`
- `OPERATOR_INSTRUCTIONS.md`

### 2.2 Specialist / LLM Reports

- Canonical specialist reports live under `reports/specialist_reports/` (NBA/NHL/NFL subfolders).
- Raw, longform LLM outputs live under `reports/specialist_reports/raw/` and must be:
  - Parsed into per‑game canonical files.
  - Archived to `reports/specialist_reports/archive/raw_processed/` or `reports/specialist_reports/archive/raw_unparsed/` after processing.
- Never delete or rewrite past specialist ledger rows; only append or fill missing fields.
- For v8.4‑style directives that rely on tabular grounding, prefer using the prebuilt slate‑level CSV packets under `reports/llm_packets/<league>/<YYYYMMDD>/` (via `build_llm_packets.py`) as the primary truth source for standings, injuries, schedule/fatigue, odds, and H2H instead of re‑scraping external sites ad hoc in the prompt.

Before working on specialist flows, read:

- `reports/specialist_reports/README.md`

### 2.3 Legacy / Ladders / Archive

- v1/v7 “betting ladders” and any ladder workflows are **legacy only** and **must not** be re‑introduced into v2c.
- Do **not** modify `_archive` or explicit legacy paths unless a human explicitly asks you to.
- Tools not listed in `chimera_v2c/tools/README.md` should be treated as **legacy**.

---

## 3. Doc Map — Where to Look for What

Use this as your navigation map. Only open what you need for the task.

- **Contributor & agent rules**
  - `AGENTS.md` — canonical contributor guide (project structure, coding style, testing, ledger safety summary).

- **High‑level system behavior / runbook**
  - `README.md` — Phase‑1 pre‑game overview, quickstart commands, pipeline steps, and top‑level ledger reminders.
  - `docs/DOCTRINE_AND_HISTORY.md` — v2c doctrine, history, and rationale.

- **Engine & tools**
  - `chimera_v2c/README.md` — engine directory overview; key entry‑point tools and data inputs/outputs.
  - `chimera_v2c/tools/README.md` — authoritative list of active tools (daily flow, ETL, ledgers, backtests, research, legacy).
  - `docs/EV_ANALYSIS.md` — EV vs Kalshi market and calibration analysis using daily ledgers (read‑only), including
    `analyze_ev_vs_kalshi.py` and `fit_calibration_from_daily.py` for per‑league v2c calibration.

- **Ledgers & operator safety**
  - `OPERATOR_INSTRUCTIONS.md` — human/agent operator rules; what is allowed vs forbidden for ledgers.
  - `reports/daily_ledgers/README.md` — detailed rules for per‑day ledgers (immutable append‑only, backfill boundaries).
  - `reports/specialist_reports/README.md` — end‑to‑end specialist report workflow.

- **Roadmap / priorities**
  - `docs/gpt_roadmap_v2c_phase1_to_phase2.md` — current roadmap from Phase 1 (pre‑game) to Phase 2 (live), with what is done vs partial vs deferred.

- **Legacy context**
  - `docs/GEMINI.md` — older but still useful project overview; treat its **ledger section as legacy** and defer to `AGENTS.md` + the daily ledger docs above.

---

## 4. Startup Checklist for a New LLM / Agent

When a fresh instance is started and pointed at this repo, follow this sequence:

1. **Read this file (`bootstrap.md`)** to understand scope and safety rules.
2. **Read `AGENTS.md`** for coding style, test expectations, and additional ledger constraints.
3. **Skim `README.md`** for the active Phase‑1 pre‑game pipeline and key commands.
4. If you will run or modify tools:
   - Read `chimera_v2c/README.md` and `chimera_v2c/tools/README.md`.
5. If you will touch ledgers or anything under `reports/daily_ledgers/`:
   - Read `OPERATOR_INSTRUCTIONS.md` and `reports/daily_ledgers/README.md` completely.
6. If you will work with specialist reports:
   - Read `reports/specialist_reports/README.md`.
7. For larger refactors or new features:
   - Consult `docs/gpt_roadmap_v2c_phase1_to_phase2.md` to align with current priorities.

### 4.1 Kalshi Markets (Read‑Only Access)

- Live Kalshi markets are available from the public HTTP API at  
  `https://api.elections.kalshi.com/trade-api/v2/markets` without private keys.
- The helper `kalshi_utils.list_public_markets` defaults to the demo base (`demo-api.kalshi.co`) **unless** `KALSHI_PUBLIC_BASE` is set. In this repo, `load_env_from_env_list()` will set `KALSHI_PUBLIC_BASE` from `KALSHI_BASE`/`KALSHI_API_BASE` when available, so planners/snapshots should see live GAME markets by default.
- For read‑only GAME mids (e.g., to fill `kalshi_mid` or analyze EV) an LLM *may*:
  - Call `GET https://api.elections.kalshi.com/trade-api/v2/markets` with `status=open`, `series_ticker=KXNBAGAME` (or `KXNHLGAME` / `KXNFLGAME`), and `limit` as needed, or
  - Set `KALSHI_PUBLIC_BASE=https://api.elections.kalshi.com/trade-api/v2` and then use existing helpers that rely on the public base.
- When mapping markets to games, prefer the existing `market_linker` utilities so tickers like `KXNBAGAME-25DEC10PHXOKC-OKC` resolve to matchup keys like `PHX@OKC` and home‑implied mids.

---

## 5. Environment, Commands, and Testing (LLM‑Friendly Summary)

**Environment**

- Language: Python 3.x.
- Recommended: local virtualenv.
  - `python -m venv .venv && source .venv/bin/activate`
- Install dependencies from repo root:
  - `pip install -r requirements.txt`
- Always run tools from repo root with `PYTHONPATH=.`.

**Core Commands**

- **Run daily planner (per league)**:
  - `PYTHONPATH=. python chimera_v2c/tools/run_daily.py --config <league_config> --date YYYY-MM-DD`
  - Optional: add `--llm-injuries` to apply LLM-derived injury deltas from the ESPN digest (requires `OPENAI_API_KEY`).
- **Rule A (taker) daily wrapper (planning + reconciliation; no execution)**:
  - `PYTHONPATH=. python chimera_v2c/tools/run_rule_a_daily.py plan --date YYYY-MM-DD --leagues nba,nhl --write-research-queue`
  - `PYTHONPATH=. python chimera_v2c/tools/run_rule_a_daily.py reconcile --date YYYY-MM-DD --leagues nba,nhl`
- **Log and (optionally) execute**:
  - `PYTHONPATH=. python chimera_v2c/tools/log_plan.py --config <league_config> --date YYYY-MM-DD`
  - `PYTHONPATH=. python chimera_v2c/tools/execute_plan.py --config <league_config> --date YYYY-MM-DD --dry-run`
  - Optional: add `--llm-injuries` to either command to apply LLM-derived injury deltas from the ESPN digest (requires `OPENAI_API_KEY`).
- **One‑command pipeline**:
  - `PYTHONPATH=. python chimera_v2c/tools/run_pipeline.py --leagues all --date YYYY-MM-DD --refresh-factors --fit-calibration --backtest-days 7 --skip-preflight`
- **Daily ledger step**:
  - `PYTHONPATH=. python chimera_v2c/tools/ensure_daily_ledger.py --date YYYY-MM-DD`
  - Seed `v2c` + `kalshi_mid` from the planner (append-only):
    - `PYTHONPATH=. python chimera_v2c/tools/fill_daily_ledger_from_plan.py --date YYYY-MM-DD --config <league_config> --apply`
  - Optional: fill `market_proxy` (books no‑vig) from Odds API history (append-only):
    - `PYTHONPATH=. python chimera_v2c/tools/backfill_market_proxy_from_odds_history.py --league <league> --start-date YYYY-MM-DD --end-date YYYY-MM-DD --apply`
  - Optional: capture timestamped external baselines (Kalshi + books + MoneyPuck) and fill ledger blanks (append-only):
    - `PYTHONPATH=. python chimera_v2c/tools/external_snapshot.py --league <league> --date YYYY-MM-DD --apply`
  - When finalizing a day, enforce canonical ledger formatting (single-date, snapshots before writing):
    - `PYTHONPATH=. python chimera_v2c/tools/format_daily_ledger.py --date YYYY-MM-DD --apply`

**Testing**

- Run tests from repo root:
  - `PYTHONPATH=. pytest chimera_v2c/tests`
- When adding or modifying code:
  - Add or update tests under `chimera_v2c/tests/` near the module you changed.
  - Prefer deterministic tests; mock network / external services.

---

## 6. Task‑Specific Guidance for LLMs

**When editing code**

- Follow `AGENTS.md` for style, structure, and testing.
- Keep orchestration in `chimera_v2c/tools/` thin; keep core logic in `chimera_v2c/src/`.
- When you add a new tool or significantly change behavior of an existing one:
  - Update `chimera_v2c/tools/README.md` to document it and mark whether it is active or legacy.
- When you change core pipeline behavior or primary workflows:
  - Update `README.md` and `chimera_v2c/README.md` so the runbook matches what the code does.
- When you change anything related to ledger handling, backfills, or append‑only guarantees:
  - Update `OPERATOR_INSTRUCTIONS.md` and `reports/daily_ledgers/README.md` to reflect the new rules or safeguards.
- When you change expectations for how LLMs should bootstrap or operate:
  - Update this `bootstrap.md` file (and the note at the top of `AGENTS.md` if needed) so future agents inherit the new behavior.
- When you add or change any **LLM data packet** tooling (e.g., CSVs for standings/injuries/schedule/odds or H2H used by specialist directives):
  - Update `chimera_v2c/tools/README.md` with how to run the packet builder and what files it produces.
  - If this changes how specialists should ground their reports, add a short note under “Specialist / LLM Reports” in this file so future agents know to use the packets instead of re-scraping the world.

**When touching ledgers**

- Do **not** write new scripts that:
  - Iterate over all dates and regenerate daily ledgers in bulk, or
  - Overwrite multiple daily files automatically.
- Operate on the **single date file** you were asked to modify, and only append rows or fill blank cells.
- If a task appears to require bulk rewriting, stop and ask a human operator to confirm.

**When working with specialist reports**

- Always follow `reports/specialist_reports/README.md` for the exact raw → canonical → daily‑ledger → archive workflow.
- Treat per‑game specialist reports and daily ledgers as append‑only artifacts:
  - Do not delete historical specialist `.txt` files except when a human explicitly requests it.
  - Do not overwrite non‑blank probability cells in daily ledgers; only add rows or fill blanks as described in the specialist README.
- Before writing any code or scripts that touch specialist reports or their ledger integration:
  - Re‑read `reports/specialist_reports/README.md` and `reports/daily_ledgers/README.md`.
  - Prefer extending the existing flow over inventing a parallel ingestion pipeline.

**When encountering legacy docs or tools**

- If a document or script conflicts with `AGENTS.md`, `bootstrap.md`, or the daily ledger docs:
  - Prefer this file + `AGENTS.md` + `reports/daily_ledgers/README.md`.
- Flag inconsistencies in your summary back to the human operator instead of silently following legacy behavior.

---

## 7. Quick Glossary

- **v2c:** Current, active maker‑only engine (no ladders).
- **Daily ledger:** Per‑day CSV in `reports/daily_ledgers/`, canonical record of probabilities, mids, and outcomes.
- **Plan log:** JSON/CSV outputs in `reports/execution_logs/` from `run_daily.py` / `log_plan.py` / `execute_plan.py`.
- **Specialist report:** LLM‑generated research file under `reports/specialist_reports/`, eventually feeding into ledgers and calibration.

If you are an LLM and have read this file plus the documents it references for your task type, you should be **bootstrap‑ready to act as the operator** for Chimera v2c within the constraints above.

---

## 8. LLM Cost & Context Discipline

To keep new sessions fast and cheap while still safe:

- **Do not dump large files or docs into chat.**  
  - When you read `AGENTS.md`, `README.md`, tool docs, or EV docs, keep summaries to a few short bullets and only when needed for the current task.
- **Only open what you need.**  
  - Do not pre‑scan the entire repo or open every linked doc “just in case.”  
  - For ledgers, use the existing tools (`ensure_daily_ledger.py`, `analyze_ev_vs_kalshi.py`, `fit_calibration_from_daily.py`, `run_daily.py`, `log_plan.py`) instead of inventing new backfills.
- **Be surgical with ledgers.**  
  - Operate on the single requested date (e.g., `2025‑12‑10`) and league(s).  
  - Never overwrite existing daily ledger files unless a human explicitly asks and provides `--overwrite`/`--force` for that specific date.
- **Ask before heavy workflows.**  
  - After bootstrapping, always present a short A/B/C/… menu and wait for the user’s choice before running multi‑step pipelines, rebuilding master tables, or backfilling many days.  
  - Use “Something else (custom task)” when the user’s request is outside the standard options.
