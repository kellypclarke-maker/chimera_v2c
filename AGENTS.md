# Repository Guidelines

This document is the canonical contributor guide for `chimera_v2c`. If instructions here ever conflict with other docs, prefer this file for day‑to‑day coding and review.

For any LLM/agent (Codex, Gemini, GPT, etc.), the primary entrypoint is `bootstrap.md`, which provides a doc map and safety summary. After reading that file, follow all rules in this `AGENTS.md`.

LLM/agent requirement: when you change behaviors, tools, or operator workflows, you must update the relevant documentation in the same change (unless the human operator explicitly tells you not to). Do not ship behavioral changes without bringing the docs up to date.

## Project Structure & Modules
- Root: live engine in `chimera_v2c/`, configuration in `config/`, outputs in `reports/`, Python deps in `requirements.txt`.
- Core code: `chimera_v2c/src/` (pipelines, doctrine, risk, logging, DB/WS utilities).
- CLI / jobs: `chimera_v2c/tools/` (ETL, planning, execution, research, listeners).
- Tests: `chimera_v2c/tests/` (pytest). Place new tests alongside the module under test.
- Data & config: `chimera_v2c/data/` JSON inputs, `chimera_v2c/config/` for engine settings.

## Build, Test, and Development
- Create venv (recommended): `python -m venv .venv && source .venv/bin/activate`.
- Install deps from root: `pip install -r requirements.txt`.
- Run unit tests: `PYTHONPATH=. pytest chimera_v2c/tests`.
- Smoke a key pipeline: `PYTHONPATH=. python chimera_v2c/tools/run_daily.py --date YYYY-MM-DD --dry-run`.
- Tools are always invoked from repo root with `PYTHONPATH=.`.

## Coding Style & Naming
- Language: Python 3.x. Use 4‑space indentation, no tabs.
- Follow existing module names (`snake_case.py`, `CamelCase` classes, `snake_case` functions and variables).
- Keep public APIs small and explicit; prefer pure functions in `src/` and thin orchestration in `tools/`.
- When in doubt, mirror patterns in `chimera_v2c/src/pipeline.py` and `chimera_v2c/tools/run_daily.py`.
- Team codes: always normalize external team strings via `chimera_v2c.lib.team_mapper.normalize_team_code(code, league)` before writing to ledgers or canonical files. If a new alias appears, add it to `team_mapper` so all tools stay in sync (ESPN/Kalshi/Odds API are unified there).

## Testing Guidelines
- Use `pytest` with descriptive test names: `test_<module>_<behavior>`.
- For new features, add tests under `chimera_v2c/tests/` that cover happy path and at least one edge case.
- Prefer deterministic tests: avoid live network or Kalshi calls; mock IO and external services.
- Run `pytest chimera_v2c/tests` before opening a PR.

## Commit & Pull Request Practices
- Commits: use clear, present‑tense messages (e.g., `Add risk cap for live guardrails`, `Refactor ws_mid_cache overlay`).
- Keep commits focused; group mechanical renames or formatting separately from logic changes.
- PRs: include a short summary, rationale, and testing notes (`pytest`, key tools run). Link issues when applicable and attach logs or snippets for behavior changes.
- Avoid touching `_archive` or legacy v1 paths unless explicitly required; all new work should target the v2c engine.
- Keep documentation in sync with behavior:
  - If you add or change a tool under `chimera_v2c/tools/`, update `chimera_v2c/tools/README.md`.
  - If you change the daily pipeline or core runbook, update `README.md` and `chimera_v2c/README.md`.
  - If you change ledger behavior or safety rules, update `OPERATOR_INSTRUCTIONS.md` and `reports/daily_ledgers/README.md`.
  - If you change LLM bootstrap behavior or expectations, update `bootstrap.md` (and this `AGENTS.md` section if needed).
- **Ledger safety:** The canonical game records are the per‑day ledgers under `reports/daily_ledgers/` (`YYYYMMDD_daily_game_ledger.csv`). Treat each file as append‑only: add rows or fill blank cells, but never bulk‑overwrite probabilities or delete rows.
- **Legacy master ledger (read-only):** A historical game‑level master lives at `reports/specialist_performance/game_level_ml_master.csv`. It is read‑only and exists only as a reference and as the source used to backfill pre‑2025‑12‑04 daily ledgers. Do not create new scripts that write to it or run `build_game_level_ml_table.py` in normal workflows.
- **Daily ledgers (immutable):** Generate per-day files with `chimera_v2c/tools/build_daily_game_ledgers.py` (writes `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv`). These files must never be auto-overwritten; only replace with an explicit `--overwrite` when intentionally regenerating a single day.
- **Daily step:** Make step #1 each day `PYTHONPATH=. python chimera_v2c/tools/ensure_daily_ledger.py` (defaults to today, header-only if no rows yet). No automation may overwrite existing daily files unless a human passes `--overwrite` for that specific date.
- **Master ledger (derived, guarded):** Do not hand-edit `reports/master_ledger/master_game_ledger.csv`. Rebuild it via `chimera_v2c/tools/build_master_ledger.py`, which snapshots the prior file and only appends or fills blank cells. If you need to touch a daily ledger, lock the date first and rebuild the master from the locked files.
