# Project Context: Chimera v2c (Aeternus v2)

> Status note (2025-12): this file provides useful historical context, but the **canonical docs for LLMs/agents** are `bootstrap.md` and `AGENTS.md`. For ledger behavior, always defer to:
> - `bootstrap.md`
> - `OPERATOR_INSTRUCTIONS.md`
> - `reports/daily_ledgers/README.md`

## Project Overview
Chimera v2c (also known as Aeternus v2) is a quantitative sports betting system engine designed for the NBA, NHL, and NFL. It integrates specialist model reports (LLM-generated), Kalshi market execution, and live game monitoring into a unified pipeline. The system operates on a "Scientist ↔ Operator" workflow where calibrated probabilities are turned into controlled staking plans.

**Key Technologies:**
*   **Language:** Python 3.x
*   **Data:** JSON-based storage (`chimera_v2c/data/`), ESPN internal APIs for scores/stats.
*   **Markets:** Kalshi (prediction markets).
*   **Architecture:** Modular design with `src/` (core logic) and `tools/` (CLI scripts).

## Building and Running

**Prerequisites:**
*   Python 3.x
*   Dependencies installed: `pip install -r requirements.txt`
*   Environment variables configured (see `config/env.list`).

**Key Commands:**
All commands should be run from the repository root with `PYTHONPATH=.`.

*   **Run Daily Planning (Pre-game):**
    ```bash
    PYTHONPATH=. python chimera_v2c/tools/run_daily.py --date YYYY-MM-DD
    ```
*   **Log Plan:**
    ```bash
    PYTHONPATH=. python chimera_v2c/tools/log_plan.py --date YYYY-MM-DD
    ```
*   **Execute Plan (Maker-only):**
    ```bash
    PYTHONPATH=. python chimera_v2c/tools/execute_plan.py --dry-run
    # Remove --dry-run to place orders
    ```
*   **Live Monitoring:**
    ```bash
    PYTHONPATH=. python chimera_v2c/tools/kalshi_ws_listener.py --league <nba|nhl|nfl> --series <SERIES_TICKER>
    ```
*   **Running Tests:**
    ```bash
    PYTHONPATH=. pytest chimera_v2c/tests
    ```

## Development Conventions

**Code Style:**
*   **Formatting:** Python 3.x standards, 4-space indentation.
*   **Naming:** `snake_case` for modules/functions/vars, `CamelCase` for classes.
*   **Structure:**
    *   `chimera_v2c/src/`: Pure functions and core pipelines.
    *   `chimera_v2c/tools/`: Thin orchestration scripts and CLI entry points.
    *   `chimera_v2c/data/`: State and model inputs.

**Workflow:**
*   **Conventions:** Adhere to `AGENTS.md` for contribution guidelines.
*   **Testing:** Write unit tests in `chimera_v2c/tests/` for new features. Prefer deterministic tests (mock IO/Network).
*   **Safety:** Always explain critical commands before execution (e.g., actual betting or file modification).

**Documentation:**
*   Refer to `README.md` for detailed operational procedures and league-specific checklists (The "Runbook").
*   Refer to `docs/DOCTRINE_AND_HISTORY.md` for architectural context, betting doctrine, and legacy information.

**Ledger safety (critical):**
* The canonical game ledger lives at `reports/specialist_performance/game_level_ml_master.csv`.
* Only write to it via `PYTHONPATH=. python chimera_v2c/tools/build_game_level_ml_table.py` (append-only; seeds from existing rows).
* Take a snapshot before any rebuild: `PYTHONPATH=. python chimera_v2c/tools/snapshot_game_level_ledger.py`.
* Never run or add scripts that edit the ledger in place (recover/clean/dedupe). All such scripts have been removed.
