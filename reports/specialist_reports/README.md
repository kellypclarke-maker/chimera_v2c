# Specialist Reports Workflow

This folder is the **canonical home** for all Specialist reports that feed the per‑day ledgers and calibration tables. Treat it as append‑only: reports move between subfolders, but nothing is ever deleted without an explicit human request, and ledger rows are never removed.

## Folder Layout

- `NBA/` – Canonical, per‑game NBA reports (`.txt`), organized into monthly subfolders (e.g. `2025-10/`, `2025-11/`, `2025-12/`).
- `NFL/` – Canonical, per‑game NFL reports, also organized by `YYYY-MM/`.
- `NHL/` – Canonical, per‑game NHL reports, also organized by `YYYY-MM/`.
- `archive/` – Single consolidated archive for all raw and legacy reports:
  - `archive/raw/` – Drop zone for freshly downloaded reports (Gemini/Grok/GPT), including long slate or deep‑dive documents.
  - `archive/raw_processed/` – Raw reports that have already been parsed into canonical per‑game files.
  - `archive/raw_unparsed/` – Raw reports that failed parsing and need manual review or cleanup.
  - `archive/versioned/` – Legacy, versioned canonical copies kept for historical reference.

## Canonical File Naming

- Per‑game processed reports must use:
  - `YYYYMMDD_<league>_<away>@<home>_<model>.txt`
  - Keep `model` to the family name (`gemini`, `grok`, `gpt`) for cleanliness; directive/version can live in the header, not the filename.
  - Examples:
    - `20251203_nba_bkn@chi_gemini.txt`
    - `20251203_nhl_buf@phi_grok.txt`
    - `20251207_nfl_mia@nyj_gpt.txt`
- `league` is `nba|nfl|nhl` (lowercase).
- `away@home` uses canonical team codes in lowercase (e.g., `gsw@lal`, `buf@phi`). Abbreviations must match the official schedule.
- The filename and the HELIOS header **must** agree on date/league/matchup and correspond to a real game in the season schedule CSVs under `reports/thesis_summaries/`.

Every per‑game canonical file must start with a **HELIOS prediction header**:

```text
HELIOS_PREDICTION_HEADER_START
Game: YYYY-MM-DD <LEAGUE> AWAY@HOME
Model: <ModelLabel>

Prediction_Moneyline:
Winner: <AWAY_or_HOME_code>
p_home: <0.00-1.00>   # REQUIRED: probability the home team wins
p_true: <0.00-1.00>   # optional legacy field (winner win prob)
ptcs: 
HELIOS_PREDICTION_HEADER_END
```

Narrative content may follow the header and is optional for ingestion.

## End‑to‑End Processing Flow (Raw → Canonical → Daily Ledger → Archive)

When a raw specialist report is dropped into `reports/specialist_reports/archive/raw/`, follow this sequence.

### 1. Deep Research (Human)

- Human spins up Gemini/Grok/GPT for a slate or specific game and saves the raw output (as‑downloaded) into `reports/specialist_reports/archive/raw/`.

### 2. Process Raw → Canonical (LLM Task)

Instruction: “Go process the Gemini/Grok/GPT reports in `reports/specialist_reports/archive/raw/`.”

For each raw `.txt` file:

- Parse the games:
  - If the raw file already contains one or more `HELIOS_PREDICTION_HEADER_START ... HELIOS_PREDICTION_HEADER_END` blocks, treat each block as a game.
  - If there is no HELIOS header yet (legacy narrative formats), you must synthesize one for each game before writing canonical files.
- For each game:
  - Determine `date`, `league`, and `AWAY@HOME` (from the HELIOS header if present, otherwise from context/filename).
  - Create a new per‑game `.txt` in the correct league folder (`NBA/`, `NFL/`, `NHL/`) under the appropriate monthly subfolder (e.g., `2025-12/`), using the **canonical filename pattern** above.
  - Ensure the per‑game file begins with a single HELIOS header that includes:
    - `Game: YYYY-MM-DD LEAGUE AWAY@HOME`
    - `Model: <family + directive>` (e.g., `Gemini_Scientist_v8.2`, `Grok_Scientist_v8.2`, `GPT_Scientist_v8.2`).
    - A `Prediction_Moneyline` block with `Winner` and **`p_home`**. (`p_true` may be included as a legacy field, but `p_home` is the ingestion target.)
  - Preserve narrative content below the header (you may trim for brevity if needed, but do not change the header fields).

For historical backfills where canonical reports exist but lack headers, you can use:

- `PYTHONPATH=. python chimera_v2c/tools/add_helios_headers_to_canonical.py`

Preview-first automation:
- `PYTHONPATH=. python chimera_v2c/tools/ingest_raw_specialist_reports_v2c.py --dry-run`
  will parse/synthesize HELIOS headers, show intended canonical filenames and ledger fills,
  and **will not write**. Add `--apply` to write canonical files, fill blank model cells in the
  daily ledger (append-only, respects lockfiles unless `--force`), and move raw files into
  `archive/raw_processed/` or `archive/raw_unparsed/`.

This script will synthesize or wrap HELIOS headers for older canonical files without touching daily ledgers.

### 3. Update Daily Ledgers (LLM Task)

For each canonical per‑game report:

- Extract from the HELIOS header:
  - `date` (YYYY‑MM‑DD),
  - `league` (`NBA`, `NFL`, `NHL`),
  - `AWAY@HOME`,
  - `Winner` and `p_home` from `Prediction_Moneyline` if available.
- Convert `date` to `YYYYMMDD` and open the corresponding daily ledger:
  - `reports/daily_ledgers/YYYYMMDD_daily_game_ledger.csv`
  - If the file does not exist yet for that date, create it via:
    - `PYTHONPATH=. python chimera_v2c/tools/ensure_daily_ledger.py --date YYYY-MM-DD`
- In that ledger:
  - Ensure there is exactly one row per `(league, matchup)` (matchup is `AWAY@HOME`):
    - If no row exists yet for that game, append a new row with:
      - `date`, `league`, `matchup` filled,
      - All model/market columns blank except the one you are updating.
  - Determine the correct model column:
    - `gemini` for Gemini family models.
    - `grok` for Grok.
    - `gpt` for GPT.
    - `v2c` for the internal Chimera engine.
  - Determine `p_home`:
    - Prefer `p_home` directly from the HELIOS header.
    - Legacy fallback (if only `p_true` exists): if `Winner` is home, `p_home = p_true`; else `p_home = 1.0 - p_true`.
    - Round `p_home` to two decimal places (e.g., `0.56`).
  - Write `p_home` into the appropriate model column **only if that cell is currently blank**:
    - Do not change existing non‑blank probabilities.

Do **not** write to any legacy master ledger or `reports/specialist_performance/specialist_manual_ledger.csv` as part of normal v2c operations; the per‑day files in `reports/daily_ledgers/` are the canonical source of truth.

For grading and backfills (outcomes):

- Use ESPN/scoreboard helpers (e.g., `chimera_v2c/lib/espn_schedule.py`, `chimera_v2c/lib/nhl_scoreboard.py`) or the dedicated tool:
  - `PYTHONPATH=. python chimera_v2c/tools/fill_missing_daily_outcomes.py`
- Only fill `actual_outcome` cells that are blank; do not overwrite known results.

### 4. Archive Raw (LLM Task)

- After all per‑game canonical files and daily ledger updates are written successfully:
  - Move the original raw report from `archive/raw/` into either:
    - `archive/raw_processed/` (parsed successfully), or
    - `archive/raw_unparsed/` (failed parsing, needs manual review).
- Do **not** overwrite or delete raw files during normal operation; moving preserves a full audit trail.

## Data Integrity Rules

- **Daily ledgers are canonical**:
  - Game‑level metrics, probabilities, and final scores must be read from and written to the per‑day files in `reports/daily_ledgers/`.
  - Any legacy master ledger under `archive/specialist_performance/` is read‑only and used only for historical backfill.
- **Append‑only behavior**:
  - Never delete or rewrite existing rows in daily ledgers.
  - Only add new rows or fill blank cells in the appropriate model columns and `actual_outcome`.
- **Canonical per‑game reports**:
  - Once written, per‑game canonical `.txt` files under `NBA/`, `NFL/`, `NHL/` should not be destroyed.
  - If a report is found to be a hallucination (game never played according to ESPN/official schedule):
    - Prefer moving the file into `archive/raw_unparsed/` (or another clearly marked quarantine folder).
    - Add a note in the relevant daily ledger row (if one exists) explaining the hallucination.
    - Only delete canonical files for hallucinated games when a human explicitly asks you to do so.
- **No bulk regeneration**:
  - Any cleaning or backfill should operate via dedicated tools that:
    - Read from existing daily ledgers and canonical reports,
    - Write derived tables (e.g., `model_wr_by_league.csv`),
    - Avoid mutating historical data in place.

## LLM / Agent Checklist (Raw Report Ingest)

When you see a request like “I just dropped new specialist reports in `archive/raw/`, please wire them up,” do the following:

1. Read this `reports/specialist_reports/README.md` and `reports/daily_ledgers/README.md`.
2. For each raw file under `reports/specialist_reports/archive/raw/`:
   - Parse or synthesize HELIOS headers per game.
   - Write per‑game canonical `.txt` files in the correct league/month folder with a proper HELIOS header.
   - Update the corresponding `YYYYMMDD_daily_game_ledger.csv` file:
     - Add missing rows,
     - Fill the correct model column with `p_home` as described above.
   - Move the raw file into `archive/raw_processed/` or `archive/raw_unparsed/`.
3. Report back:
   - Which games were ingested,
   - Which daily ledgers were updated,
   - Any files that could not be parsed or matched to real games (and how you quarantined them).
