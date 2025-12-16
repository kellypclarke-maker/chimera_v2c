# Injury Snapshots (LLM Injury Delta Audits)

This directory holds **timestamped change logs and file snapshots** created when
LLM tools write per-team injury deltas into:

- `chimera_v2c/data/injury_adjustments.json`

These artifacts exist to answer: “what changed, when, and why?” without relying
on memory.

## What Writes Here

- `chimera_v2c/tools/apply_llm_injuries_v2.py` (NBA/NFL generic)
- `chimera_v2c/tools/apply_llm_nhl_injuries.py` (NHL-specific)

## File Types

- `injury_llm_changes_<YYYY-MM-DD>_<league>_<YYYYMMDDTHHMMSSZ>.json`
  - A compact per-team diff (`old` → `new`) for that league/date, plus pointers
    to the raw LLM audit and the input digest path.
- `injury_adjustments.json.<YYYYMMDDTHHMMSSZ>.bak` (+ `.sha256`)
  - A full snapshot of the pre-write `chimera_v2c/data/injury_adjustments.json`
    file for rollback/debugging.

## Notes

- These snapshots are **not** a source of truth for modeling; the engine reads
  only `chimera_v2c/data/injury_adjustments.json`.
- Daily ledgers remain append-only and are not written by these tools.

