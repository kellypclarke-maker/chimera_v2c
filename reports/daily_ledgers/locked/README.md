Locked Daily Ledgers
====================

- This directory holds lock markers (`YYYYMMDD.lock`) for daily ledgers that are finalized.
- When a lock file exists, tools should refuse to modify that date unless an explicit `--force` (or equivalent) is provided by a human operator.
- Create a lock after finishing a day’s ledger to prevent accidental overwrites. Example: `touch reports/daily_ledgers/locked/20251210.lock`.
