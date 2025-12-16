"""Shared safeguards for ledger writes.

This module centralizes the append-only and lock semantics so tools do not
silently overwrite canonical ledgers. Import this before writing any ledger.
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple


Record = MutableMapping[str, str]


@dataclass(frozen=True)
class AppendOnlyDiff:
    added: List[Tuple[str, str]]
    filled: List[Tuple[str, str, str, str]]


class LedgerGuardError(RuntimeError):
    """Raised when a write would violate append-only or lock rules."""


def load_csv_records(path: Path) -> List[Record]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def snapshot_file(path: Path, snapshot_dir: Path) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    dst = snapshot_dir / f"{path.name}.{ts}.bak"
    shutil.copy2(path, dst)
    checksum = hashlib.sha256(dst.read_bytes()).hexdigest()
    (dst.with_suffix(dst.suffix + ".sha256")).write_text(checksum, encoding="utf-8")
    return dst


def load_locked_dates(lock_dir: Path) -> Set[str]:
    if not lock_dir.exists():
        return set()
    return {p.stem for p in lock_dir.glob("*.lock") if p.is_file()}


def assert_no_locked_mutations(
    new_rows: Sequence[Record],
    locked_dates: Set[str],
    date_field: str = "date",
    allow_locked: bool = False,
) -> None:
    if allow_locked or not locked_dates:
        return
    violating = {row.get(date_field, "") for row in new_rows if row.get(date_field, "") in locked_dates}
    if violating:
        raise LedgerGuardError(
            f"Refusing to modify locked dates: {sorted(violating)}. Pass an explicit override if intended."
        )


def compute_append_only_diff(
    old_rows: Sequence[Record],
    new_rows: Sequence[Record],
    key_fields: Sequence[str],
    value_fields: Sequence[str],
    *,
    blank_sentinels: Set[str] | None = None,
) -> AppendOnlyDiff:
    blank_norm = {s.strip().lower() for s in (blank_sentinels or set()) if s and s.strip()}

    def normalize(val: object) -> str:
        s = (str(val) if val is not None else "").strip()
        if not s:
            return ""
        if blank_norm and s.lower() in blank_norm:
            return ""
        return s

    old_map = _index_by_key(old_rows, key_fields)
    new_map = _index_by_key(new_rows, key_fields)

    removed = set(old_map) - set(new_map)
    if removed:
        raise LedgerGuardError(f"Append-only violation: attempted to drop keys {sorted(removed)[:5]}...")

    added = sorted(set(new_map) - set(old_map))
    filled: List[Tuple[str, str, str, str]] = []

    for key in set(new_map) & set(old_map):
        old = old_map[key]
        new = new_map[key]
        for field in value_fields:
            old_val = normalize(old.get(field))
            new_val = normalize(new.get(field))
            if old_val and not new_val:
                raise LedgerGuardError(
                    f"Append-only violation for {key}: field '{field}' lost value '{old_val}'."
                )
            if old_val and new_val and old_val != new_val:
                raise LedgerGuardError(
                    f"Append-only violation for {key}: field '{field}' overwrite '{old_val}' -> '{new_val}'."
                )
            if not old_val and new_val:
                filled.append((str(key), field, old_val, new_val))

    return AppendOnlyDiff(added=[tuple(a) for a in added], filled=filled)


def _index_by_key(rows: Sequence[Record], key_fields: Sequence[str]) -> Dict[Tuple[str, ...], Record]:
    keyed: Dict[Tuple[str, ...], Record] = {}
    for row in rows:
        key = tuple((row.get(k) or "").strip() for k in key_fields)
        keyed[key] = row
    return keyed


def write_csv(path: Path, rows: Iterable[Mapping[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
