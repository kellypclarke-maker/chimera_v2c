"""
NHL-specific LLM injury delta applier (safe merge) for injury_adjustments.json.

Prompt focuses on goalie penalties (GSAx-equivalent) and skater role-based deltas.
Clamp to [-40, 0]. Audit writes to reports/specialist_reports/raw/injury_llm_<date>_nhl.json.

Usage:
  PYTHONPATH=. OPENAI_API_KEY=... python chimera_v2c/tools/apply_llm_nhl_injuries.py --date 2025-12-09 --input injuries.txt
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

from chimera_v2c.lib.team_mapper import normalize_team_code
from chimera_v2c.src.ledger.guard import snapshot_file

INJURY_PATH = Path("chimera_v2c/data/injury_adjustments.json")
AUDIT_DIR = Path("reports/specialist_reports/raw")
SNAPSHOT_DIR = Path("reports/injury_snapshots")

SYSTEM_PROMPT = """
You are an NHL injury-to-rating estimator. Return strictly JSON with per-team Elo-like deltas (negative = penalty).
Schema:
{
  "NHL": {
    "YYYY-MM-DD": {
      "TEAM": -25.0,
      ...
    }
  }
}
Rules:
- Use NHL team codes (ANA, BOS, TOR, etc.).
- Penalties only; clamp each to [-40, 0]. Never return positives.
- Goalie injuries: starter out = -20 to -40; backup out = -5 to -15.
- Skaters: top-line/top-pair = -10 to -20; middle six / middle pair = -5 to -10; depth = -1 to -5.
- If multiple injuries, sum them into one team delta for the date.
- If impact is uncertain or minor, use 0.0 rather than guessing.
- Do not include explanations; JSON only.
"""


def load_existing() -> dict:
    if not INJURY_PATH.exists():
        return {}
    try:
        return json.loads(INJURY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _normalize_team(team: str) -> str:
    norm = normalize_team_code(team, "nhl")
    return (norm or team or "").strip().upper()


def merge(existing: dict, new_data: dict) -> dict:
    out = existing.copy()
    for league, dates in new_data.items():
        league_key = str(league or "").strip().upper() or "NHL"
        out.setdefault(league_key, {})
        for date_str, teams in dates.items():
            out[league_key].setdefault(date_str, {})
            for team, delta in teams.items():
                team_key = _normalize_team(str(team))
                if not team_key:
                    continue
                try:
                    val = float(delta)
                except Exception:
                    continue
                val = max(-40.0, min(0.0, val))
                out[league_key][date_str][team_key] = val
    return out


def call_llm(prompt: str, model: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not available")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply NHL injury deltas via LLM (safe merge).")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--input", required=True, help="Path to injury/news text")
    ap.add_argument("--model", default="gpt-5.1", help="OpenAI model (default: gpt-5.1)")
    args = ap.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    prompt = f"LEAGUE: NHL\nDATE: {args.date}\nINJURY NEWS:\n{text}"
    raw = call_llm(prompt, model=args.model)

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"[error] failed to parse LLM JSON: {exc}")

    # Coerce the LLM output into the NHL league key.
    league_key = "NHL"
    if isinstance(parsed, dict) and league_key not in parsed:
        if "LEAGUE" in parsed and isinstance(parsed.get("LEAGUE"), dict):
            parsed = {league_key: parsed["LEAGUE"]}
        elif len(parsed) == 1:
            only_key = next(iter(parsed.keys()))
            if isinstance(parsed.get(only_key), dict):
                parsed = {league_key: parsed[only_key]}

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = AUDIT_DIR / f"injury_llm_{args.date}_nhl.json"
    audit_path.write_text(raw, encoding="utf-8")

    existing = load_existing()
    before = dict((existing.get(league_key) or {}).get(args.date) or {})
    merged = merge(existing, parsed)

    after = dict((merged.get(league_key) or {}).get(args.date) or {})
    snapshot_path = None
    if INJURY_PATH.exists():
        snapshot_path = snapshot_file(INJURY_PATH, SNAPSHOT_DIR)

    changes: dict[str, dict[str, float]] = {}
    teams = set(before.keys()) | set(after.keys())
    for team in sorted(teams):
        try:
            old = float(before.get(team, 0.0))
        except Exception:
            old = 0.0
        try:
            new = float(after.get(team, 0.0))
        except Exception:
            new = 0.0
        if old != new:
            changes[team] = {"old": old, "new": new}

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    change_path = SNAPSHOT_DIR / f"injury_llm_changes_{args.date}_nhl_{ts}.json"
    change_path.write_text(
        json.dumps(
            {
                "snapshot_ts": ts,
                "league": league_key,
                "date": args.date,
                "llm_model": args.model,
                "input_path": str(Path(args.input)),
                "audit_path": str(audit_path),
                "injury_adjustments_snapshot": str(snapshot_path) if snapshot_path else "",
                "changes": changes,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    INJURY_PATH.parent.mkdir(parents=True, exist_ok=True)
    INJURY_PATH.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[ok] merged NHL injury deltas into {INJURY_PATH}")
    print(f"[info] audit saved to {audit_path}")
    print(f"[info] changes saved to {change_path}")
    if snapshot_path:
        print(f"[info] prior file snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
