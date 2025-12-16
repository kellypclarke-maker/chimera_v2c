"""
LLM injury delta applier (safe merge) for injury_adjustments.json.

Inputs:
- Freeform injury/news text via --input <file>
- Target league/date
- Model (default gpt-5.1)

Behavior:
- Prompts the LLM to emit JSON: {"LEAGUE": {"YYYY-MM-DD": {"TEAM": -25.0, ...}}}
- Merges into chimera_v2c/data/injury_adjustments.json (append/overwrite per team/date only)
- Writes a copy of the LLM response to reports/specialist_reports/raw/injury_llm_<date>_<league>.json for audit.

Usage:
  PYTHONPATH=. OPENAI_API_KEY=... python chimera_v2c/tools/apply_llm_injuries_v2.py --league nba --date 2025-12-08 --input injuries.txt
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

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
You are an injury-to-Elo estimator. Return strictly JSON with per-team Elo deltas (negative = penalty).
Schema:
{
  "LEAGUE": {
    "YYYY-MM-DD": {
      "TEAM": -35.0,
      ...
    }
  }
}
Rules:
- Use league/team codes (NBA/NHL/NFL standard 2-4 letter).
- Ignore teams not mentioned.
- If injury impact is uncertain or minor, use 0.0 rather than guessing.
- Penalties are bounded: clamp to [-40, 0]. Never exceed -40; never return positive bonuses.
- Consider role/impact: star/out ~ -20 to -40; starter/rotation ~ -8 to -20; minor/bench ~ -1 to -8.
- Do not include explanations; JSON only.
"""


def load_existing() -> Dict:
    if not INJURY_PATH.exists():
        return {}
    try:
        return json.loads(INJURY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

def _normalize_team(team: str, league: str) -> str:
    norm = normalize_team_code(team, league.lower())
    return (norm or team or "").strip().upper()


def merge(existing: Dict, new_data: Dict) -> Dict:
    out = existing.copy()
    for league, dates in new_data.items():
        league_key = str(league or "").strip().upper()
        if not league_key:
            continue
        out.setdefault(league_key, {})
        for date_str, teams in dates.items():
            out[league_key].setdefault(date_str, {})
            for team, delta in teams.items():
                team_key = _normalize_team(str(team), league_key)
                if not team_key:
                    continue
                try:
                    val = float(delta)
                except Exception:
                    continue
                # Clamp to bounds for safety.
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
    ap = argparse.ArgumentParser(description="Apply LLM-derived injury deltas to injury_adjustments.json (safe merge).")
    ap.add_argument("--league", required=True, help="League (nba|nhl|nfl)")
    ap.add_argument("--date", required=True, help="Target date YYYY-MM-DD")
    ap.add_argument("--input", required=True, help="Path to injury/news text")
    ap.add_argument("--model", default="gpt-5.1", help="OpenAI model (default: gpt-5.1)")
    args = ap.parse_args()

    league_key = args.league.upper().strip()
    if not league_key:
        raise SystemExit("[error] empty --league")

    text = Path(args.input).read_text(encoding="utf-8")
    prompt = f"LEAGUE: {league_key}\nDATE: {args.date}\nINJURY NEWS:\n{text}"
    raw = call_llm(prompt, model=args.model)

    try:
        parsed = json.loads(raw)
    except Exception as exc:
        raise SystemExit(f"[error] failed to parse LLM JSON: {exc}")

    # Coerce the LLM output into the requested league key.
    if isinstance(parsed, dict) and league_key not in parsed:
        if "LEAGUE" in parsed and isinstance(parsed.get("LEAGUE"), dict):
            parsed = {league_key: parsed["LEAGUE"]}
        elif len(parsed) == 1:
            only_key = next(iter(parsed.keys()))
            if isinstance(parsed.get(only_key), dict):
                parsed = {league_key: parsed[only_key]}

    # Write audit
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = AUDIT_DIR / f"injury_llm_{args.date}_{args.league}.json"
    audit_path.write_text(raw, encoding="utf-8")

    existing = load_existing()
    before = dict((existing.get(league_key) or {}).get(args.date) or {})
    merged = merge(existing, parsed)
    after = dict((merged.get(league_key) or {}).get(args.date) or {})

    # Snapshot the full file before writing.
    snapshot_path = None
    if INJURY_PATH.exists():
        snapshot_path = snapshot_file(INJURY_PATH, SNAPSHOT_DIR)

    # Write a small “what changed” memo for this date/league.
    changes: Dict[str, Dict[str, float]] = {}
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
    change_path = SNAPSHOT_DIR / f"injury_llm_changes_{args.date}_{args.league}_{ts}.json"
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
    print(f"[ok] merged injury deltas into {INJURY_PATH}")
    print(f"[info] audit saved to {audit_path}")
    print(f"[info] changes saved to {change_path}")
    if snapshot_path:
        print(f"[info] prior file snapshot: {snapshot_path}")


if __name__ == "__main__":
    main()
