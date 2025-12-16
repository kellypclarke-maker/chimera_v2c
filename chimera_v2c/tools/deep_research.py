"""
Deep research harness: build a dossier, prompt an LLM (stub), parse JSON, and log outputs.

Usage:
  PYTHONPATH=. python chimera_v2c/tools/deep_research.py --league nba --date YYYY-MM-DD --game AWAY@HOME --model gpt
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from chimera_v2c.src.doctrine import DoctrineConfig, doctrine_decide_trade
from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.tools.build_dossier import build_dossier
from chimera_v2c.lib.env_loader import load_env_from_env_list

RAW_DIR = Path("specialist_reports/raw")
LOG_DIR = Path("reports/specialist_performance")
LOG_PATH = LOG_DIR / "model_comparison.csv"
HEADER = [
    "date",
    "league",
    "game",
    "model",
    "p_true_home",
    "p_true_away",
    "confidence",
    "thesis_ptcs",
    "data_ptcs",
    "llm_allowed",
    "llm_stake_fraction",
    "llm_reason",
    "raw_path",
    "market_home_closing_p",
    "market_away_closing_p",
    "clv_home",
    "clv_away",
    "gpt_home_p",
    "gpt_away_p",
    "gpt_confidence",
    "gpt_doctrine_allowed",
    "gpt_stake_fraction",
    "gpt_doctrine_reason",
    "gemini_home_p",
    "gemini_away_p",
    "gemini_confidence",
    "gemini_doctrine_allowed",
    "gemini_stake_fraction",
    "gemini_doctrine_reason",
]

SCHEMA_HINT = """
Respond with ONLY JSON matching:
{
  "version": 1,
  "league": "<nba|nhl>",
  "game": "AWAY@HOME",
  "p_true_home": 0.0-1.0,
  "p_true_away": 0.0-1.0,
  "confidence": 0.0-1.0,
  "thesis_ptcs": 0.0-1.0,
  "data_ptcs": 0.0-1.0,
  "per_player_impact": [
    {
      "player": "Name",
      "team": "ABC",
      "status": "out|doubtful|questionable|probable|active",
      "role": "primary_creator|primary_scorer|secondary|guard|wing|big|rotation",
      "impact_score": 1-10,
      "reason": "short text"
    }
  ],
  "team_rating_shifts": { "HOME": 0.0, "AWAY": 0.0 },
  "thesis": "2-5 sentences"
}
Rules: p_true_home + p_true_away must be ~1 (+/-0.02). 0 <= confidence <= 1. impact_score 1-10. Output JSON ONLY.
"""


def call_llm_stub(prompt: str) -> str:
    # Stubbed LLM response; replace with real call later.
    return json.dumps(
        {
            "version": 1,
            "league": "nba",
            "game": "",
            "p_true_home": 0.55,
            "p_true_away": 0.45,
            "confidence": 0.7,
            "per_player_impact": [],
            "team_rating_shifts": {},
            "thesis": "stubbed response",
        }
    )


def call_llm_gpt(prompt: str, model: str = "gpt-5.1") -> str:
    if OpenAI is None:
        raise RuntimeError("openai package not installed")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a sports betting Specialist. Respond with JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def normalize_probabilities(p_home: float, p_away: float) -> Tuple[float, float]:
    total = p_home + p_away
    if total <= 0:
        return p_home, p_away
    if abs(total - 1.0) <= 0.02:
        # keep as-is
        return p_home, p_away
    return p_home / total, p_away / total


def strip_code_fences(text: str) -> str:
    if not text:
        return text
    lines = text.strip().splitlines()
    if not lines:
        return text
    if lines[0].strip().startswith("```"):
        # drop first fence
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _append_to_file(path: Path, row: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    existing_header: List[str] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]
        if lines:
            existing_header = lines[0].split(",")
            for line in lines[1:]:
                if not line:
                    continue
                parts = line.split(",")
                rows.append({k: parts[i] if i < len(parts) else "" for i, k in enumerate(existing_header)})
    rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(HEADER) + "\n")
        for r in rows:
            values = [str(r.get(k, "")) for k in HEADER]
            f.write(",".join(values) + "\n")


def append_log(row: Dict[str, Any]) -> None:
    date_part = row.get("date", "")
    league = row.get("league", "")
    _append_to_file(LOG_PATH, row)
    if date_part:
        _append_to_file(LOG_DIR / f"model_comparison_{date_part}.csv", row)
    if date_part and league:
        _append_to_file(LOG_DIR / f"model_comparison_{league}_{date_part}.csv", row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep research harness (LLM stub).")
    ap.add_argument("--league", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--model", default="gpt", choices=["stub", "gpt", "gemini"])
    ap.add_argument("--gpt-model", default="gpt-5.1", help="OpenAI model name")
    args = ap.parse_args()

    # Load env vars (e.g., OPENAI_API_KEY) from config/env.list
    load_env_from_env_list()

    dossier = build_dossier(args.league, args.date, args.game)
    prompt = (
        "You are the sports betting Specialist. Given this dossier, estimate calibrated true win probabilities for home and away, and return ONLY JSON.\n"
        f"Schema:\n{SCHEMA_HINT}\nDossier:\n{json.dumps(dossier, indent=2)}"
    )
    try:
        if args.model == "stub":
            raw_response = call_llm_stub(prompt)
        elif args.model == "gpt":
            raw_response = call_llm_gpt(prompt, model=args.gpt_model)
        else:
            raw_response = call_llm_stub(prompt)  # gemini placeholder for now
    except Exception as exc:
        raw_response = ""
        print(f"[error] LLM call failed: {exc}")

    raw_dir = RAW_DIR / args.league / args.date.replace("-", "")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / f"{args.game}_{args.model}.txt"
    raw_file.write_text(raw_response, encoding="utf-8")

    cleaned = strip_code_fences(raw_response)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {}

    p_home = parsed.get("p_true_home")
    p_away = parsed.get("p_true_away")
    conf = parsed.get("confidence")
    thesis_ptcs = parsed.get("thesis_ptcs")
    data_ptcs = parsed.get("data_ptcs")
    doc_allowed = ""
    doc_stake = ""
    doc_reason = ""
    reason = ""
    reason = ""
    # Invariant checks
    try:
        if p_home is None or p_away is None or conf is None:
            raise ValueError("missing fields")
        if not (0 <= p_home <= 1 and 0 <= p_away <= 1):
            raise ValueError("prob out of range")
        p_home, p_away = normalize_probabilities(p_home, p_away)
        if not (0 <= conf <= 1):
            raise ValueError("confidence out of range")
        # Doctrine log-only decision (home vs market)
        market_home = (dossier.get("market") or {}).get("home_prob")
        cfg = V2CConfig.load("chimera_v2c/config/defaults.yaml")
        doc_cfg = DoctrineConfig(
            max_fraction=cfg.max_fraction,
            target_spread_bp=cfg.target_spread_bp,
            require_confluence=cfg.require_confluence,
            enable_bucket_guardrails=bool(cfg.doctrine_cfg.get("enable_bucket_guardrails", False)),
            bucket_guardrails_path=cfg.doctrine_cfg.get("bucket_guardrails_path", "reports/roi_by_bucket.csv"),
            league_min_samples=cfg.doctrine_cfg.get("league_min_samples"),
            paper_mode_enforce=bool(cfg.doctrine_cfg.get("paper_mode_enforce", True)),
            league=cfg.league,
            negative_roi_buckets=cfg.doctrine_cfg.get("negative_roi_buckets"),
        )
        stake_val, _, doc_reason = doctrine_decide_trade(
            p_model=p_home,
            p_market=market_home,
            cfg=doc_cfg,
            used_fraction=0.0,
            daily_cap=getattr(cfg, "daily_max_fraction", cfg.max_fraction * 8),
            internal_prob=p_home,
            market_signal=market_home,
        )
        doc_allowed = bool(stake_val) and stake_val is not None
        doc_stake = stake_val if stake_val is not None else ""
        if doc_allowed and stake_val is not None:
            doc_stake = round(stake_val, 6)
    except Exception as exc:
        doc_reason = doc_reason or f"invalid_output:{exc}"
        doc_allowed = ""
        doc_stake = ""

    row = {
        "date": args.date,
        "league": args.league,
        "game": args.game,
        "model": args.model,
        "p_true_home": p_home,
        "p_true_away": p_away,
        "confidence": conf,
        "thesis_ptcs": thesis_ptcs if thesis_ptcs is not None else "",
        "data_ptcs": data_ptcs if data_ptcs is not None else "",
        "llm_allowed": doc_allowed,
        "llm_stake_fraction": doc_stake,
        "llm_reason": reason or doc_reason,
        "raw_path": raw_file,
        "market_home_closing_p": "",
        "market_away_closing_p": "",
        "clv_home": "",
        "clv_away": "",
        "gpt_home_p": p_home if args.model == "gpt" else "",
        "gpt_away_p": p_away if args.model == "gpt" else "",
        "gpt_confidence": conf if args.model == "gpt" else "",
        "gpt_doctrine_allowed": doc_allowed if args.model == "gpt" else "",
        "gpt_stake_fraction": doc_stake if args.model == "gpt" else "",
        "gpt_doctrine_reason": reason or doc_reason if args.model == "gpt" else "",
        "gemini_home_p": p_home if args.model == "gemini" else "",
        "gemini_away_p": p_away if args.model == "gemini" else "",
        "gemini_confidence": conf if args.model == "gemini" else "",
        "gemini_doctrine_allowed": doc_allowed if args.model == "gemini" else "",
        "gemini_stake_fraction": doc_stake if args.model == "gemini" else "",
        "gemini_doctrine_reason": reason or doc_reason if args.model == "gemini" else "",
    }
    append_log(row)
    print(f"[info] saved raw to {raw_file}")
    print(f"[info] appended log to {LOG_PATH}")


if __name__ == "__main__":
    main()
