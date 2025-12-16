"""
LLM-assisted NHL factor weight tuning (log-only).

Reads: reports/master_ledger/master_game_ledger.csv
Prompts an LLM to suggest factor weights within bounds (sum=1.0) to minimize Brier on a slice.
Does NOT write configs; prints YAML snippet to stdout.

Usage (from repo root):
  PYTHONPATH=. python chimera_v2c/tools/llm_tune_nhl_weights.py --days 30 --model gpt-5.1
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from chimera_v2c.src.ledger.outcomes import parse_home_win

LEDGER_PATH = Path("reports/master_ledger/master_game_ledger.csv")

SYSTEM_PROMPT = """
You are an NHL probability calibration assistant. Suggest factor weights to minimize Brier score.
Constraints:
- Keys: xgf_weight, hdcf_weight, pp_weight, pk_weight, goalie_weight, oi_sh_weight
- Bounds: xgf 0.30-0.40; hdcf 0.15-0.25; pp 0.15-0.25; pk 0.10-0.20; goalie 0.05-0.15; oi_sh 0.0-0.10
- Sum must equal 1.0 (within 1e-3).
Input provides rows: p_pred (current), outcome (0/1), current_weights, and basic stats.
Respond ONLY with JSON:
{"weights": {"xgf_weight": 0.xx, "hdcf_weight": 0.xx, "pp_weight": 0.xx, "pk_weight": 0.xx, "goalie_weight": 0.xx, "oi_sh_weight": 0.xx}}
"""


def load_slice(days: int) -> List[Tuple[float, int]]:
    if not LEDGER_PATH.exists():
        raise SystemExit(f"[error] ledger missing: {LEDGER_PATH}")
    df = pd.read_csv(LEDGER_PATH)
    df = df[df["league"].str.lower() == "nhl"]
    if "v2c" not in df.columns:
        raise SystemExit("[error] ledger missing v2c column")
    if "actual_outcome" not in df.columns:
        raise SystemExit("[error] ledger missing actual_outcome column")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cutoff = datetime.utcnow().date() - timedelta(days=days)
        df = df[df["date"].dt.date >= cutoff]
    preds = df["v2c"]

    def resolve_y(outcome: object) -> int | None:
        hw = parse_home_win(outcome)
        if hw == 1.0:
            return 1
        if hw == 0.0:
            return 0
        return None

    ys = df["actual_outcome"].apply(resolve_y)
    rows = []
    for p, y in zip(preds, ys):
        try:
            p_float = float(p)
        except Exception:
            continue
        if y in (0, 1):
            rows.append((p_float, int(y)))
    return rows


def build_user_prompt(pairs: List[Tuple[float, int]], current_weights: dict) -> str:
    preview = "\n".join([f"{p:.3f},{y}" for p, y in pairs[:100]])
    return (
        "Recent NHL outcomes (p_pred, outcome):\n"
        f"{preview}\n"
        f"Current weights: {json.dumps(current_weights)}\n"
        "Suggest new weights within bounds to minimize Brier. Return JSON only."
    )


def call_llm(prompt: str, model: str) -> str:
    from openai import OpenAI  # local import

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing")
    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM tuning helper for NHL factor weights (log-only).")
    ap.add_argument("--days", type=int, default=30, help="Use last N days (default 30).")
    ap.add_argument("--model", default="gpt-5.1", help="OpenAI model name.")
    ap.add_argument(
        "--current",
        default='{"xgf_weight":0.35,"hdcf_weight":0.20,"pp_weight":0.20,"pk_weight":0.15,"goalie_weight":0.05,"oi_sh_weight":0.05}',
        help="Current weights JSON string.",
    )
    args = ap.parse_args()

    pairs = load_slice(args.days)
    if len(pairs) < 20:
        raise SystemExit(f"[error] only {len(pairs)} samples; not enough to tune")

    try:
        current_weights = json.loads(args.current)
    except Exception as exc:
        raise SystemExit(f"[error] failed to parse current weights: {exc}")

    prompt = build_user_prompt(pairs, current_weights)
    try:
        raw = call_llm(prompt, args.model)
    except Exception as exc:
        raise SystemExit(f"[error] LLM call failed: {exc}")

    print(raw)
    print("\n# Apply manually after review (nhl_defaults.yaml -> four_factors)")


if __name__ == "__main__":
    main()
