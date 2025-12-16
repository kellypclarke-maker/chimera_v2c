"""Helpers for parsing and formatting ledger `actual_outcome` cells.

Canonical `actual_outcome` format (when final):
  "<AWAY> <away_score>-<home_score> <HOME>"

Examples:
  "CHA 119-111 CLE"
  "EDM 2-3 MTL"

Outcomes may be blank while a game is not final.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple


_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)")
_TAG_RE = re.compile(r"\((home|away|push)\)", re.IGNORECASE)


def parse_scores(actual_outcome: object) -> Optional[Tuple[float, float]]:
    """Return (away_score, home_score) if scores are present."""
    if not isinstance(actual_outcome, str):
        return None
    s = actual_outcome.strip()
    if not s or s.lower() in {"nan", "none", "nr"}:
        return None
    m = _SCORE_RE.search(s)
    if not m:
        return None
    try:
        away_score = float(m.group(1))
        home_score = float(m.group(2))
    except ValueError:
        return None
    return away_score, home_score


def parse_home_win(actual_outcome: object) -> Optional[float]:
    """
    Return:
      - 1.0 for home win
      - 0.0 for away win
      - 0.5 for tie/push
      - None if unknown / not final
    """
    if not isinstance(actual_outcome, str):
        return None
    s = actual_outcome.strip()
    if not s or s.lower() in {"nan", "none", "nr"}:
        return None

    scores = parse_scores(s)
    if scores is not None:
        away_score, home_score = scores
        if home_score > away_score:
            return 1.0
        if away_score > home_score:
            return 0.0
        return 0.5

    sl = s.lower()
    if sl in {"home", "home_win"}:
        return 1.0
    if sl in {"away", "away_win"}:
        return 0.0
    if sl in {"push", "tie"}:
        return 0.5

    m = _TAG_RE.search(sl)
    if not m:
        return None
    tag = m.group(1).lower()
    if tag == "home":
        return 1.0
    if tag == "away":
        return 0.0
    if tag == "push":
        return 0.5
    return None


def _score_to_str(score: float) -> str:
    if abs(score - round(score)) < 1e-9:
        return str(int(round(score)))
    return f"{score:.1f}".rstrip("0").rstrip(".")


def format_final_score(away_team: str, home_team: str, away_score: float, home_score: float) -> str:
    return f"{away_team} {_score_to_str(away_score)}-{_score_to_str(home_score)} {home_team}"


def is_placeholder_outcome(actual_outcome: object) -> bool:
    """
    Heuristic for clearly-non-final placeholders that should not live in canon.

    Examples:
      - "0.0-0.0 (push)"
      - "pending"
      - "home_win"/"away_win"
      - "NR"
    """
    if not isinstance(actual_outcome, str):
        return False
    s = actual_outcome.strip()
    if not s:
        return False
    sl = s.lower()
    if sl == "nr":
        return True
    if "pending" in sl:
        return True
    if sl in {"home_win", "away_win", "push"}:
        return True
    if "push" in sl and "0" in sl:
        # Common placeholder pattern: 0-0 (push)
        return True
    # If we have scores but both are zero, treat as placeholder for these leagues.
    scores = parse_scores(sl)
    if scores is not None:
        away_score, home_score = scores
        if abs(away_score) < 1e-9 and abs(home_score) < 1e-9:
            return True
    return False

