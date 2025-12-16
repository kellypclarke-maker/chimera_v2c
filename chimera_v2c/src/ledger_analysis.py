"""
Ledger-based EV vs Kalshi and calibration helpers (read-only).

This module provides pure functions to analyze per-game daily ledgers:
- Compute realized EV vs the Kalshi mid for each model.
- Compute accuracy and Brier scores vs actual outcomes.
- Group results by absolute edge buckets.

It is intentionally non-destructive: callers must treat daily ledgers in
`reports/daily_ledgers/` as append-only and read-only.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.ledger.outcomes import parse_home_win

LEDGER_DIR = Path("reports/daily_ledgers")


@dataclass
class GameRow:
    """Minimal parsed view of a ledger row used for analysis."""

    date: datetime
    league: str
    matchup: str
    kalshi_mid: Optional[float]
    probs: Dict[str, float]
    home_win: Optional[float]


@dataclass
class EvStats:
    bets: int = 0
    total_pnl: float = 0.0

    @property
    def avg_pnl(self) -> float:
        if self.bets == 0:
            return 0.0
        return self.total_pnl / self.bets


@dataclass
class BrierStats:
    n: int = 0
    sum_sq_error: float = 0.0

    @property
    def mean_brier(self) -> Optional[float]:
        if self.n == 0:
            return None
        return self.sum_sq_error / self.n


@dataclass
class ModelSummary:
    model: str
    ev: EvStats = field(default_factory=EvStats)
    brier: BrierStats = field(default_factory=BrierStats)


def _parse_home_win(actual_outcome: str) -> Optional[float]:
    return parse_home_win(actual_outcome)


def _parse_ledger_date_from_filename(path: Path) -> Optional[datetime]:
    """
    Filenames follow: YYYYMMDD_daily_game_ledger.csv
    """
    m = re.match(r"(\d{8})_daily_game_ledger\.csv", path.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d")
    except ValueError:
        return None


def load_games(
    daily_dir: Path = LEDGER_DIR,
    days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    league_filter: Optional[str] = None,
    models: Optional[List[str]] = None,
) -> List[GameRow]:
    """
    Load ledger rows into GameRow objects with basic parsing applied.

    Parameters
    ----------
    daily_dir:
        Directory containing *_daily_game_ledger.csv files.
    days:
        Optional rolling window: include only the last N ledger files
        by filename date. Ignored if start_date/end_date is provided.
    start_date, end_date:
        Optional inclusive date range in YYYY-MM-DD. If provided, these
        filter by the ledger filename date, not the CSV 'date' column.
    league_filter:
        Optional league code (e.g., 'nba', 'nhl', 'nfl'). Case-insensitive.
    models:
        Optional list of model column names to load probabilities for.
        If None, all probability-like columns found in the header will
        be included.
    """
    if not daily_dir.exists():
        return []

    ledger_paths = sorted(daily_dir.glob("*_daily_game_ledger.csv"))
    if not ledger_paths:
        return []

    # Determine which files to include based on date filters.
    dated_paths: List[Tuple[datetime, Path]] = []
    for p in ledger_paths:
        d = _parse_ledger_date_from_filename(p)
        if d is None:
            continue
        dated_paths.append((d, p))
    dated_paths.sort(key=lambda x: x[0])

    if start_date or end_date:
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else dated_paths[0][0]
            end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else dated_paths[-1][0]
        except ValueError:
            # If date parsing fails, fall back to all files.
            filtered_paths = [p for _, p in dated_paths]
        else:
            filtered_paths = [p for d, p in dated_paths if start_dt <= d <= end_dt]
    elif days is not None and days > 0:
        filtered_paths = [p for _, p in dated_paths[-days:]]
    else:
        filtered_paths = [p for _, p in dated_paths]

    league_filter_norm = league_filter.lower() if league_filter else None

    games: List[GameRow] = []

    for path in filtered_paths:
        ledger_date = _parse_ledger_date_from_filename(path)
        if ledger_date is None:
            continue
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Determine probability columns if not provided explicitly.
            header_models: List[str]
            if models is None:
                header_models = [
                    c
                    for c in reader.fieldnames or []
                    if c
                    and c not in {"date", "league", "matchup", "actual_outcome"}
                    and not c.endswith("_score")
                ]
            else:
                header_models = models
            for row in reader:
                league_raw = (row.get("league") or "").strip()
                if not league_raw:
                    continue
                league = league_raw.lower()
                if league_filter_norm and league != league_filter_norm:
                    continue

                outcome = row.get("actual_outcome") or ""
                home_win = _parse_home_win(outcome)

                kalshi_mid_str = row.get("kalshi_mid")
                kalshi_mid: Optional[float]
                if kalshi_mid_str is None or str(kalshi_mid_str).strip() == "":
                    kalshi_mid = None
                else:
                    try:
                        kalshi_mid = float(kalshi_mid_str)
                    except ValueError:
                        kalshi_mid = None

                probs: Dict[str, float] = {}
                for m in header_models:
                    val = row.get(m)
                    if val is None or str(val).strip() == "":
                        continue
                    try:
                        probs[m] = float(val)
                    except ValueError:
                        continue

                matchup = row.get("matchup") or ""
                games.append(
                    GameRow(
                        date=ledger_date,
                        league=league,
                        matchup=matchup,
                        kalshi_mid=kalshi_mid,
                        probs=probs,
                        home_win=home_win,
                    )
                )

    return games


def compute_ev_vs_kalshi(
    games: Iterable[GameRow],
    models: List[str],
) -> Dict[str, EvStats]:
    """
    Compute realized EV vs Kalshi mid for each model.

    For each model and game:
      - If p_model > p_kalshi, we treat this as buying home at price p_kalshi.
      - If p_model < p_kalshi, we treat this as being effectively long away
        (short home) at price p_kalshi.
      - Each game is sized as 1 unit notional; PnL is in contract units:
          * Long home win:  (1 - p_kalshi)
          * Long home loss: -p_kalshi
          * Long away win:  p_kalshi
          * Long away loss: -(1 - p_kalshi)
      - Games with push/unknown outcomes are skipped.
    """
    stats: Dict[str, EvStats] = {m: EvStats() for m in models}

    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if g.kalshi_mid is None:
            continue

        price = float(g.kalshi_mid)
        price = max(0.01, min(0.99, price))

        for m in models:
            if m not in g.probs:
                continue
            p_model = g.probs[m]
            if p_model == price:
                continue

            # Long home if model > market, else effectively long away.
            if p_model > price:
                pnl = (1.0 - price) if g.home_win == 1.0 else -price
            else:
                pnl = price if g.home_win == 0.0 else -(1.0 - price)

            s = stats[m]
            s.bets += 1
            s.total_pnl += pnl

    return stats


def compute_brier(
    games: Iterable[GameRow],
    models: List[str],
) -> Dict[str, BrierStats]:
    """
    Compute Brier score for each model over the given games.

    Brier is averaged over non-push outcomes with known probabilities.
    """
    stats: Dict[str, BrierStats] = {m: BrierStats() for m in models}

    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        for m in models:
            if m not in g.probs:
                continue
            p = g.probs[m]
            s = stats[m]
            s.n += 1
            s.sum_sq_error += (p - g.home_win) ** 2

    return stats


def edge_bucket(abs_edge: float, bucket_width: float = 0.025) -> str:
    """
    Map an absolute edge value |p_model - p_kalshi| to a human-readable bucket.
    Example with bucket_width=0.025:
      [0.000,0.025), [0.025,0.050), ..., [0.100,inf)
    """
    if abs_edge < 0:
        abs_edge = -abs_edge
    # Cap "large" edges in an open-ended bucket.
    max_bucket = 0.10
    if abs_edge >= max_bucket:
        return f"[{max_bucket:.3f},inf)"
    lo = (abs_edge // bucket_width) * bucket_width
    hi = lo + bucket_width
    return f"[{lo:.3f},{hi:.3f})"


def compute_bucketed_ev_vs_kalshi(
    games: Iterable[GameRow],
    models: List[str],
    bucket_width: float = 0.025,
) -> Dict[str, Dict[str, EvStats]]:
    """
    Compute EV vs Kalshi per model per absolute edge bucket.

    Returns:
      model -> bucket_label -> EvStats
    """
    # Initialize nested dicts on demand.
    stats: Dict[str, Dict[str, EvStats]] = {}

    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if g.kalshi_mid is None:
            continue

        price = float(g.kalshi_mid)
        price = max(0.01, min(0.99, price))

        for m in models:
            if m not in g.probs:
                continue
            p_model = g.probs[m]
            if p_model == price:
                continue

            abs_edge = abs(p_model - price)
            bucket = edge_bucket(abs_edge, bucket_width=bucket_width)

            if m not in stats:
                stats[m] = {}
            if bucket not in stats[m]:
                stats[m][bucket] = EvStats()

            if p_model > price:
                pnl = (1.0 - price) if g.home_win == 1.0 else -price
            else:
                pnl = price if g.home_win == 0.0 else -(1.0 - price)

            s = stats[m][bucket]
            s.bets += 1
            s.total_pnl += pnl

    return stats
