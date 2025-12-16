"""
Scheme D analysis (read-only): I/J-gated home-favorite fades vs Kalshi mid.

Scheme D (see docs/EV_ANALYSIS.md) focuses on the asymmetric regime where:
  - Kalshi favors HOME (kalshi_mid >= 0.5), and
  - one or more models think home is rich by >= edge_threshold (default 5c),
    so we fade home (bet away) at the mid.

This tool:
  1) Computes per-(league,model) realized EV for Rule I vs Rule J separately.
  2) Marks (league,model,rule) buckets as "allowed" when avg_pnl >= ev_threshold
     and bets >= min_bets.
  3) Backtests Scheme D consensus sizing (1/3/5 units by #strong models).

It never writes to daily ledgers. It may write derived CSVs under
`reports/ev_rulebooks/` for operator review.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games


DEFAULT_MODELS = ["v2c", "gemini", "grok", "gpt"]


@dataclass
class RuleStats:
    bets: int = 0
    wins: int = 0
    total_pnl: float = 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.bets if self.bets else 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.bets if self.bets else 0.0


@dataclass(frozen=True)
class RuleKey:
    league: str
    model: str
    rule: str  # "I" or "J"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Scheme D analysis from daily ledgers (read-only on ledgers)."
    )
    ap.add_argument(
        "--days",
        type=int,
        default=30,
        help="Include only the most recent N ledger days by filename date (default: 30). "
        "Use 0 or a negative value to include all days.",
    )
    ap.add_argument("--start-date", help="Optional start date (YYYY-MM-DD). If set, overrides --days.")
    ap.add_argument("--end-date", help="Optional end date (YYYY-MM-DD). If set, overrides --days.")
    ap.add_argument("--league", help="Optional league filter (nba|nhl|nfl).")
    ap.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model columns to include (default: v2c gemini grok gpt).",
    )
    ap.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="Edge threshold for 'home rich' vs Kalshi mid (default: 0.05).",
    )
    ap.add_argument(
        "--ev-threshold",
        type=float,
        default=0.10,
        help="Minimum avg_pnl_per_bet for a (league,model,rule) bucket to be allowed (default: 0.10).",
    )
    ap.add_argument(
        "--min-bets",
        type=int,
        default=1,
        help="Minimum bets in a (league,model,rule) bucket to be eligible for gating (default: 1).",
    )
    ap.add_argument(
        "--out-dir",
        default="reports/ev_rulebooks",
        help="Directory to write derived CSV artifacts (default: reports/ev_rulebooks).",
    )
    ap.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write CSV artifacts; print-only.",
    )
    return ap.parse_args()


def _pnl_for_away_fade(p_mid: float, home_win: float) -> float:
    """
    Realized PnL (1 unit) for betting AWAY at price p_mid, given outcome.
      - If home loses (away wins): +p_mid
      - If home wins:            -(1 - p_mid)
    """
    p_mid = max(0.01, min(0.99, float(p_mid)))
    if home_win == 0.0:
        return p_mid
    return -(1.0 - p_mid)


def iter_graded_games(games: Iterable[GameRow]) -> Iterable[GameRow]:
    for g in games:
        if g.home_win is None or g.home_win == 0.5:
            continue
        if g.kalshi_mid is None:
            continue
        yield g


def classify_rule_i_j(
    p_mid: float,
    p_model: float,
    edge_threshold: float,
) -> Optional[str]:
    """
    Return "I" or "J" when Scheme D fires; otherwise None.

    Requires:
      - home-favorite market: p_mid >= 0.5
      - home rich by >= edge_threshold: p_model - p_mid <= -edge_threshold

    Rule I: p_model < 0.5 (model prefers away)
    Rule J: p_model >= 0.5 (model still prefers home but is >= threshold less bullish)
    """
    if p_mid < 0.5:
        return None
    if (p_model - p_mid) > -edge_threshold:
        return None
    if p_model < 0.5:
        return "I"
    return "J"


def compute_rule_stats(
    games: Iterable[GameRow],
    models: List[str],
    edge_threshold: float,
) -> Dict[RuleKey, RuleStats]:
    stats: Dict[RuleKey, RuleStats] = {}

    for g in iter_graded_games(games):
        p_mid = float(g.kalshi_mid)
        for model in models:
            p_model = g.probs.get(model)
            if p_model is None:
                continue
            rule = classify_rule_i_j(p_mid=p_mid, p_model=p_model, edge_threshold=edge_threshold)
            if rule is None:
                continue
            key = RuleKey(league=g.league, model=model, rule=rule)
            s = stats.setdefault(key, RuleStats())
            s.bets += 1
            pnl = _pnl_for_away_fade(p_mid=p_mid, home_win=float(g.home_win))
            s.total_pnl += pnl
            if g.home_win == 0.0:
                s.wins += 1

    return stats


def compute_allowed_buckets(
    rule_stats: Dict[RuleKey, RuleStats],
    ev_threshold: float,
    min_bets: int,
) -> Dict[RuleKey, bool]:
    allowed: Dict[RuleKey, bool] = {}
    for key, s in rule_stats.items():
        ok = s.bets >= min_bets and s.avg_pnl >= ev_threshold
        allowed[key] = ok
    return allowed


@dataclass
class ConsensusTotals:
    units: int = 0
    total_pnl: float = 0.0
    tier_units: Dict[int, int] = None  # tier -> units
    tier_pnl: Dict[int, float] = None  # tier -> pnl

    def __post_init__(self) -> None:
        if self.tier_units is None:
            self.tier_units = {1: 0, 3: 0, 5: 0}
        if self.tier_pnl is None:
            self.tier_pnl = {1: 0.0, 3: 0.0, 5: 0.0}

    @property
    def avg_pnl_per_unit(self) -> float:
        return self.total_pnl / self.units if self.units else 0.0


def scheme_d_units(n_models: int) -> int:
    if n_models <= 0:
        return 0
    if n_models == 1:
        return 1
    if n_models == 2:
        return 3
    return 5


def backtest_scheme_d_consensus(
    games: Iterable[GameRow],
    models: List[str],
    allowed: Dict[RuleKey, bool],
    edge_threshold: float,
) -> Tuple[Dict[str, ConsensusTotals], Dict[str, ConsensusTotals]]:
    """
    Returns (consensus_135, baseline_1_per_signal) per league plus overall.
    """
    consensus: Dict[str, ConsensusTotals] = {}
    baseline: Dict[str, ConsensusTotals] = {}

    def get_totals(d: Dict[str, ConsensusTotals], league: str) -> ConsensusTotals:
        if league not in d:
            d[league] = ConsensusTotals()
        if "overall" not in d:
            d["overall"] = ConsensusTotals()
        return d[league]

    for g in iter_graded_games(games):
        league = g.league
        p_mid = float(g.kalshi_mid)
        pnl_per_unit = _pnl_for_away_fade(p_mid=p_mid, home_win=float(g.home_win))

        fired_models: List[str] = []
        for model in models:
            p_model = g.probs.get(model)
            if p_model is None:
                continue
            rule = classify_rule_i_j(p_mid=p_mid, p_model=p_model, edge_threshold=edge_threshold)
            if rule is None:
                continue
            key = RuleKey(league=league, model=model, rule=rule)
            if not allowed.get(key, False):
                continue
            fired_models.append(model)

        n = len(fired_models)

        # Baseline: 1 unit per allowed signal (per model).
        if n:
            for _ in fired_models:
                for bucket in (league, "overall"):
                    t = get_totals(baseline, bucket)
                    t.units += 1
                    t.total_pnl += pnl_per_unit
            # Track tiers for baseline using the same 1/3/5 bins by n.
            tier = scheme_d_units(n)
            for bucket in (league, "overall"):
                t = get_totals(baseline, bucket)
                t.tier_units[tier] += n
                t.tier_pnl[tier] += pnl_per_unit * n

        # Consensus 1/3/5: 1 bet per game sized by n.
        units = scheme_d_units(n)
        if units:
            for bucket in (league, "overall"):
                t = get_totals(consensus, bucket)
                t.units += units
                t.total_pnl += pnl_per_unit * units
                t.tier_units[units] += units
                t.tier_pnl[units] += pnl_per_unit * units

    return consensus, baseline


def _fmt_float(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def print_rule_stats_table(
    rule_stats: Dict[RuleKey, RuleStats],
    allowed: Dict[RuleKey, bool],
    models: List[str],
    leagues: List[str],
) -> None:
    print("\n=== Scheme D Rule I/J bucket stats (bet away at mid) ===")
    print(f"{'league':5s} {'model':10s} {'rule':4s} {'bets':>6s} {'avg_pnl':>8s} {'win_rate':>9s} {'allowed':>8s}")
    for league in leagues:
        for model in models:
            for rule in ("I", "J"):
                key = RuleKey(league=league, model=model, rule=rule)
                s = rule_stats.get(key)
                if not s or s.bets == 0:
                    continue
                is_allowed = allowed.get(key, False)
                print(
                    f"{league:5s} {model:10s} {rule:4s} {s.bets:6d} "
                    f"{_fmt_float(s.avg_pnl, 3):>8s} {_fmt_float(s.win_rate, 3):>9s} {str(is_allowed):>8s}"
                )


def print_consensus_summary(label: str, totals: Dict[str, ConsensusTotals], leagues: List[str]) -> None:
    print(f"\n=== {label} ===")
    print(f"{'league':7s} {'units':>7s} {'total_pnl':>10s} {'avg_pnl/unit':>12s} {'tier1':>7s} {'tier3':>7s} {'tier5':>7s}")
    for league in leagues + ["overall"]:
        t = totals.get(league)
        if not t or t.units == 0:
            continue
        print(
            f"{league:7s} {t.units:7d} {t.total_pnl:10.3f} {t.avg_pnl_per_unit:12.3f} "
            f"{t.tier_units[1]:7d} {t.tier_units[3]:7d} {t.tier_units[5]:7d}"
        )


def write_rule_stats_csv(
    out_path: Path,
    rule_stats: Dict[RuleKey, RuleStats],
    allowed: Dict[RuleKey, bool],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "league",
        "model",
        "rule",
        "bets",
        "wins",
        "win_rate",
        "total_pnl",
        "avg_pnl",
        "allowed",
    ]
    rows = []
    for key in sorted(rule_stats.keys(), key=lambda k: (k.league, k.model, k.rule)):
        s = rule_stats[key]
        rows.append(
            {
                "league": key.league,
                "model": key.model,
                "rule": key.rule,
                "bets": s.bets,
                "wins": s.wins,
                "win_rate": _fmt_float(s.win_rate, 6),
                "total_pnl": _fmt_float(s.total_pnl, 6),
                "avg_pnl": _fmt_float(s.avg_pnl, 6),
                "allowed": str(bool(allowed.get(key, False))),
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_consensus_csv(
    out_path: Path,
    totals: Dict[str, ConsensusTotals],
    start_date: Optional[str],
    end_date: Optional[str],
    edge_threshold: float,
    ev_threshold: float,
    min_bets: int,
    label: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "league",
        "start_date",
        "end_date",
        "edge_threshold",
        "ev_threshold",
        "min_bets",
        "units",
        "total_pnl",
        "avg_pnl_per_unit",
        "tier1_units",
        "tier1_pnl",
        "tier3_units",
        "tier3_pnl",
        "tier5_units",
        "tier5_pnl",
    ]
    leagues = sorted([k for k in totals.keys() if k != "overall"]) + ["overall"]
    rows = []
    for league in leagues:
        t = totals.get(league)
        if not t or t.units == 0:
            continue
        rows.append(
            {
                "label": label,
                "league": league,
                "start_date": start_date or "",
                "end_date": end_date or "",
                "edge_threshold": _fmt_float(edge_threshold, 3),
                "ev_threshold": _fmt_float(ev_threshold, 3),
                "min_bets": min_bets,
                "units": t.units,
                "total_pnl": _fmt_float(t.total_pnl, 6),
                "avg_pnl_per_unit": _fmt_float(t.avg_pnl_per_unit, 6),
                "tier1_units": t.tier_units[1],
                "tier1_pnl": _fmt_float(t.tier_pnl[1], 6),
                "tier3_units": t.tier_units[3],
                "tier3_pnl": _fmt_float(t.tier_pnl[3], 6),
                "tier5_units": t.tier_units[5],
                "tier5_pnl": _fmt_float(t.tier_pnl[5], 6),
            }
        )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rulebook_a_i_j_csv(
    out_path: Path,
    leagues: List[str],
    models: List[str],
    allowed: Dict[RuleKey, bool],
    start_date: Optional[str],
    end_date: Optional[str],
    edge_threshold: float,
    ev_threshold: float,
    min_bets: int,
) -> None:
    """
    Write a compact (league,model) rulebook mirroring the repo's historical
    `rulebook_A_I_J_by_model_league.csv` convention.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["league", "model", "allowed_rules", "notes"]
    rows: List[Dict[str, str]] = []
    for league in leagues:
        for model in models:
            rules = []
            if allowed.get(RuleKey(league=league, model=model, rule="I"), False):
                rules.append("I")
            if allowed.get(RuleKey(league=league, model=model, rule="J"), False):
                rules.append("J")
            allowed_rules = ""
            if rules:
                allowed_rules = "A," + ",".join(rules)
            notes = (
                "Derived from daily ledgers"
                f" ({start_date or '...'}..{end_date or '...'})"
                f"; edge_threshold={edge_threshold:.3f}"
                f"; allow I/J when bets>={min_bets} and avg_pnl>={ev_threshold:.3f}."
            )
            rows.append(
                {
                    "league": league,
                    "model": model,
                    "allowed_rules": allowed_rules,
                    "notes": notes,
                }
            )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    if not LEDGER_DIR.exists():
        raise SystemExit(f"[error] daily ledger directory not found: {LEDGER_DIR}")

    # Use start/end-date if provided; otherwise fall back to days-based window.
    days: Optional[int]
    if args.start_date or args.end_date:
        days = None
    else:
        days = args.days
        if days <= 0:
            days = None

    league_filter = args.league.lower() if args.league else None
    models: List[str] = list(args.models)

    games = load_games(
        daily_dir=LEDGER_DIR,
        days=days,
        start_date=args.start_date,
        end_date=args.end_date,
        league_filter=league_filter,
        models=models + ["kalshi_mid"],
    )
    if not games:
        raise SystemExit("[error] no games found for given filters")

    leagues = sorted({g.league for g in games})
    print(
        f"[info] loaded {len(games)} game rows from {LEDGER_DIR} "
        f"(league={league_filter or 'all'}, models={','.join(models)}, "
        f"window={'custom' if (args.start_date or args.end_date) else (days or 'all')} days)"
    )

    rule_stats = compute_rule_stats(games, models=models, edge_threshold=args.edge_threshold)
    allowed = compute_allowed_buckets(rule_stats, ev_threshold=args.ev_threshold, min_bets=args.min_bets)
    print_rule_stats_table(rule_stats, allowed, models=models, leagues=leagues)

    consensus, baseline = backtest_scheme_d_consensus(
        games,
        models=models,
        allowed=allowed,
        edge_threshold=args.edge_threshold,
    )
    print_consensus_summary("Scheme D consensus sizing (1/3/5 units)", consensus, leagues=leagues)
    print_consensus_summary("Baseline (1 unit per allowed I/J signal)", baseline, leagues=leagues)

    if args.no_write:
        return

    out_dir = Path(args.out_dir)
    write_rule_stats_csv(out_dir / "scheme_d_rule_stats.csv", rule_stats, allowed)
    write_consensus_csv(
        out_dir / "scheme_d_backtest_consensus_135.csv",
        consensus,
        start_date=args.start_date,
        end_date=args.end_date,
        edge_threshold=args.edge_threshold,
        ev_threshold=args.ev_threshold,
        min_bets=args.min_bets,
        label="scheme_d_consensus_135",
    )
    write_consensus_csv(
        out_dir / "scheme_d_backtest_baseline_1u.csv",
        baseline,
        start_date=args.start_date,
        end_date=args.end_date,
        edge_threshold=args.edge_threshold,
        ev_threshold=args.ev_threshold,
        min_bets=args.min_bets,
        label="scheme_d_baseline_1u_per_signal",
    )
    write_rulebook_a_i_j_csv(
        out_dir / "rulebook_A_I_J_by_model_league.csv",
        leagues=leagues,
        models=models,
        allowed=allowed,
        start_date=args.start_date,
        end_date=args.end_date,
        edge_threshold=args.edge_threshold,
        ev_threshold=args.ev_threshold,
        min_bets=args.min_bets,
    )
    print(f"\n[info] wrote CSVs under {out_dir}/")


if __name__ == "__main__":
    main()
