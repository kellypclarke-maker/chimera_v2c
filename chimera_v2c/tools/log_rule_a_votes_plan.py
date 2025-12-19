#!/usr/bin/env python
"""
Log a production Rule-A BLIND+VOTES aggregated plan using LIVE Kalshi quotes (taker-only).

Rule A (baseline):
  - Qualifying game: Kalshi favors HOME now (home YES mid > 0.50).
  - Trade: buy AWAY YES at the AWAY YES ask (+ optional slippage).
  - Size: configurable (see --size-mode), capped at cap_units.
  - Vote for model m: require BOTH:
      1) (mid_home - p_model_home) >= vote_delta_m
      2) edge_net_m >= vote_edge_m
    where edge_net_m is fee-aware expected net per contract at the away ask:
      edge_net ≈ (1 - p_home) - price_away - fee(1 contract at price_away)
  - I-trigger (flip) for model m: p_model_home < 0.50 AND edge_net_m >= vote_edge_m
  - Weighted sizing option: count votes using --vote-weight and flips using --flip-weight.
  - Optional: load per-league per-model calibration from --calibration-json.

Outputs:
  - Writes a per-run plan CSV under:
      reports/execution_logs/rule_a_votes/YYYYMMDD/rule_a_votes_plan_<league>_<ts>.csv
    with empty fill columns (contracts_filled / price_away_filled / fees_filled) so the
    operator can record actual execution outcomes for true OOS tracking.

Safety:
  - Read-only on daily ledgers.
  - Uses Kalshi public markets (no private trading API).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from chimera_v2c.lib import nhl_scoreboard, team_mapper
from chimera_v2c.lib.env_loader import load_env_from_env_list
from chimera_v2c.src import market_linker
from chimera_v2c.src.ledger_analysis import LEDGER_DIR, GameRow, load_games
from chimera_v2c.src.rule_a_unit_policy import expected_edge_net_per_contract, mid_bucket, parse_matchup
from chimera_v2c.src.rule_a_vote_calibration import VoteDeltaCalibration
from chimera_v2c.src.rule_a_model_eligibility import ModelEligibility


SERIES_TICKER_BY_LEAGUE = {
    "nba": "KXNBAGAME",
    "nhl": "KXNHLGAME",
    "nfl": "KXNFLGAME",
}

DEFAULT_MODELS = ["v2c", "grok", "gemini", "gpt", "market_proxy", "moneypuck"]
V2C_PLAN_LOG_JSON = Path("reports/execution_logs/v2c_plan_log.json")


def _normalize_league(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"nba", "nhl", "nfl"}:
        return v
    raise SystemExit("[error] --league must be one of: nba, nhl, nfl")


def _parse_date_iso(text: str) -> date:
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"[error] invalid --date (expected YYYY-MM-DD): {text}") from exc


def _parse_iso_utc(text: str) -> Optional[datetime]:
    s = (text or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _clamp_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _fetch_start_times_by_matchup(league: str, date_iso: str) -> Dict[str, datetime]:
    fetcher = {
        "nba": nhl_scoreboard.fetch_nba_scoreboard,
        "nhl": nhl_scoreboard.fetch_nhl_scoreboard,
        "nfl": nhl_scoreboard.fetch_nfl_scoreboard,
    }.get(league)
    if fetcher is None:
        return {}

    sb = fetcher(date_iso)
    if sb.get("status") not in {"ok", "empty"}:
        return {}

    out: Dict[str, datetime] = {}
    for g in sb.get("games") or []:
        teams = g.get("teams") or {}
        away_alias = ((teams.get("away") or {}).get("alias") or "").strip()
        home_alias = ((teams.get("home") or {}).get("alias") or "").strip()
        away = team_mapper.normalize_team_code(away_alias, league)
        home = team_mapper.normalize_team_code(home_alias, league)
        if not away or not home:
            continue
        start = _parse_iso_utc(str(g.get("start_time") or ""))
        if start is None:
            continue
        out[f"{away}@{home}"] = start
    return out


@dataclass(frozen=True)
class VoteConfig:
    default_delta: float
    deltas_by_model: Dict[str, float]
    default_edge: float
    edges_by_model: Dict[str, float]

    def delta_for(self, model: str) -> float:
        if model in self.deltas_by_model:
            return float(self.deltas_by_model[model])
        return float(self.default_delta)

    def edge_for(self, model: str) -> float:
        if model in self.edges_by_model:
            return float(self.edges_by_model[model])
        return float(self.default_edge)


def _parse_model_thresholds(items: Sequence[str], *, arg_name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw in items:
        s = (raw or "").strip()
        if not s:
            continue
        if ":" not in s:
            raise SystemExit(f"[error] invalid {arg_name} '{raw}' (expected model:value)")
        model, val = s.split(":", 1)
        model = model.strip()
        try:
            v = float(val.strip())
        except ValueError as exc:
            raise SystemExit(f"[error] invalid {arg_name} '{raw}' (value must be float)") from exc
        if not math.isfinite(v):
            raise SystemExit(f"[error] invalid {arg_name} '{raw}' (value must be finite)")
        if model:
            out[model] = v
    return out


def _markets_by_matchup(
    markets: Sequence[market_linker.MarketQuote],
) -> Dict[str, Dict[str, market_linker.MarketQuote]]:
    out: Dict[str, Dict[str, market_linker.MarketQuote]] = {}
    for mq in markets:
        if not mq.away or not mq.home:
            continue
        yes_team = (mq.yes_team or "").strip().upper()
        if not yes_team:
            continue
        out.setdefault(f"{mq.away}@{mq.home}", {})[yes_team] = mq
    return out


def _plan_fieldnames(models: Sequence[str]) -> List[str]:
    base = [
        "date",
        "league",
        "matchup",
        "away",
        "home",
        "event_ticker",
        "start_time_utc",
        "minutes_to_start",
        "market_ticker_home",
        "market_ticker_away",
        "home_yes_bid_cents",
        "home_yes_ask_cents",
        "mid_home",
        "away_yes_bid_cents",
        "away_yes_ask_cents",
        "price_away_planned",
        "slippage_cents",
        "mid_bucket",
        "weak_bucket",
        "votes",
        "vote_models",
        "i_votes",
        "i_models",
        "union_models",
        "weighted_votes",
        "vote_delta_default",
        "vote_edge_default",
        "model_delta_overrides",
        "model_edge_overrides",
        "eligibility_primary_models",
        "eligibility_secondary_models",
        "votes_primary",
        "votes_secondary",
        "weighted_votes_primary",
        "weighted_votes_secondary",
        "size_mode",
        "contracts_planned",
        "edge_net_vote_sum",
        "contracts_filled",
        "price_away_filled",
        "fees_filled",
        "fill_ts_utc",
        "notes",
    ]
    dynamic: List[str] = []
    for m in models:
        m = str(m)
        dynamic.append(f"p_home_{m}")
    for m in models:
        m = str(m)
        dynamic.append(f"edge_net_{m}")
    for m in models:
        m = str(m)
        dynamic.append(f"vote_{m}")
    return base + dynamic


def _write_csv(
    path: Path,
    rows: Sequence[Dict[str, object]],
    *,
    fieldnames: Optional[Sequence[str]] = None,
    allow_empty: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not allow_empty:
            raise SystemExit("[error] no qualifying Rule-A games found to write")
        if not fieldnames:
            raise SystemExit("[error] cannot write empty plan without fieldnames")
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fieldnames))
            w.writeheader()
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames) if fieldnames else list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _append_rule_a_to_v2c_plan_log(*, rows: Sequence[Dict[str, object]], plan_csv: Path, ts_utc: str) -> None:
    """
    Append one entry per planned Rule-A trade into the shared v2c plan log.

    Downstream v2c analysis/backtests should filter `strategy == "v2c"` (or missing)
    to avoid mixing maker vs taker tracks.
    """
    entries: List[Dict[str, object]] = []
    for r in rows:
        date = str(r.get("date") or "").strip()
        league = str(r.get("league") or "").strip().lower()
        matchup = str(r.get("matchup") or "").strip().upper()
        away = str(r.get("away") or "").strip().upper()
        home = str(r.get("home") or "").strip().upper()
        ticker = str(r.get("market_ticker_away") or "").strip()

        entries.append(
            {
                "date": date,
                "league": league,
                "matchup": matchup,
                "home": home,
                "away": away,
                "yes_team": away,
                "mid": None,
                "p_yes": None,
                "p_home": None,
                "p_away": None,
                "edge_yes": None,
                "stake_fraction": None,
                "selected": True,
                "ticker": ticker or None,
                "strategy": "rule_a",
                "ts_utc": ts_utc,
                "source_plan_csv": str(plan_csv),
                "contracts_planned": r.get("contracts_planned"),
                "price_away_planned": r.get("price_away_planned"),
                "slippage_cents": r.get("slippage_cents"),
                "mid_home": r.get("mid_home"),
                "votes": r.get("votes"),
                "vote_models": r.get("vote_models"),
                "weighted_votes": r.get("weighted_votes"),
                "edge_net_vote_sum": r.get("edge_net_vote_sum"),
                "size_mode": r.get("size_mode"),
            }
        )

    V2C_PLAN_LOG_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, object]] = []
    if V2C_PLAN_LOG_JSON.exists():
        try:
            loaded = json.loads(V2C_PLAN_LOG_JSON.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []
    existing.extend(entries)
    V2C_PLAN_LOG_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def build_plan_rows(
    games: Sequence[GameRow],
    *,
    date_iso: str,
    league: str,
    markets: Sequence[market_linker.MarketQuote],
    models: Sequence[str],
    vote_cfg: VoteConfig,
    base_units: int,
    units_per_vote: int,
    cap_units: int,
    slippage_cents: int,
    min_minutes_to_start: Optional[float],
    max_minutes_to_start: Optional[float],
    weak_buckets: Sequence[str],
    weak_bucket_cap: int,
    size_mode: str,
    flip_delta_mode: str,
    vote_weight: int,
    flip_weight: int,
    eligible_models: Optional[Sequence[str]] = None,
    eligibility: Optional[ModelEligibility] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    now = datetime.now(timezone.utc)
    starts = _fetch_start_times_by_matchup(league, date_iso)
    by_matchup = _markets_by_matchup(markets)

    stats = {
        "ledger_games": 0,
        "missing_market": 0,
        "missing_quote": 0,
        "missing_start": 0,
        "filtered_by_time": 0,
        "not_home_fav": 0,
        "qualifying": 0,
    }

    slip = float(int(slippage_cents)) / 100.0
    cap = int(cap_units)
    base = int(base_units)
    per_vote = int(units_per_vote)
    weak = {str(b) for b in weak_buckets}
    weak_cap = int(weak_bucket_cap)
    mode = (size_mode or "blind_plus_votes").strip().lower()
    flip_mode = (flip_delta_mode or "none").strip().lower()
    vote_w = int(vote_weight)
    flip_w = int(flip_weight)
    eligible = set(str(m) for m in (eligible_models or []))
    use_eligibility = bool(eligible)
    primary_models = set(str(m) for m in (eligibility.primary_models if eligibility else []))
    secondary_models = set(str(m) for m in (eligibility.secondary_models if eligibility else []))
    use_primary_secondary = bool(eligibility is not None)

    rows: List[Dict[str, object]] = []
    for g in games:
        stats["ledger_games"] += 1
        parsed = parse_matchup(league=league, matchup=g.matchup)
        if parsed is None:
            continue
        away, home = parsed
        matchup = f"{away}@{home}"

        start = starts.get(matchup)
        minutes_to_start: Optional[float]
        if start is None:
            stats["missing_start"] += 1
            minutes_to_start = None
        else:
            minutes_to_start = (start - now).total_seconds() / 60.0
            if min_minutes_to_start is not None and minutes_to_start < float(min_minutes_to_start):
                stats["filtered_by_time"] += 1
                continue
            if max_minutes_to_start is not None and minutes_to_start > float(max_minutes_to_start):
                stats["filtered_by_time"] += 1
                continue

        quotes = by_matchup.get(matchup)
        if not quotes:
            stats["missing_market"] += 1
            continue
        home_q = quotes.get(home)
        away_q = quotes.get(away)
        if home_q is None or away_q is None:
            stats["missing_market"] += 1
            continue

        mid_home = home_q.mid
        if mid_home is None:
            stats["missing_quote"] += 1
            continue
        if float(mid_home) <= 0.5:
            stats["not_home_fav"] += 1
            continue

        if away_q.yes_ask is None:
            stats["missing_quote"] += 1
            continue
        price_away_planned = _clamp_price((float(away_q.yes_ask) / 100.0) + slip)

        trig: Dict[str, Dict[str, object]] = {}
        edge_sum = 0.0

        p_cols: Dict[str, object] = {}
        edge_cols: Dict[str, object] = {}
        vote_cols: Dict[str, object] = {}
        for m in models:
            p = g.probs.get(str(m))
            p_cols[f"p_home_{m}"] = "" if p is None else f"{float(p):.6f}"
            if p is None:
                edge_cols[f"edge_net_{m}"] = ""
                vote_cols[f"vote_{m}"] = ""
                continue
            edge = expected_edge_net_per_contract(p_home=float(p), price_away=float(price_away_planned))
            edge_cols[f"edge_net_{m}"] = f"{float(edge):.6f}"

            delta = vote_cfg.delta_for(str(m))
            edge_thr = vote_cfg.edge_for(str(m))
            edge_ok = float(edge) >= float(edge_thr)
            delta_ok = (float(mid_home) - float(p)) >= float(delta)
            is_i = (float(p) < 0.5) and edge_ok and (flip_mode != "same" or delta_ok)
            is_vote = delta_ok and edge_ok
            vote_cols[f"vote_{m}"] = int(is_vote)
            trig[str(m)] = {"vote": bool(is_vote), "flip": bool(is_i), "edge": float(edge)}

        # Apply calibration-eligible model filter (when using --calibration-json).
        if use_eligibility:
            for m in list(trig.keys()):
                if m not in eligible:
                    trig[m]["vote"] = False
                    trig[m]["flip"] = False

        # Apply primary/secondary gating:
        # - primary triggers always count
        # - secondary triggers count only if any primary triggers on the game
        # - models outside (primary U secondary) are ignored
        primary_tripped = False
        if use_primary_secondary:
            primary_tripped = any(
                (bool(v.get("vote")) or bool(v.get("flip"))) and (m in primary_models) for m, v in trig.items()
            )
            for m in list(trig.keys()):
                if m in primary_models:
                    continue
                if m in secondary_models:
                    if not primary_tripped:
                        trig[m]["vote"] = False
                        trig[m]["flip"] = False
                else:
                    trig[m]["vote"] = False
                    trig[m]["flip"] = False

        votes = 0
        vote_models: List[str] = []
        i_votes = 0
        i_models: List[str] = []
        votes_primary = 0
        votes_secondary = 0

        for m, v in trig.items():
            if bool(v.get("vote")):
                votes += 1
                vote_models.append(m)
                edge_sum += float(v.get("edge") or 0.0)
                if use_primary_secondary:
                    if m in primary_models:
                        votes_primary += 1
                    elif m in secondary_models:
                        votes_secondary += 1
            if bool(v.get("flip")):
                i_votes += 1
                i_models.append(m)

        vote_models.sort()
        i_models.sort()
        i_set = set(i_models)
        union_models = sorted(set(vote_models) | i_set)
        weighted_votes = sum(int(flip_w) if m in i_set else int(vote_w) for m in union_models)
        w_primary = 0
        w_secondary = 0
        if use_primary_secondary:
            for m in union_models:
                addw = int(flip_w) if m in i_set else int(vote_w)
                if m in primary_models:
                    w_primary += addw
                elif m in secondary_models:
                    w_secondary += addw

        # Sizing modes
        if mode == "blind_plus_i":
            contracts = min(cap, base + (per_vote * int(len(i_models))))
        elif mode == "blind_plus_weighted_flip2":
            contracts = min(cap, base + (per_vote * int(weighted_votes)))
        elif mode == "blind_plus_j2":
            extra = per_vote * int(votes) if int(votes) >= 2 else 0
            contracts = min(cap, base + int(extra))
        elif mode == "blind_plus_ij_extra":
            extra = per_vote * int(len(union_models)) if (int(len(i_models)) >= 1 or int(votes) >= 2) else 0
            contracts = min(cap, base + int(extra))
        elif mode == "ij_gated":
            eligible = (int(len(i_models)) >= 1) or (int(votes) >= 2)
            contracts = min(cap, base + (per_vote * int(len(union_models)))) if eligible else 0
        else:
            # default: blind_plus_votes
            contracts = min(cap, base + (per_vote * int(votes)))

        bucket = mid_bucket(float(mid_home))
        is_weak_bucket = bool(weak) and (bucket in weak)
        if is_weak_bucket:
            contracts = min(int(contracts), weak_cap)

        stats["qualifying"] += 1
        rows.append(
            {
                "date": date_iso,
                "league": league,
                "matchup": matchup,
                "away": away,
                "home": home,
                "event_ticker": home_q.event_ticker or away_q.event_ticker or "",
                "market_ticker_home": home_q.ticker,
                "market_ticker_away": away_q.ticker,
                "start_time_utc": "" if start is None else start.isoformat().replace("+00:00", "Z"),
                "minutes_to_start": "" if minutes_to_start is None else f"{minutes_to_start:.2f}",
                "home_yes_bid_cents": "" if home_q.yes_bid is None else int(home_q.yes_bid),
                "home_yes_ask_cents": "" if home_q.yes_ask is None else int(home_q.yes_ask),
                "mid_home": f"{float(mid_home):.6f}",
                "away_yes_bid_cents": "" if away_q.yes_bid is None else int(away_q.yes_bid),
                "away_yes_ask_cents": "" if away_q.yes_ask is None else int(away_q.yes_ask),
                "price_away_planned": f"{float(price_away_planned):.6f}",
                "slippage_cents": int(slippage_cents),
                "mid_bucket": bucket,
                "weak_bucket": int(is_weak_bucket),
                "votes": int(votes),
                "vote_models": ",".join(vote_models),
                "i_votes": int(i_votes),
                "i_models": ",".join(i_models),
                "union_models": ",".join(union_models),
                "weighted_votes": int(weighted_votes),
                "vote_delta_default": float(vote_cfg.default_delta),
                "vote_edge_default": float(vote_cfg.default_edge),
                "model_delta_overrides": ",".join(f"{k}:{vote_cfg.deltas_by_model[k]:.6f}" for k in sorted(vote_cfg.deltas_by_model)),
                "model_edge_overrides": ",".join(f"{k}:{vote_cfg.edges_by_model[k]:.6f}" for k in sorted(vote_cfg.edges_by_model)),
                "eligibility_primary_models": "" if eligibility is None else ",".join(sorted(primary_models)),
                "eligibility_secondary_models": "" if eligibility is None else ",".join(sorted(secondary_models)),
                "votes_primary": "" if eligibility is None else int(votes_primary),
                "votes_secondary": "" if eligibility is None else int(votes_secondary),
                "weighted_votes_primary": "" if eligibility is None else int(w_primary),
                "weighted_votes_secondary": "" if eligibility is None else int(w_secondary),
                "size_mode": str(mode),
                "contracts_planned": int(contracts),
                "edge_net_vote_sum": f"{float(edge_sum):.6f}",
                "contracts_filled": "",
                "price_away_filled": "",
                "fees_filled": "",
                "fill_ts_utc": "",
                "notes": "",
                **p_cols,
                **edge_cols,
                **vote_cols,
            }
        )

    rows.sort(key=lambda r: (r["start_time_utc"] or "9999", r["matchup"]))
    return rows, {k: int(v) for k, v in stats.items()}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Log a Rule-A BLIND+VOTES plan using live Kalshi quotes (taker-only).")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (ledger date).")
    ap.add_argument("--league", required=True, help="nba|nhl|nfl")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model columns to use for votes (default: v2c grok gemini gpt market_proxy moneypuck).")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="If no qualifying games are found, write a header-only plan CSV instead of exiting with an error.",
    )
    ap.add_argument("--base-units", type=int, default=1, help="Baseline contracts per qualifying game (default: 1).")
    ap.add_argument("--units-per-vote", type=int, default=1, help="Extra contracts per vote (default: 1).")
    ap.add_argument("--vote-weight", type=int, default=1, help="Weight per normal vote (used in weighted mode; default: 1).")
    ap.add_argument("--flip-weight", type=int, default=2, help="Weight per flip/I vote (used in weighted mode; default: 2).")
    ap.add_argument(
        "--size-mode",
        choices=["blind_plus_votes", "blind_plus_i", "blind_plus_weighted_flip2", "blind_plus_j2", "blind_plus_ij_extra", "ij_gated"],
        default="blind_plus_votes",
        help="How to convert votes into contracts (default: blind_plus_votes).",
    )
    ap.add_argument(
        "--flip-delta-mode",
        choices=["none", "same"],
        default="none",
        help="Whether a flip/I signal must also satisfy the model delta gate (default: none).",
    )
    ap.add_argument("--cap-units", type=int, default=10, help="Max contracts per game (default: 10).")
    ap.add_argument("--slippage-cents", type=int, default=0, help="Extra slippage added to away ask (default: 0).")
    ap.add_argument("--vote-delta", type=float, default=0.0, help="Default vote delta: require (mid_home - p_home) >= delta (default: 0).")
    ap.add_argument(
        "--vote-edge",
        type=float,
        default=0.0,
        help="Default vote edge: require fee-aware edge_net >= threshold (default: 0.0).",
    )
    ap.add_argument(
        "--model-delta",
        nargs="*",
        default=[],
        help="Per-model vote deltas, e.g. --model-delta grok:0.03 gpt:0.05 (default: none).",
    )
    ap.add_argument(
        "--model-edge",
        nargs="*",
        default=[],
        help="Per-model fee-aware edge thresholds, e.g. --model-edge market_proxy:0.01 moneypuck:0.02 (default: none).",
    )
    ap.add_argument(
        "--calibration-json",
        default="",
        help="Optional VoteDeltaCalibration JSON (overrides model deltas/edges, flip-delta-mode, and weights for voting).",
    )
    ap.add_argument(
        "--eligibility-json",
        default="",
        help="Optional ModelEligibility JSON (primary/secondary gating: secondary votes count only when a primary also triggers).",
    )
    ap.add_argument(
        "--no-eligibility",
        action="store_true",
        help="Disable auto-loading per-league model eligibility JSON (chimera_v2c/data/rule_a_model_eligibility_<league>.json).",
    )
    ap.add_argument(
        "--min-minutes-to-start",
        type=float,
        default=None,
        help="If set, include only games with minutes_to_start >= this value.",
    )
    ap.add_argument(
        "--max-minutes-to-start",
        type=float,
        default=None,
        help="If set, include only games with minutes_to_start <= this value.",
    )
    ap.add_argument(
        "--weak-buckets",
        nargs="*",
        default=[],
        help="Optional mid buckets to cap sizing in (e.g. 0.60-0.65 0.65-0.70).",
    )
    ap.add_argument("--weak-bucket-cap", type=int, default=3, help="Max contracts in weak buckets (default: 3).")
    ap.add_argument(
        "--weak-buckets-json",
        default="",
        help="Optional JSON with key weak_buckets (and optionally weak_bucket_cap) to cap sizing by mid bucket.",
    )
    ap.add_argument(
        "--no-auto-weak-buckets",
        action="store_true",
        help="Disable auto-loading chimera_v2c/data/rule_a_weak_buckets_<league>.json when --weak-buckets-json is omitted.",
    )
    ap.add_argument(
        "--kalshi-public-base",
        default="https://api.elections.kalshi.com/trade-api/v2",
        help="Kalshi public base (default: live trade-api/v2).",
    )
    ap.add_argument(
        "--append-v2c-plan-log",
        action="store_true",
        help="Also append Rule-A planned trades into reports/execution_logs/v2c_plan_log.json (strategy=rule_a).",
    )
    ap.add_argument("--out", default="", help="Optional output CSV path (default under reports/execution_logs/rule_a_votes/YYYYMMDD/).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    load_env_from_env_list()

    league = _normalize_league(args.league)
    date_obj = _parse_date_iso(args.date)
    date_iso = date_obj.isoformat()

    os.environ["KALSHI_PUBLIC_BASE"] = str(args.kalshi_public_base).strip()

    cap = int(args.cap_units)
    if cap < 1 or cap > 10_000:
        raise SystemExit("[error] --cap-units must be >= 1")
    base_units = int(args.base_units)
    if base_units < 0:
        raise SystemExit("[error] --base-units must be >= 0")
    units_per_vote = int(args.units_per_vote)
    if units_per_vote < 0:
        raise SystemExit("[error] --units-per-vote must be >= 0")
    if base_units == 0 and units_per_vote == 0:
        raise SystemExit("[error] base-units and units-per-vote cannot both be 0")

    eligible_models: Optional[List[str]] = None
    eligibility: Optional[ModelEligibility] = None
    if str(args.calibration_json).strip():
        calib = VoteDeltaCalibration.load_json(Path(str(args.calibration_json).strip()))
        deltas_by_model = dict(calib.vote_delta_by_model)
        edges_by_model = dict(calib.vote_edge_by_model)
        args.vote_delta = float(calib.vote_delta_default)
        args.vote_edge = float(calib.vote_edge_default)
        args.flip_delta_mode = str(calib.flip_delta_mode)
        args.vote_weight = int(calib.vote_weight)
        args.flip_weight = int(calib.flip_weight)
        eligible_models = list(calib.models)
    else:
        deltas_by_model = _parse_model_thresholds(list(args.model_delta), arg_name="--model-delta")
        edges_by_model = _parse_model_thresholds(list(args.model_edge), arg_name="--model-edge")

    if not bool(args.no_eligibility):
        if str(args.eligibility_json).strip():
            eligibility = ModelEligibility.load_json(Path(str(args.eligibility_json).strip()))
        else:
            default_path = Path("chimera_v2c/data") / f"rule_a_model_eligibility_{league}.json"
            if default_path.exists():
                eligibility = ModelEligibility.load_json(default_path)

    if eligibility is not None:
        if str(eligibility.league).strip().lower() != str(league).strip().lower():
            print(f"[warn] ignoring eligibility JSON for league={eligibility.league} (expected {league})")
            eligibility = None
        else:
            overlap = set(str(m) for m in args.models) & (
                set(str(m) for m in eligibility.primary_models) | set(str(m) for m in eligibility.secondary_models)
            )
            if not overlap:
                print("[warn] eligibility JSON has no overlap with --models; ignoring eligibility")
                eligibility = None
    vote_cfg = VoteConfig(
        default_delta=float(args.vote_delta),
        deltas_by_model=deltas_by_model,
        default_edge=float(args.vote_edge),
        edges_by_model=edges_by_model,
    )

    weak_buckets = list(args.weak_buckets)
    weak_bucket_cap = int(args.weak_bucket_cap)
    weak_json = str(args.weak_buckets_json).strip()
    if not weak_json and not bool(args.no_auto_weak_buckets):
        p = Path("chimera_v2c/data") / f"rule_a_weak_buckets_{league}.json"
        if p.exists():
            weak_json = str(p)
    if weak_json:
        import json

        with Path(str(weak_json).strip()).open("r", encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            wb = d.get("weak_buckets")
            if isinstance(wb, list):
                weak_buckets = [str(x) for x in wb]
            wbc = d.get("weak_bucket_cap")
            if wbc is not None:
                weak_bucket_cap = int(wbc)

    series = SERIES_TICKER_BY_LEAGUE.get(league)
    if not series:
        raise SystemExit(f"[error] missing series ticker mapping for league: {league}")

    games = load_games(
        daily_dir=LEDGER_DIR,
        start_date=date_iso,
        end_date=date_iso,
        league_filter=league,
        models=list(args.models),
    )
    if not games:
        raise SystemExit(
            f"[error] no daily ledger games found for {league} {date_iso}. "
            f"Ensure the file exists: {LEDGER_DIR / (date_obj.strftime('%Y%m%d') + '_daily_game_ledger.csv')}"
        )

    markets = market_linker.fetch_markets(
        league=league,
        series_ticker=series,
        use_private=False,
        status="open",
        target_date=date_obj,
    )

    rows, stats = build_plan_rows(
        games,
        date_iso=date_iso,
        league=league,
        markets=markets,
        models=list(args.models),
        vote_cfg=vote_cfg,
        base_units=base_units,
        units_per_vote=units_per_vote,
        cap_units=cap,
        slippage_cents=int(args.slippage_cents),
        min_minutes_to_start=args.min_minutes_to_start,
        max_minutes_to_start=args.max_minutes_to_start,
        weak_buckets=weak_buckets,
        weak_bucket_cap=int(weak_bucket_cap),
        size_mode=str(args.size_mode),
        flip_delta_mode=str(args.flip_delta_mode),
        vote_weight=int(args.vote_weight),
        flip_weight=int(args.flip_weight),
        eligible_models=eligible_models,
        eligibility=eligibility,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_path: Path
    if args.out:
        out_path = Path(str(args.out))
    else:
        out_path = Path("reports/execution_logs/rule_a_votes") / date_obj.strftime("%Y%m%d") / f"rule_a_votes_plan_{league}_{ts}.csv"

    fieldnames = _plan_fieldnames(list(args.models))
    _write_csv(out_path, rows, fieldnames=fieldnames, allow_empty=bool(args.allow_empty))
    print(f"[ok] wrote {len(rows)} qualifying games to {out_path}" if rows else f"[ok] wrote header-only plan (0 qualifying games) to {out_path}")
    print("[info] stats: " + ", ".join(f"{k}={v}" for k, v in stats.items()))
    if bool(args.append_v2c_plan_log):
        _append_rule_a_to_v2c_plan_log(rows=rows, plan_csv=out_path, ts_utc=str(ts))
        print(f"[ok] appended {len(rows)} Rule-A trades to {V2C_PLAN_LOG_JSON}")


if __name__ == "__main__":
    main()
