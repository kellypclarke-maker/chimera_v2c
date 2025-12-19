from __future__ import annotations

import csv
import math
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from chimera_v2c.lib import team_mapper
from chimera_v2c.src.kalshi_fees import taker_fee_dollars
from chimera_v2c.src.ledger_analysis import GameRow


SnapshotKey = Tuple[str, str, str, str]  # (date_iso, league, matchup, yes_team)


@dataclass(frozen=True)
class TeamBidAsk:
    bid: float
    ask: float
    mid: float
    spread: float
    event_ticker: Optional[str] = None
    market_ticker: Optional[str] = None


@dataclass(frozen=True)
class RuleAGame:
    date: str  # YYYY-MM-DD
    league: str
    matchup: str  # AWAY@HOME (normalized)
    mid_home: float  # home YES mid at anchor
    price_away: float  # executed away YES price (ask + slippage, clamped)
    home_win: Optional[int]  # 1 home win, 0 away win, None unknown
    probs: Dict[str, float]  # p_home by model/proxy
    event_ticker: Optional[str] = None
    market_ticker_home: Optional[str] = None
    market_ticker_away: Optional[str] = None


@dataclass(frozen=True)
class SignalMetrics:
    threshold: float
    n: int
    mean_net: float
    mean_low: float
    conf_gt0: float


@dataclass(frozen=True)
class ModelPolicy:
    model: str
    threshold: float
    weight: float  # conservative mean net per added contract
    metrics: SignalMetrics


@dataclass(frozen=True)
class PolicyCalibration:
    league: str
    confidence_level: float
    unit_scale: float
    models: Dict[str, ModelPolicy]


@dataclass
class Totals:
    bets: int = 0
    contracts: int = 0
    risked: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0

    @property
    def net_pnl(self) -> float:
        return float(self.gross_pnl - self.fees)

    @property
    def roi_net(self) -> float:
        return 0.0 if self.risked <= 0 else float(self.net_pnl / self.risked)


def _clamp_price(p: float) -> float:
    return max(0.01, min(0.99, float(p)))


def _stable_seed_u32(*parts: object) -> int:
    """
    Deterministic 32-bit seed helper (avoid Python's randomized hash()).
    """
    s = "\x1f".join(str(p) for p in parts)
    return int(zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF)


def expected_edge_net_per_contract(*, p_home: float, price_away: float) -> float:
    """
    Fee-aware expected net edge (in dollars) for buying AWAY YES at taker.

    We approximate expected net per contract as:
      edge_net ≈ (1 - p_home) - price_away - fee(1 contract at price_away)
    """
    p = max(0.0, min(1.0, float(p_home)))
    price = _clamp_price(float(price_away))
    expected_gross = (1.0 - p) - price
    fee = taker_fee_dollars(contracts=1, price=price)
    return float(expected_gross - fee)


def mid_bucket(mid_home: float) -> str:
    """
    Match the Rule-A bid/ask suite bucketization:
      0.50-0.55, 0.55-0.60, 0.60-0.65, 0.65-0.70, 0.70-0.80, 0.80-1.00
    """
    p = float(mid_home)
    if p < 0.55:
        return "0.50-0.55"
    if p < 0.60:
        return "0.55-0.60"
    if p < 0.65:
        return "0.60-0.65"
    if p < 0.70:
        return "0.65-0.70"
    if p < 0.80:
        return "0.70-0.80"
    return "0.80-1.00"


def parse_matchup(*, league: str, matchup: str) -> Optional[Tuple[str, str]]:
    if "@" not in matchup:
        return None
    away_raw, home_raw = matchup.split("@", 1)
    away = team_mapper.normalize_team_code(away_raw.strip(), league)
    home = team_mapper.normalize_team_code(home_raw.strip(), league)
    if not away or not home:
        return None
    return away, home


def load_bidask_csv(path: Path) -> Dict[SnapshotKey, TeamBidAsk]:
    """
    Read a kalshi_bidask_tminus_*.csv export into a lookup table.

    Key: (date_iso, league, matchup, yes_team)
      - date_iso: YYYY-MM-DD
      - league: nba|nhl|nfl
      - matchup: AWAY@HOME (normalized to upper)
      - yes_team: team code (upper)
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return {}
        need = {"date", "league", "matchup", "yes_team", "yes_bid_cents", "yes_ask_cents", "mid", "spread", "ok"}
        missing = sorted(need - set(reader.fieldnames))
        if missing:
            raise ValueError(f"bid/ask csv missing columns: {missing}")

        out: Dict[SnapshotKey, TeamBidAsk] = {}
        for r in reader:
            if str(r.get("ok", "")).strip() not in {"1", "True", "true"}:
                continue
            date = str(r.get("date", "")).strip()
            league = str(r.get("league", "")).strip().lower()
            matchup = str(r.get("matchup", "")).strip().upper()
            yes_team = str(r.get("yes_team", "")).strip().upper()
            if not date or not league or not matchup or not yes_team:
                continue
            event_ticker = str(r.get("event_ticker") or "").strip() or None
            market_ticker = str(r.get("market_ticker") or "").strip() or None
            try:
                bid = float(r["yes_bid_cents"]) / 100.0
                ask = float(r["yes_ask_cents"]) / 100.0
                mid = float(r["mid"])
                spread = float(r["spread"])
            except Exception:
                continue
            out[(date, league, matchup, yes_team)] = TeamBidAsk(
                bid=bid,
                ask=ask,
                mid=mid,
                spread=spread,
                event_ticker=event_ticker,
                market_ticker=market_ticker,
            )
        return out


def iter_rule_a_games(
    ledger_games: Iterable[GameRow],
    *,
    bidask: Dict[SnapshotKey, TeamBidAsk],
    models: Sequence[str],
    slippage_cents: int,
    require_outcome: bool,
) -> List[RuleAGame]:
    slip = float(slippage_cents) / 100.0
    out: List[RuleAGame] = []
    for g in ledger_games:
        if require_outcome and g.home_win not in (0.0, 1.0):
            continue
        parsed = parse_matchup(league=g.league, matchup=g.matchup)
        if parsed is None:
            continue
        away, home = parsed
        date = g.date.strftime("%Y-%m-%d")
        matchup = f"{away}@{home}"
        home_row = bidask.get((date, g.league, matchup, home))
        away_row = bidask.get((date, g.league, matchup, away))
        if home_row is None or away_row is None:
            continue
        mid_home = float(home_row.mid)
        if mid_home <= 0.5:
            continue
        price_away = _clamp_price(float(away_row.ask) + slip)
        event_ticker = away_row.event_ticker or home_row.event_ticker
        home_win: Optional[int]
        if g.home_win in (0.0, 1.0):
            home_win = int(g.home_win)
        else:
            home_win = None
        probs = {m: float(g.probs[m]) for m in models if m in g.probs}
        out.append(
            RuleAGame(
                date=date,
                league=g.league,
                matchup=matchup,
                mid_home=mid_home,
                price_away=price_away,
                home_win=home_win,
                probs=probs,
                event_ticker=event_ticker,
                market_ticker_home=home_row.market_ticker,
                market_ticker_away=away_row.market_ticker,
            )
        )
    return out


def away_gross_pnl_per_contract(*, price_away: float, home_win: int) -> float:
    if int(home_win) == 0:
        return 1.0 - float(price_away)
    return -float(price_away)


def net_pnl_taker(*, contracts: int, price_away: float, home_win: int) -> Tuple[float, float, float]:
    """
    Return (gross_pnl, fees, risked) for buying AWAY YES at taker (price_away).
    """
    c = int(contracts)
    if c <= 0:
        return 0.0, 0.0, 0.0
    price = _clamp_price(float(price_away))
    gross = float(c) * away_gross_pnl_per_contract(price_away=price, home_win=int(home_win))
    fees = taker_fee_dollars(contracts=c, price=price)
    risked = float(c) * price
    return gross, float(fees), risked


def votes_for_game(
    g: RuleAGame,
    *,
    models: Sequence[str],
    vote_delta_default: float = 0.0,
    vote_edge_default: float = 0.0,
    vote_delta_by_model: Optional[Dict[str, float]] = None,
    vote_edge_by_model: Optional[Dict[str, float]] = None,
) -> Tuple[int, List[str]]:
    """
    Fee-aware Rule-A vote rule used for "BLIND + VOTES agg" sizing.

    A model m casts a vote when BOTH conditions pass:
      1) (mid_home - p_home_m) >= vote_delta_m
      2) edge_net_m >= vote_edge_m

    Where:
      - vote_delta_m defaults to vote_delta_default and can be overridden per-model.
      - vote_edge_m defaults to vote_edge_default and can be overridden per-model.
      - edge_net_m is fee-aware expected net per contract for buying AWAY YES at price_away.
    """
    delta_by_model = dict(vote_delta_by_model or {})
    edge_by_model = dict(vote_edge_by_model or {})
    votes = 0
    triggering: List[str] = []
    for m in models:
        p = g.probs.get(str(m))
        if p is None:
            continue
        delta_thr = float(delta_by_model.get(str(m), float(vote_delta_default)))
        if (float(g.mid_home) - float(p)) < delta_thr:
            continue
        edge_thr = float(edge_by_model.get(str(m), float(vote_edge_default)))
        edge = expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away))
        if float(edge) < edge_thr:
            continue
        votes += 1
        triggering.append(str(m))
    triggering.sort()
    return int(votes), triggering


def votes_agg_totals(
    games: Iterable[RuleAGame],
    *,
    models: Sequence[str],
    cap_units: int,
    weak_buckets: Optional[Sequence[str]] = None,
    weak_bucket_cap: int = 1,
) -> Totals:
    """
    Baseline "BLIND + VOTES agg":
      contracts = 1 + (# models with p_home < mid_home), capped.
    """
    cap = int(cap_units)
    if cap < 1:
        raise ValueError("cap_units must be >= 1")
    weak = {str(b) for b in (weak_buckets or [])}
    weak_cap = int(weak_bucket_cap)
    if weak_cap < 1:
        raise ValueError("weak_bucket_cap must be >= 1")
    tot = Totals()
    for g in games:
        if g.home_win is None:
            continue
        votes, _ = votes_for_game(g, models=models)
        contracts = min(cap, 1 + int(votes))
        if weak and mid_bucket(g.mid_home) in weak:
            contracts = min(int(contracts), weak_cap)
        gross, fees, risked = net_pnl_taker(contracts=contracts, price_away=g.price_away, home_win=int(g.home_win))
        tot.bets += 1
        tot.contracts += int(contracts)
        tot.risked += float(risked)
        tot.gross_pnl += float(gross)
        tot.fees += float(fees)
    return tot


def learn_weak_mid_buckets_for_votes(
    train: Sequence[RuleAGame],
    *,
    models: Sequence[str],
    cap_units: int,
    confidence_level: float,
    bootstrap_sims: int,
    min_bucket_bets: int,
    seed: int,
) -> Dict[str, SignalMetrics]:
    """
    Learn which mid-home buckets are "weak" for the BLIND+VOTES strategy,
    using only train outcomes.

    Returns bucket -> SignalMetrics where samples are net-per-contract for
    the BLIND+VOTES aggregated order in that bucket.
    """
    cap = int(cap_units)
    buckets: Dict[str, List[float]] = {}
    for g in train:
        if g.home_win is None:
            continue
        votes, _ = votes_for_game(g, models=models)
        contracts = min(cap, 1 + int(votes))
        gross, fees, _risk = net_pnl_taker(contracts=contracts, price_away=g.price_away, home_win=int(g.home_win))
        net = float(gross - fees)
        per_contract = net / float(contracts) if contracts > 0 else 0.0
        buckets.setdefault(mid_bucket(g.mid_home), []).append(float(per_contract))

    out: Dict[str, SignalMetrics] = {}
    for b, samples in buckets.items():
        if len(samples) < int(min_bucket_bets):
            continue
        metrics = signal_metrics(
            samples,
            threshold=0.0,
            confidence_level=float(confidence_level),
            bootstrap_sims=int(bootstrap_sims),
            seed=int(seed) ^ _stable_seed_u32("bucket", str(b)),
        )
        if metrics is None:
            continue
        out[str(b)] = metrics
    return out


def bootstrap_mean_distribution(
    samples: Sequence[float],
    *,
    sims: int,
    seed: int,
) -> np.ndarray:
    x = np.asarray(list(samples), dtype=float)
    n = int(x.size)
    if n == 0:
        return np.asarray([], dtype=float)
    if n == 1:
        return np.asarray([float(x[0])] * int(max(1, sims)), dtype=float)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n, size=(int(max(1, sims)), n), dtype=np.int64)
    return x[idx].mean(axis=1)


def signal_metrics(
    samples: Sequence[float],
    *,
    threshold: float,
    confidence_level: float,
    bootstrap_sims: int,
    seed: int,
) -> Optional[SignalMetrics]:
    xs = [float(x) for x in samples]
    if not xs:
        return None
    mean_net = float(np.mean(xs))
    means = bootstrap_mean_distribution(xs, sims=int(bootstrap_sims), seed=int(seed))
    if means.size == 0:
        return None
    cl = float(confidence_level)
    if not (0.5 <= cl < 1.0):
        raise ValueError("confidence_level must be in [0.5, 1.0).")
    q = 1.0 - cl
    mean_low = float(np.quantile(means, q))
    conf_gt0 = float((means > 0.0).mean())
    return SignalMetrics(
        threshold=float(threshold),
        n=len(xs),
        mean_net=mean_net,
        mean_low=mean_low,
        conf_gt0=conf_gt0,
    )


def select_model_policy(
    train: Sequence[RuleAGame],
    *,
    model: str,
    thresholds: Sequence[float],
    min_signals: int,
    confidence_level: float,
    bootstrap_sims: int,
    seed: int,
    select_mode: str,
) -> Optional[ModelPolicy]:
    mode = (select_mode or "max_total_mean_low").strip().lower()
    best: Optional[Tuple[float, float, SignalMetrics]] = None  # (score, threshold, metrics)
    for t in thresholds:
        tt = float(t)
        if tt < 0:
            continue
        samples: List[float] = []
        for g in train:
            if g.home_win is None:
                continue
            p = g.probs.get(model)
            if p is None:
                continue
            edge = expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away))
            if edge < tt:
                continue
            gross, fees, _risk = net_pnl_taker(contracts=1, price_away=g.price_away, home_win=int(g.home_win))
            samples.append(float(gross - fees))
        if len(samples) < int(min_signals):
            continue
        metrics = signal_metrics(
            samples,
            threshold=tt,
            confidence_level=float(confidence_level),
            bootstrap_sims=int(bootstrap_sims),
            seed=int(seed) ^ _stable_seed_u32(str(model), f"{tt:.6f}"),
        )
        if metrics is None:
            continue
        if metrics.mean_low <= 0.0:
            continue
        if mode == "min_threshold":
            score = -tt
        elif mode == "max_mean_low":
            score = metrics.mean_low
        else:
            score = metrics.mean_low * metrics.n
        cand = (float(score), float(tt), metrics)
        if best is None or cand[0] > best[0] + 1e-12 or (abs(cand[0] - best[0]) <= 1e-12 and cand[1] < best[1]):
            best = cand
    if best is None:
        return None
    _score, t_best, metrics_best = best
    return ModelPolicy(model=str(model), threshold=float(t_best), weight=float(metrics_best.mean_low), metrics=metrics_best)


def policy_totals(
    games: Iterable[RuleAGame],
    *,
    models: Dict[str, ModelPolicy],
    unit_scale: float,
    cap_units: int,
    weak_buckets: Optional[Sequence[str]] = None,
    weak_bucket_cap: int = 1,
) -> Totals:
    scale = float(unit_scale)
    if scale <= 0:
        raise ValueError("unit_scale must be > 0")
    cap = int(cap_units)
    if cap < 1:
        raise ValueError("cap_units must be >= 1")
    weak = {str(b) for b in (weak_buckets or [])}
    weak_cap = int(weak_bucket_cap)
    if weak_cap < 1:
        raise ValueError("weak_bucket_cap must be >= 1")

    tot = Totals()
    for g in games:
        if g.home_win is None:
            continue
        score = 0.0
        for mp in models.values():
            p = g.probs.get(mp.model)
            if p is None:
                continue
            edge = expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away))
            if edge >= float(mp.threshold):
                score += float(edge)
        extra = int(math.floor((score / scale) + 1e-12))
        contracts = min(cap, 1 + max(0, extra))
        if weak and mid_bucket(g.mid_home) in weak:
            contracts = min(int(contracts), weak_cap)
        gross, fees, risked = net_pnl_taker(contracts=contracts, price_away=g.price_away, home_win=int(g.home_win))
        tot.bets += 1
        tot.contracts += int(contracts)
        tot.risked += float(risked)
        tot.gross_pnl += float(gross)
        tot.fees += float(fees)
    return tot


def select_unit_scale(
    train: Sequence[RuleAGame],
    *,
    models: Dict[str, ModelPolicy],
    unit_scales: Sequence[float],
    cap_units: int,
    roi_floor_mult: float,
    weak_buckets: Optional[Sequence[str]] = None,
    weak_bucket_cap: int = 1,
) -> float:
    # Baseline ROI (1 contract per game).
    baseline = policy_totals(
        train,
        models={},
        unit_scale=1e9,
        cap_units=int(cap_units),
        weak_buckets=weak_buckets,
        weak_bucket_cap=int(weak_bucket_cap),
    )
    roi_floor = float("-inf")
    if baseline.roi_net > 0:
        roi_floor = float(baseline.roi_net) * float(roi_floor_mult)

    best_scale: Optional[float] = None
    best_net = float("-inf")
    best_roi = float("-inf")

    for s in unit_scales:
        scale = float(s)
        if scale <= 0:
            continue
        tot = policy_totals(
            train,
            models=models,
            unit_scale=scale,
            cap_units=int(cap_units),
            weak_buckets=weak_buckets,
            weak_bucket_cap=int(weak_bucket_cap),
        )
        if tot.roi_net + 1e-12 < roi_floor:
            continue
        if tot.net_pnl > best_net + 1e-12 or (abs(tot.net_pnl - best_net) <= 1e-12 and tot.roi_net > best_roi + 1e-12):
            best_scale = scale
            best_net = float(tot.net_pnl)
            best_roi = float(tot.roi_net)

    if best_scale is not None:
        return float(best_scale)

    # If nothing meets the ROI floor, fall back to max ROI among candidates (including baseline).
    best_scale = 1e9
    best_roi = float(baseline.roi_net)
    best_net = float(baseline.net_pnl)
    for s in unit_scales:
        scale = float(s)
        if scale <= 0:
            continue
        tot = policy_totals(
            train,
            models=models,
            unit_scale=scale,
            cap_units=int(cap_units),
            weak_buckets=weak_buckets,
            weak_bucket_cap=int(weak_bucket_cap),
        )
        if tot.roi_net > best_roi + 1e-12 or (abs(tot.roi_net - best_roi) <= 1e-12 and tot.net_pnl > best_net + 1e-12):
            best_scale = scale
            best_roi = float(tot.roi_net)
            best_net = float(tot.net_pnl)
    return float(best_scale)


def train_rule_a_policy(
    train: Sequence[RuleAGame],
    *,
    league: str,
    models: Sequence[str],
    thresholds: Sequence[float],
    unit_scales: Sequence[float],
    cap_units: int,
    min_signals: int,
    confidence_level: float,
    bootstrap_sims: int,
    roi_floor_mult: float,
    threshold_select_mode: str,
    seed: int,
    weak_buckets: Optional[Sequence[str]] = None,
    weak_bucket_cap: int = 1,
) -> PolicyCalibration:
    per_model: Dict[str, ModelPolicy] = {}
    for m in models:
        mp = select_model_policy(
            train,
            model=str(m),
            thresholds=thresholds,
            min_signals=int(min_signals),
            confidence_level=float(confidence_level),
            bootstrap_sims=int(bootstrap_sims),
            seed=int(seed) ^ _stable_seed_u32(str(league), str(m)),
            select_mode=str(threshold_select_mode),
        )
        if mp is None:
            continue
        per_model[str(m)] = mp

    unit_scale = select_unit_scale(
        train,
        models=per_model,
        unit_scales=unit_scales,
        cap_units=int(cap_units),
        roi_floor_mult=float(roi_floor_mult),
        weak_buckets=weak_buckets,
        weak_bucket_cap=int(weak_bucket_cap),
    )
    return PolicyCalibration(
        league=str(league),
        confidence_level=float(confidence_level),
        unit_scale=float(unit_scale),
        models=per_model,
    )


def units_for_game(
    g: RuleAGame,
    *,
    policy: PolicyCalibration,
    cap_units: int,
    weak_buckets: Optional[Sequence[str]] = None,
    weak_bucket_cap: int = 1,
) -> Tuple[int, float, List[str]]:
    score = 0.0
    triggering: List[str] = []
    for mp in policy.models.values():
        p = g.probs.get(mp.model)
        if p is None:
            continue
        edge = expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away))
        if edge >= float(mp.threshold):
            score += float(edge)
            triggering.append(mp.model)
    extra = int(math.floor((score / float(policy.unit_scale)) + 1e-12))
    contracts = min(int(cap_units), 1 + max(0, extra))
    weak = {str(b) for b in (weak_buckets or [])}
    if weak and mid_bucket(g.mid_home) in weak:
        contracts = min(int(contracts), int(weak_bucket_cap))
    triggering.sort()
    return contracts, float(score), triggering
