from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from chimera_v2c.src.rule_a_unit_policy import (
    RuleAGame,
    SignalMetrics,
    Totals,
    expected_edge_net_per_contract,
    net_pnl_taker,
)
from chimera_v2c.src.rule_a_model_eligibility import ModelEligibility


FlipDeltaMode = str  # "none" | "same"


@dataclass(frozen=True)
class VoteDeltaCalibration:
    league: str
    trained_through: str  # YYYY-MM-DD (inclusive)
    models: List[str]
    vote_delta_by_model: Dict[str, float]
    vote_delta_default: float
    vote_edge_by_model: Dict[str, float]
    vote_edge_default: float
    flip_delta_mode: FlipDeltaMode
    vote_weight: int
    flip_weight: int
    cap_units: int
    base_units: int
    roi_guardrail_mode: str  # "strict" | "soft"
    roi_epsilon: float

    def to_json_dict(self) -> Dict[str, object]:
        return {
            "league": self.league,
            "trained_through": self.trained_through,
            "models": list(self.models),
            "vote_delta_default": float(self.vote_delta_default),
            "vote_delta_by_model": {k: float(v) for k, v in sorted(self.vote_delta_by_model.items())},
            "vote_edge_default": float(self.vote_edge_default),
            "vote_edge_by_model": {k: float(v) for k, v in sorted(self.vote_edge_by_model.items())},
            "flip_delta_mode": str(self.flip_delta_mode),
            "vote_weight": int(self.vote_weight),
            "flip_weight": int(self.flip_weight),
            "cap_units": int(self.cap_units),
            "base_units": int(self.base_units),
            "roi_guardrail_mode": str(self.roi_guardrail_mode),
            "roi_epsilon": float(self.roi_epsilon),
        }

    @classmethod
    def from_json_dict(cls, d: Dict[str, object]) -> "VoteDeltaCalibration":
        def _int_field(key: str, default: int) -> int:
            if key not in d or d.get(key) is None:
                return int(default)
            return int(d.get(key))  # allow 0 when explicitly set

        return cls(
            league=str(d.get("league") or ""),
            trained_through=str(d.get("trained_through") or ""),
            models=[str(x) for x in (d.get("models") or [])],
            vote_delta_by_model={str(k): float(v) for k, v in dict(d.get("vote_delta_by_model") or {}).items()},
            vote_delta_default=float(d.get("vote_delta_default") or 0.0),
            vote_edge_by_model={str(k): float(v) for k, v in dict(d.get("vote_edge_by_model") or {}).items()},
            vote_edge_default=float(d.get("vote_edge_default") or 0.0),
            flip_delta_mode=str(d.get("flip_delta_mode") or "none"),
            vote_weight=_int_field("vote_weight", 1),
            flip_weight=_int_field("flip_weight", 2),
            cap_units=_int_field("cap_units", 10),
            base_units=_int_field("base_units", 1),
            roi_guardrail_mode=str(d.get("roi_guardrail_mode") or "soft"),
            roi_epsilon=float(d.get("roi_epsilon") or 0.0),
        )

    @classmethod
    def load_json(cls, path: Path) -> "VoteDeltaCalibration":
        with path.open("r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("calibration JSON must be an object")
        return cls.from_json_dict(d)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=2, sort_keys=True)
            f.write("\n")


def _delta_thr(*, model: str, default: float, by_model: Dict[str, float]) -> float:
    if model in by_model:
        return float(by_model[model])
    return float(default)


def _edge_thr(*, model: str, default: float, by_model: Dict[str, float]) -> float:
    if model in by_model:
        return float(by_model[model])
    return float(default)


def model_contribution(
    g: RuleAGame,
    *,
    model: str,
    vote_delta_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_default: float,
    vote_edge_by_model: Dict[str, float],
    flip_delta_mode: FlipDeltaMode,
    vote_weight: int,
    flip_weight: int,
) -> Tuple[int, str]:
    """
    Return (added_contracts, signal_type) where signal_type is:
      - "flip" when p_home < 0.50 and gates pass
      - "vote" when (mid_home - p_home) >= delta and gates pass
      - "" otherwise
    """
    p = g.probs.get(str(model))
    if p is None:
        return 0, ""

    delta_thr = _delta_thr(model=str(model), default=float(vote_delta_default), by_model=vote_delta_by_model)
    delta_ok = (float(g.mid_home) - float(p)) >= float(delta_thr)

    edge_thr = _edge_thr(model=str(model), default=float(vote_edge_default), by_model=vote_edge_by_model)
    edge = expected_edge_net_per_contract(p_home=float(p), price_away=float(g.price_away))
    edge_ok = float(edge) >= float(edge_thr)

    is_flip = float(p) < 0.5
    if is_flip and edge_ok:
        if str(flip_delta_mode) == "same" and not delta_ok:
            return 0, ""
        return int(flip_weight), "flip"

    if delta_ok and edge_ok:
        return int(vote_weight), "vote"

    return 0, ""


def totals_for_policy(
    games: Iterable[RuleAGame],
    *,
    models: Sequence[str],
    vote_delta_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_default: float,
    vote_edge_by_model: Dict[str, float],
    flip_delta_mode: FlipDeltaMode,
    vote_weight: int,
    flip_weight: int,
    cap_units: int,
    base_units: int,
) -> Totals:
    cap = int(cap_units)
    base = int(base_units)
    tot = Totals()
    for g in games:
        if g.home_win is None:
            continue
        extra = 0
        for m in models:
            add, _ = model_contribution(
                g,
                model=str(m),
                vote_delta_default=float(vote_delta_default),
                vote_delta_by_model=vote_delta_by_model,
                vote_edge_default=float(vote_edge_default),
                vote_edge_by_model=vote_edge_by_model,
                flip_delta_mode=flip_delta_mode,
                vote_weight=int(vote_weight),
                flip_weight=int(flip_weight),
            )
            extra += int(add)
        contracts = min(cap, max(0, base + int(extra)))
        gross, fees, risked = net_pnl_taker(contracts=contracts, price_away=float(g.price_away), home_win=int(g.home_win))
        tot.bets += 1
        tot.contracts += int(contracts)
        tot.risked += float(risked)
        tot.gross_pnl += float(gross)
        tot.fees += float(fees)
    return tot


def contracts_for_game(
    g: RuleAGame,
    *,
    models: Sequence[str],
    vote_delta_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_default: float,
    vote_edge_by_model: Dict[str, float],
    flip_delta_mode: FlipDeltaMode,
    vote_weight: int,
    flip_weight: int,
    cap_units: int,
    base_units: int,
) -> Tuple[int, List[str]]:
    """
    Return (contracts, triggers) where triggers are strings like "model:vote:1" or "model:flip:2".
    """
    base = int(base_units)
    cap = int(cap_units)
    extra = 0
    triggers: List[str] = []
    for m in models:
        add, typ = model_contribution(
            g,
            model=str(m),
            vote_delta_default=float(vote_delta_default),
            vote_delta_by_model=vote_delta_by_model,
            vote_edge_default=float(vote_edge_default),
            vote_edge_by_model=vote_edge_by_model,
            flip_delta_mode=flip_delta_mode,
            vote_weight=int(vote_weight),
            flip_weight=int(flip_weight),
        )
        if int(add) > 0 and typ:
            triggers.append(f"{m}:{typ}:{int(add)}")
            extra += int(add)
    return min(cap, max(0, base + int(extra))), sorted(triggers)


def contracts_for_game_with_eligibility(
    g: RuleAGame,
    *,
    models: Sequence[str],
    vote_delta_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_default: float,
    vote_edge_by_model: Dict[str, float],
    flip_delta_mode: FlipDeltaMode,
    vote_weight: int,
    flip_weight: int,
    cap_units: int,
    base_units: int,
    eligibility: ModelEligibility,
) -> Tuple[int, List[str]]:
    """
    Like contracts_for_game(), but applies primary/secondary gating:
      - Primary triggers always count.
      - Secondary triggers count only if at least one primary also triggers.
      - Models not listed in primary/secondary are ignored.
    """
    primary = set(str(m) for m in eligibility.primary_models)
    secondary = set(str(m) for m in eligibility.secondary_models)

    base = int(base_units)
    cap = int(cap_units)

    primary_extra = 0
    secondary_extra = 0
    triggers: List[str] = []
    primary_tripped = False

    for m in models:
        add, typ = model_contribution(
            g,
            model=str(m),
            vote_delta_default=float(vote_delta_default),
            vote_delta_by_model=vote_delta_by_model,
            vote_edge_default=float(vote_edge_default),
            vote_edge_by_model=vote_edge_by_model,
            flip_delta_mode=flip_delta_mode,
            vote_weight=int(vote_weight),
            flip_weight=int(flip_weight),
        )
        if int(add) <= 0 or not typ:
            continue
        mm = str(m)
        if mm in primary:
            primary_tripped = True
            primary_extra += int(add)
            triggers.append(f"{mm}:{typ}:{int(add)}:primary")
        elif mm in secondary:
            secondary_extra += int(add)
            triggers.append(f"{mm}:{typ}:{int(add)}:secondary")
        else:
            # ineligible
            continue

    extra = int(primary_extra)
    if primary_tripped:
        extra += int(secondary_extra)
    else:
        # drop secondary-only triggers
        triggers = [t for t in triggers if t.endswith(":primary")]

    return min(cap, max(0, base + int(extra))), sorted(triggers)


def counts_by_model(
    games: Iterable[RuleAGame],
    *,
    models: Sequence[str],
    vote_delta_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_default: float,
    vote_edge_by_model: Dict[str, float],
    flip_delta_mode: FlipDeltaMode,
    vote_weight: int,
    flip_weight: int,
) -> Dict[str, int]:
    out: Dict[str, int] = {str(m): 0 for m in models}
    for g in games:
        if g.home_win is None:
            continue
        for m in models:
            add, _ = model_contribution(
                g,
                model=str(m),
                vote_delta_default=float(vote_delta_default),
                vote_delta_by_model=vote_delta_by_model,
                vote_edge_default=float(vote_edge_default),
                vote_edge_by_model=vote_edge_by_model,
                flip_delta_mode=flip_delta_mode,
                vote_weight=int(vote_weight),
                flip_weight=int(flip_weight),
            )
            if int(add) > 0:
                out[str(m)] = int(out.get(str(m), 0) + 1)
    return out


def learn_weak_mid_buckets_for_weighted_votes(
    train: Sequence[RuleAGame],
    *,
    models: Sequence[str],
    vote_delta_default: float,
    vote_delta_by_model: Dict[str, float],
    vote_edge_default: float,
    vote_edge_by_model: Dict[str, float],
    flip_delta_mode: FlipDeltaMode,
    vote_weight: int,
    flip_weight: int,
    cap_units: int,
    base_units: int,
    confidence_level: float,
    bootstrap_sims: int,
    min_bucket_bets: int,
    seed: int,
) -> Dict[str, SignalMetrics]:
    """
    Learn which mid-home buckets are "weak" for a weighted-votes Rule-A sizing policy,
    using only train outcomes.

    Samples per bucket are net-per-contract for the aggregated per-game order sized
    by the policy (including the base contract), so this detects "overbetting" regimes.
    """
    from chimera_v2c.src.rule_a_unit_policy import _stable_seed_u32, mid_bucket
    import numpy as np

    buckets: Dict[str, List[float]] = {}
    for g in train:
        if g.home_win is None:
            continue
        contracts, _ = contracts_for_game(
            g,
            models=list(models),
            vote_delta_default=float(vote_delta_default),
            vote_delta_by_model=vote_delta_by_model,
            vote_edge_default=float(vote_edge_default),
            vote_edge_by_model=vote_edge_by_model,
            flip_delta_mode=flip_delta_mode,
            vote_weight=int(vote_weight),
            flip_weight=int(flip_weight),
            cap_units=int(cap_units),
            base_units=int(base_units),
        )
        if int(contracts) <= 0:
            continue
        gross, fees, _ = net_pnl_taker(contracts=int(contracts), price_away=float(g.price_away), home_win=int(g.home_win))
        net_per_contract = (float(gross) - float(fees)) / float(int(contracts))
        buckets.setdefault(mid_bucket(float(g.mid_home)), []).append(float(net_per_contract))

    out: Dict[str, SignalMetrics] = {}
    alpha = 1.0 - float(confidence_level)
    for b, xs in buckets.items():
        n = int(len(xs))
        if n < int(min_bucket_bets):
            continue
        arr = np.array(xs, dtype=float)
        mean = float(arr.mean())

        rng = np.random.default_rng(int(_stable_seed_u32("weak_bucket_weighted", seed, b, n)))
        sims = int(bootstrap_sims)
        means = np.empty(sims, dtype=float)
        for i in range(sims):
            sample = rng.choice(arr, size=n, replace=True)
            means[i] = float(sample.mean())
        mean_low = float(np.quantile(means, alpha))
        conf_gt0 = float((means > 0.0).mean())
        out[b] = SignalMetrics(threshold=0.0, n=n, mean_net=mean, mean_low=mean_low, conf_gt0=conf_gt0)

    return out


def _passes_roi_guardrail(
    *,
    candidate: Totals,
    baseline: Totals,
    mode: str,
    roi_epsilon: float,
) -> bool:
    base_roi = float(baseline.roi_net)
    cand_roi = float(candidate.roi_net)
    if str(mode) == "strict":
        return cand_roi >= base_roi
    return cand_roi >= (base_roi - float(roi_epsilon))


def train_vote_delta_calibration(
    train: Sequence[RuleAGame],
    *,
    league: str,
    trained_through: str,
    models: Sequence[str],
    vote_deltas: Sequence[float],
    flip_delta_mode: FlipDeltaMode,
    cap_units: int,
    base_units: int,
    vote_weight: int,
    flip_weight: int,
    vote_delta_default: float = 0.0,
    vote_edge_default: float = 0.0,
    vote_edge_by_model: Optional[Dict[str, float]] = None,
    roi_guardrail_mode: str = "soft",
    roi_epsilon: float = 0.03,
    min_signals: int = 10,
    max_iters: int = 3,
    seed_deltas_by_model: Optional[Dict[str, float]] = None,
) -> Tuple[VoteDeltaCalibration, Totals, Totals, Dict[str, int]]:
    """
    Train per-model vote_delta thresholds on the provided train set using coordinate ascent.

    Returns:
      (calibration, totals_candidate, totals_baseline, signal_counts_train)
    """
    edge_by_model = dict(vote_edge_by_model or {})
    model_list = [str(m) for m in models]

    baseline_delta_by_model: Dict[str, float] = {m: 0.0 for m in model_list}
    baseline = totals_for_policy(
        train,
        models=model_list,
        vote_delta_default=float(vote_delta_default),
        vote_delta_by_model=baseline_delta_by_model,
        vote_edge_default=float(vote_edge_default),
        vote_edge_by_model=edge_by_model,
        flip_delta_mode="none",
        vote_weight=int(vote_weight),
        flip_weight=int(flip_weight),
        cap_units=int(cap_units),
        base_units=int(base_units),
    )

    current: Dict[str, float] = dict(seed_deltas_by_model or baseline_delta_by_model)
    deltas = sorted({float(x) for x in vote_deltas if float(x) >= 0.0})
    if not deltas:
        deltas = [0.0]

    def score(delta_by_model: Dict[str, float]) -> Optional[Totals]:
        cand = totals_for_policy(
            train,
            models=model_list,
            vote_delta_default=float(vote_delta_default),
            vote_delta_by_model=delta_by_model,
            vote_edge_default=float(vote_edge_default),
            vote_edge_by_model=edge_by_model,
            flip_delta_mode=flip_delta_mode,
            vote_weight=int(vote_weight),
            flip_weight=int(flip_weight),
            cap_units=int(cap_units),
            base_units=int(base_units),
        )
        if not _passes_roi_guardrail(candidate=cand, baseline=baseline, mode=str(roi_guardrail_mode), roi_epsilon=float(roi_epsilon)):
            return None
        return cand

    for _ in range(max(1, int(max_iters))):
        changed = False
        for m in model_list:
            best_delta = float(current.get(m, 0.0))
            best_totals = score(dict(current))
            if best_totals is None:
                best_totals = baseline

            for d in deltas:
                cand_delta_by_model = dict(current)
                cand_delta_by_model[m] = float(d)

                # Anti-snoop: if a delta triggers in too few games, treat it as "disabled" (skip).
                counts = counts_by_model(
                    train,
                    models=[m],
                    vote_delta_default=float(vote_delta_default),
                    vote_delta_by_model=cand_delta_by_model,
                    vote_edge_default=float(vote_edge_default),
                    vote_edge_by_model=edge_by_model,
                    flip_delta_mode=flip_delta_mode,
                    vote_weight=int(vote_weight),
                    flip_weight=int(flip_weight),
                )
                n = int(counts.get(m, 0))
                if 0 < n < int(min_signals):
                    continue

                cand_totals = score(cand_delta_by_model)
                if cand_totals is None:
                    continue

                better = False
                if float(cand_totals.net_pnl) > float(best_totals.net_pnl) + 1e-12:
                    better = True
                elif abs(float(cand_totals.net_pnl) - float(best_totals.net_pnl)) <= 1e-12:
                    if float(cand_totals.roi_net) > float(best_totals.roi_net) + 1e-12:
                        better = True
                    elif abs(float(cand_totals.roi_net) - float(best_totals.roi_net)) <= 1e-12:
                        if float(d) < float(best_delta):
                            better = True

                if better:
                    best_delta = float(d)
                    best_totals = cand_totals

            if float(best_delta) != float(current.get(m, 0.0)):
                current[m] = float(best_delta)
                changed = True
        if not changed:
            break

    final_totals = score(dict(current)) or totals_for_policy(
        train,
        models=model_list,
        vote_delta_default=float(vote_delta_default),
        vote_delta_by_model=dict(current),
        vote_edge_default=float(vote_edge_default),
        vote_edge_by_model=edge_by_model,
        flip_delta_mode=flip_delta_mode,
        vote_weight=int(vote_weight),
        flip_weight=int(flip_weight),
        cap_units=int(cap_units),
        base_units=int(base_units),
    )

    signal_counts = counts_by_model(
        train,
        models=model_list,
        vote_delta_default=float(vote_delta_default),
        vote_delta_by_model=dict(current),
        vote_edge_default=float(vote_edge_default),
        vote_edge_by_model=edge_by_model,
        flip_delta_mode=flip_delta_mode,
        vote_weight=int(vote_weight),
        flip_weight=int(flip_weight),
    )

    calib = VoteDeltaCalibration(
        league=str(league),
        trained_through=str(trained_through),
        models=model_list,
        vote_delta_by_model={str(k): float(v) for k, v in current.items()},
        vote_delta_default=float(vote_delta_default),
        vote_edge_by_model={str(k): float(v) for k, v in edge_by_model.items()},
        vote_edge_default=float(vote_edge_default),
        flip_delta_mode=str(flip_delta_mode),
        vote_weight=int(vote_weight),
        flip_weight=int(flip_weight),
        cap_units=int(cap_units),
        base_units=int(base_units),
        roi_guardrail_mode=str(roi_guardrail_mode),
        roi_epsilon=float(roi_epsilon),
    )
    return calib, final_totals, baseline, signal_counts


def choose_flip_delta_mode(
    train: Sequence[RuleAGame],
    *,
    league: str,
    trained_through: str,
    models: Sequence[str],
    vote_deltas: Sequence[float],
    flip_delta_modes: Sequence[FlipDeltaMode],
    cap_units: int,
    base_units: int,
    vote_weight: int,
    flip_weight: int,
    vote_delta_default: float = 0.0,
    vote_edge_default: float = 0.0,
    vote_edge_by_model: Optional[Dict[str, float]] = None,
    roi_guardrail_mode: str = "soft",
    roi_epsilon: float = 0.03,
    min_signals: int = 10,
    max_iters: int = 3,
) -> Tuple[VoteDeltaCalibration, Totals, Totals, Dict[str, int]]:
    """
    Train calibrations under multiple flip-delta modes and pick the best on train-only data.
    """
    best: Optional[Tuple[VoteDeltaCalibration, Totals, Totals, Dict[str, int]]] = None
    for mode in flip_delta_modes:
        calib, totals_c, totals_b, counts = train_vote_delta_calibration(
            train,
            league=league,
            trained_through=trained_through,
            models=models,
            vote_deltas=vote_deltas,
            flip_delta_mode=str(mode),
            cap_units=cap_units,
            base_units=base_units,
            vote_weight=vote_weight,
            flip_weight=flip_weight,
            vote_delta_default=vote_delta_default,
            vote_edge_default=vote_edge_default,
            vote_edge_by_model=vote_edge_by_model,
            roi_guardrail_mode=roi_guardrail_mode,
            roi_epsilon=roi_epsilon,
            min_signals=min_signals,
            max_iters=max_iters,
        )
        if best is None:
            best = (calib, totals_c, totals_b, counts)
            continue
        _, best_totals, _, _ = best
        better = False
        if float(totals_c.net_pnl) > float(best_totals.net_pnl) + 1e-12:
            better = True
        elif abs(float(totals_c.net_pnl) - float(best_totals.net_pnl)) <= 1e-12:
            if float(totals_c.roi_net) > float(best_totals.roi_net) + 1e-12:
                better = True
        if better:
            best = (calib, totals_c, totals_b, counts)
    if best is None:
        raise ValueError("no flip_delta_modes provided")
    return best
