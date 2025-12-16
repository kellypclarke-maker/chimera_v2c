from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from chimera_v2c.lib.kalshi_executor import plan_orders_for_game, PlannedOrder
from chimera_v2c.lib.edge_calculator import compute_edge
from chimera_v2c.lib.sharp_odds import fetch_sharp_home_probs
from chimera_v2c.lib.env_loader import load_env_from_env_list

from chimera_v2c.src.config_loader import V2CConfig
from chimera_v2c.src.market_linker import MarketQuote, fetch_markets, fetch_matchups, match_markets_to_games
from chimera_v2c.src.probability import ProbabilityEngine
from chimera_v2c.src.risk import RiskSettings, compute_trade_decision
from chimera_v2c.src.ws_mid_cache import load_ws_cache
from chimera_v2c.src.calibration import PlattScaler


@dataclass
class SidePlan:
    yes_team: str
    market: Optional[MarketQuote]
    p_yes: float
    edge: Optional[float]
    stake_fraction: Optional[float]
    planned_orders: List[PlannedOrder]

    def to_dict(self) -> Dict:
        return {
            "yes_team": self.yes_team,
            "market": asdict(self.market) if self.market else None,
            "p_yes": self.p_yes,
            "edge": self.edge,
            "stake_fraction": self.stake_fraction,
            "planned_orders": [asdict(po) for po in self.planned_orders],
        }


@dataclass
class GamePlan:
    key: str
    home: str
    away: str
    yes_team: Optional[str]
    p_yes_selected: Optional[float]
    market: Optional[MarketQuote]
    p_final: float
    components: Dict[str, float]
    edge: Optional[float]
    stake_fraction: Optional[float]
    planned_orders: List[PlannedOrder]
    sides: List[SidePlan]
    diagnostics: Dict[str, object]

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Replace complex MarketQuote/PlannedOrder with dicts
        d["market"] = asdict(self.market) if self.market else None
        d["planned_orders"] = [asdict(po) for po in self.planned_orders]
        d["sides"] = [s.to_dict() for s in self.sides]
        return d


def build_daily_plan(cfg: V2CConfig, date: datetime.date) -> List[GamePlan]:
    load_env_from_env_list()
    use_private = bool(cfg.execution.get("use_private", False))
    exec_cfg = cfg.execution or {}
    require_quotes = bool(exec_cfg.get("require_quotes", False))
    max_spread = exec_cfg.get("max_spread")
    max_spread = float(max_spread) if max_spread is not None else None
    matchups = fetch_matchups(cfg.league, date)
    try:
        markets = fetch_markets(cfg.league, cfg.series_ticker, use_private=use_private, target_date=date)
    except RuntimeError as exc:
        if use_private and "KALSHI_API_KEY_ID" in str(exc):
            print("[warn] private Kalshi creds missing; falling back to public markets.")
            markets = fetch_markets(cfg.league, cfg.series_ticker, use_private=False, target_date=date)
        else:
            raise
    market_map = match_markets_to_games(matchups, markets)

    # Gating: ensure ratings and injury files are present so the operator
    # does not accidentally trade with incomplete information.
    ratings_path = Path(cfg.paths["ratings"])
    if not ratings_path.exists() or ratings_path.stat().st_size == 0:
        print(f"[error] ratings file missing or empty at {ratings_path}; "
              "run chimera_v2c/tools/prepare_data.py to populate team_ratings.json before planning.")
        return []
    try:
        ratings_data = json.loads(ratings_path.read_text(encoding="utf-8"))
        if not ratings_data:
            print(f"[error] ratings file at {ratings_path} is empty; "
                  "run chimera_v2c/tools/prepare_data.py before planning.")
            return []
    except Exception:
        print(f"[error] ratings file at {ratings_path} could not be parsed; "
              "regenerate it with chimera_v2c/tools/prepare_data.py before planning.")
        return []

    injury_path = Path(cfg.paths["injury"])
    if not injury_path.exists() or injury_path.stat().st_size == 0:
        print(f"[error] injury file missing or empty at {injury_path}; "
              "run chimera_v2c/tools/news_watcher.py before planning.")
        return []
    # Overlay WS cache mids if available
    ws_cache = load_ws_cache(Path(cfg.paths.get("ws_cache", ""))) if cfg.paths.get("ws_cache") else {}
    for market_dict in market_map.values():
        for mq in market_dict.values():
            cached = ws_cache.get(mq.ticker)
            if cached:
                if cached.get("yes_bid") is not None:
                    mq.yes_bid = cached["yes_bid"]
                if cached.get("yes_ask") is not None:
                    mq.yes_ask = cached["yes_ask"]

    # Fetch sharp home win probabilities (when available) as an additional prior.
    # When cfg.use_sharp_prior is true, treat The Odds API as required input:
    # missing keys or unexpected failures are fatal so we do not plan on
    # incomplete data. Hitting the quota/limit is allowed but logged, and
    # we fall back to Kalshi mids only.
    sharp_probs = (
        fetch_sharp_home_probs(cfg.league, date, require=True) if cfg.use_sharp_prior else {}
    )

    # Timestamp metadata for diagnostics
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    market_fetch_ts = now_utc.isoformat().replace("+00:00", "Z")
    ratings_mtime = datetime.datetime.fromtimestamp(ratings_path.stat().st_mtime, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        factors_mtime = datetime.datetime.fromtimestamp(Path(cfg.paths["factors"]).stat().st_mtime, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    except FileNotFoundError:
        factors_mtime = None
        print(f"[warn] factors file missing at {cfg.paths['factors']}; four_factors component may be empty.")
    injury_mtime = datetime.datetime.fromtimestamp(injury_path.stat().st_mtime, tz=datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    ws_overlay_used = bool(ws_cache)
    sharp_used = bool(sharp_probs)

    prob = ProbabilityEngine(
        league=cfg.league,
        weights=cfg.weights,
        elo_cfg=cfg.elo,
        ff_cfg=cfg.four_factors,
        ratings_path=Path(cfg.paths["ratings"]),
        factors_path=Path(cfg.paths["factors"]),
        injury_path=Path(cfg.paths["injury"]),
        goalie_cfg=getattr(cfg, "goalie", {}) or {},
        goalie_path=Path(cfg.paths["goalies"]) if cfg.paths.get("goalies") else None,
    )
    risk_cfg = RiskSettings(edge_min=cfg.edge_min, max_fraction=cfg.max_fraction)
    cal_cfg = getattr(cfg, "calibration", {}) or {}
    calibrator: Optional[PlattScaler] = None
    if cal_cfg.get("enabled"):
        cal_path = Path(cal_cfg.get("path", ""))
        if cal_path.exists():
            try:
                cal_raw = json.loads(cal_path.read_text(encoding="utf-8"))
                calibrator = PlattScaler(a=float(cal_raw.get("a", 1.0)), b=float(cal_raw.get("b", 0.0)))
            except Exception:
                print(f"[warn] failed to load calibration params from {cal_path}; skipping calibration.")
        else:
            print(f"[warn] calibration enabled but file missing at {cal_path}; skipping calibration.")

    def usable_mid(mq: Optional[MarketQuote]) -> tuple[Optional[float], Optional[str], Optional[float]]:
        if mq is None:
            return None, None, None
        spread = getattr(mq, "spread", None)
        if require_quotes and (mq.yes_bid is None or mq.yes_ask is None):
            return None, "missing_quotes", spread
        if max_spread is not None and spread is not None and spread > max_spread:
            return None, f"wide_spread>{max_spread:.2f}", spread
        return mq.mid, None, spread

    plans: List[GamePlan] = []
    used_fraction = 0.0
    daily_cap = getattr(cfg, "daily_max_fraction", cfg.max_fraction * 8)
    for m in matchups:
        # Canonical key: away@home for display/halt/logging
        key = f"{m['away']}@{m['home']}"
        market_dict = market_map.get(key, {})
        anchor_market = market_dict.get(m["home"]) or market_dict.get(m["away"])
        anchor_mid, anchor_reason, anchor_spread = usable_mid(anchor_market)
        p_sharp_home = sharp_probs.get(key)
        inj_home = prob._inj_delta(m["home"], date.strftime("%Y-%m-%d"))
        inj_away = prob._inj_delta(m["away"], date.strftime("%Y-%m-%d"))
        p_final, comps = prob.ensemble_prob(
            m["home"], m["away"], date.strftime("%Y-%m-%d"), anchor_mid, sharp_prior=p_sharp_home
        )
        if calibrator:
            comps["pre_calibration_final"] = p_final
            p_final = calibrator.predict(p_final)
            comps["final_calibrated"] = p_final

        # Compute p_yes for both tickers (home Yes and away Yes) for clarity
        p_yes_home = p_final
        p_yes_away = 1 - p_final

        side_plans: List[SidePlan] = []
        selected_side: Optional[SidePlan] = None
        selected_edge: Optional[float] = None
        reasons: List[str] = []
        if not market_dict:
            reasons.append("no_market_quotes")
        if anchor_market is None or anchor_mid is None:
            reasons.append("no_market_mid")
        if anchor_reason:
            reasons.append(anchor_reason)
        for yes_team, p_yes in [(m["home"], p_yes_home), (m["away"], p_yes_away)]:
            mq = market_dict.get(yes_team)
            mid_yes, mid_reason, spread_val = usable_mid(mq)
            if mid_reason:
                reasons.append(mid_reason)
            p_sharp_yes = None
            if p_sharp_home is not None:
                p_sharp_yes = p_sharp_home if yes_team == m["home"] else 1 - p_sharp_home
            market_signal = p_sharp_yes if p_sharp_yes is not None else mid_yes
            internal_prob = p_yes

            stake_fraction, target_price, reason_str = compute_trade_decision(
                p_model=p_yes if p_yes is not None else 0.5,
                market_mid=mid_yes,
                cfg=risk_cfg,
                used_fraction=used_fraction,
                daily_cap=daily_cap,
                target_spread_bp=cfg.target_spread_bp,
                require_confluence=cfg.require_confluence,
                internal_prob=internal_prob,
                market_signal=market_signal,
                league=cfg.league,
                doctrine_cfg=getattr(cfg, "doctrine_cfg", {}),
            )
            _, edge = compute_edge(p_yes if p_yes is not None else 0.5, mid_yes)
            planned_orders: List[PlannedOrder] = []
            if stake_fraction is not None and mq:
                planned_orders = plan_orders_for_game(
                    league=cfg.league,
                    date=date.strftime("%Y-%m-%d"),
                    game_id=m.get("event_id") or key,
                    market_ticker=mq.ticker if mq else None,
                    p_model=p_yes if p_yes is not None else p_final,
                    market_mid=mid_yes,
                    stake_fraction=stake_fraction,
                    target_price_override=target_price,
                )
            else:
                if reason_str:
                    reasons.append(reason_str)
                elif mid_yes is None:
                    reasons.append("no_mid_price")
            side_plan = SidePlan(
                yes_team=yes_team,
                market=mq,
                p_yes=p_yes,
                edge=edge,
                stake_fraction=stake_fraction,
                planned_orders=planned_orders,
            )
            side_plans.append(side_plan)

            # pick the side we would actually trade (highest edge with stake)
            if stake_fraction is not None:
                edge_for_select = edge if edge is not None else 0.0
                if selected_side is None or edge_for_select > (selected_edge if selected_edge is not None else -1):
                    selected_side = side_plan
                    selected_edge = edge_for_select

        if selected_side and selected_side.stake_fraction is not None:
            market = selected_side.market
            edge = selected_side.edge
            stake_fraction = selected_side.stake_fraction
            planned_orders = selected_side.planned_orders
            used_fraction += stake_fraction
            yes_team = selected_side.yes_team
            p_yes_selected = selected_side.p_yes
        else:
            market = anchor_market
            edge = None
            stake_fraction = None
            planned_orders = []
            yes_team = anchor_market.yes_team if anchor_market else None
            p_yes_selected = None

        plans.append(
            GamePlan(
                key=key,
                home=m["home"],
                away=m["away"],
                yes_team=yes_team,
                p_yes_selected=p_yes_selected,
                market=market,
                p_final=p_final,
                components=comps,
                edge=edge,
                stake_fraction=stake_fraction,
                planned_orders=planned_orders,
                sides=side_plans,
                diagnostics={
                    "market_fetch_ts": market_fetch_ts,
                    "ratings_mtime": ratings_mtime,
                    "factors_mtime": factors_mtime,
                    "injury_mtime": injury_mtime,
                    "injury_home_delta": inj_home,
                    "injury_away_delta": inj_away,
                    "ws_overlay_used": ws_overlay_used,
                    "sharp_prior_used": sharp_used,
                    "reasons": list(dict.fromkeys(reasons)),
                },
            )
        )
    return plans


def plans_to_json(plans: List[GamePlan]) -> str:
    payload = {"plans": [p.to_dict() for p in plans]}
    return json.dumps(payload, indent=2)
