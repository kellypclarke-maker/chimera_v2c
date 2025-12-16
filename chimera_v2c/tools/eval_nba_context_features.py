#!/usr/bin/env python
"""
Read-only evaluation of whether LLM packet context signals (form, rest/fatigue,
injury deltas, H2H) improve NBA v2c accuracy versus the baseline probabilities.
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chimera_v2c.lib.team_mapper import normalize_team_code
from chimera_v2c.src.ledger.outcomes import parse_home_win


THESIS_DIR = Path("reports/thesis_summaries")
LLM_PACKET_ROOT = Path("reports/llm_packets/nba")
MASTER_LEDGER_PATH = Path("reports/master_ledger/master_game_ledger.csv")

FEATURE_COLUMNS = [
    "logit_p0",
    "form_diff",
    "rest_diff",
    "b2b_diff",
    "inj_delta_diff",
    "h2h_home_win_pct",
    "h2h_margin_home",
    "streak_diff",
]


def clamp_prob(p: float) -> float:
    return float(np.clip(p, 0.01, 0.99))


def safe_logit(p: float) -> float:
    p = clamp_prob(p)
    return float(np.log(p / (1 - p)))


def parse_outcome(outcome: str) -> Optional[int]:
    hw = parse_home_win(outcome)
    if hw == 1.0:
        return 1
    if hw == 0.0:
        return 0
    return None


def normalize_team(code: str) -> Optional[str]:
    if code is None:
        return None
    norm = normalize_team_code(str(code), "nba")
    return norm or str(code).upper()


def parse_matchup(matchup: str) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(matchup, str) or "@" not in matchup:
        return None, None
    away_raw, home_raw = matchup.split("@", 1)
    return away_raw.strip(), home_raw.strip()


def ensure_packets(date_str: str, verbose: bool = True) -> Path:
    """
    Ensure the four expected packet CSVs exist for the given date (YYYY-MM-DD).
    """
    date_dir = LLM_PACKET_ROOT / pd.to_datetime(date_str).strftime("%Y%m%d")
    expected_files = [
        f"standings_form_{date_dir.name}_nba.csv",
        f"schedule_fatigue_{date_dir.name}_nba.csv",
        f"injuries_{date_dir.name}_nba.csv",
        f"h2h_{date_dir.name}_nba.csv",
    ]
    missing = [f for f in expected_files if not (date_dir / f).exists()]
    if missing:
        if verbose:
            print(f"[build_llm_packets] Missing packets for {date_str}: {missing}")
        env = os.environ.copy()
        env["PYTHONPATH"] = env.get("PYTHONPATH", ".")
        if "." not in env["PYTHONPATH"].split(":"):
            env["PYTHONPATH"] = f".:{env['PYTHONPATH']}"
        cmd = [
            "python",
            "chimera_v2c/tools/build_llm_packets.py",
            "--league",
            "nba",
            "--date",
            date_str,
            "--no-odds",
        ]
        subprocess.run(cmd, check=True, env=env)
    return date_dir


def load_packets(date_str: str) -> Dict[str, pd.DataFrame]:
    date_dir = ensure_packets(date_str, verbose=False)
    date_token = date_dir.name
    return {
        "form": pd.read_csv(date_dir / f"standings_form_{date_token}_nba.csv"),
        "schedule": pd.read_csv(date_dir / f"schedule_fatigue_{date_token}_nba.csv"),
        "injuries": pd.read_csv(date_dir / f"injuries_{date_token}_nba.csv"),
        "h2h": pd.read_csv(date_dir / f"h2h_{date_token}_nba.csv"),
    }


def compute_l10_pct(row: pd.Series) -> float:
    wins = pd.to_numeric(row.get("last10_w", 0), errors="coerce")
    losses = pd.to_numeric(row.get("last10_l", 0), errors="coerce")
    wins = 0 if pd.isna(wins) else wins
    losses = 0 if pd.isna(losses) else losses
    denom = wins + losses
    return float(wins / denom) if denom else 0.5


def compute_streak(row: pd.Series) -> float:
    streak_len = pd.to_numeric(row.get("streak_len", 0), errors="coerce")
    streak_len = 0 if pd.isna(streak_len) else streak_len
    streak_type = str(row.get("streak_type", "")).upper()
    if streak_type.startswith("W"):
        return float(streak_len)
    if streak_type.startswith("L"):
        return float(-streak_len)
    return 0.0


def extract_team_form(
    form_df: pd.DataFrame, team: str, is_home: int
) -> Tuple[float, float]:
    team_rows = form_df[
        (form_df["team_norm"] == team) & (form_df["is_home"] == is_home)
    ]
    if team_rows.empty:
        fallback = form_df[form_df["team_norm"] == team]
        if fallback.empty:
            return 0.5, 0.0
        team_rows = fallback
    row = team_rows.iloc[0]
    return compute_l10_pct(row), compute_streak(row)


def extract_rest_features(
    schedule_df: pd.DataFrame, home: str, away: str
) -> Tuple[float, int]:
    row = schedule_df[
        (schedule_df["home_team_norm"] == home) & (schedule_df["away_team_norm"] == away)
    ]
    if row.empty:
        return 0.0, 0
    row = row.iloc[0]
    home_rest = pd.to_numeric(row.get("home_days_rest", 0), errors="coerce")
    away_rest = pd.to_numeric(row.get("away_days_rest", 0), errors="coerce")
    home_rest = 0 if pd.isna(home_rest) else home_rest
    away_rest = 0 if pd.isna(away_rest) else away_rest
    rest_diff = float(home_rest - away_rest)
    home_b2b = int(pd.to_numeric(row.get("home_is_b2b", 0), errors="coerce") or 0)
    away_b2b = int(pd.to_numeric(row.get("away_is_b2b", 0), errors="coerce") or 0)
    b2b_diff = home_b2b - away_b2b
    return rest_diff, b2b_diff


def extract_injury_diff(team_delta: Dict[str, float], home: str, away: str) -> float:
    home_delta = team_delta.get(home, 0.0)
    away_delta = team_delta.get(away, 0.0)
    return float(home_delta - away_delta)


def extract_h2h_features(h2h_df: pd.DataFrame, home: str, away: str) -> Tuple[float, float]:
    row = h2h_df[
        (h2h_df["home_team_norm"] == home) & (h2h_df["away_team_norm"] == away)
    ]
    if row.empty:
        return 0.5, 0.0
    row = row.iloc[0]
    away_wins = pd.to_numeric(row.get("away_wins", 0), errors="coerce")
    home_wins = pd.to_numeric(row.get("home_wins", 0), errors="coerce")
    away_wins = 0 if pd.isna(away_wins) else away_wins
    home_wins = 0 if pd.isna(home_wins) else home_wins
    denom = away_wins + home_wins
    h2h_home_win_pct = float(home_wins / denom) if denom else 0.5
    margin = pd.to_numeric(row.get("avg_margin_away_minus_home", 0), errors="coerce")
    margin = 0 if pd.isna(margin) else margin
    h2h_margin_home = float(-margin)
    return h2h_home_win_pct, h2h_margin_home


def prepare_packet_views(packets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    # Normalize team codes once for quicker lookups.
    form_df = packets["form"].copy()
    form_df["team_norm"] = form_df["team"].apply(normalize_team)

    schedule_df = packets["schedule"].copy()
    schedule_df["home_team_norm"] = schedule_df["home_team"].apply(normalize_team)
    schedule_df["away_team_norm"] = schedule_df["away_team"].apply(normalize_team)

    injuries_df = packets["injuries"].copy()
    injuries_df["team_norm"] = injuries_df["team"].apply(normalize_team)
    injuries_df["team_delta"] = pd.to_numeric(injuries_df["team_delta"], errors="coerce")

    h2h_df = packets["h2h"].copy()
    h2h_df["home_team_norm"] = h2h_df["home_team"].apply(normalize_team)
    h2h_df["away_team_norm"] = h2h_df["away_team"].apply(normalize_team)
    h2h_df["away_wins"] = pd.to_numeric(h2h_df["away_wins"], errors="coerce")
    h2h_df["home_wins"] = pd.to_numeric(h2h_df["home_wins"], errors="coerce")
    h2h_df["avg_margin_away_minus_home"] = pd.to_numeric(
        h2h_df["avg_margin_away_minus_home"], errors="coerce"
    )

    return {
        "form": form_df,
        "schedule": schedule_df,
        "injuries": injuries_df,
        "h2h": h2h_df,
    }


def build_team_delta_map(injuries_df: pd.DataFrame) -> Dict[str, float]:
    delta_map: Dict[str, float] = {}
    for team, group in injuries_df.groupby("team_norm"):
        if not team:
            continue
        first_delta = group["team_delta"].dropna()
        if not first_delta.empty:
            delta_map[team] = float(first_delta.iloc[0])
    return delta_map


def build_dataset(master_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    records: List[Dict] = []
    unique_dates = sorted(master_df["date"].unique())
    for date_value in unique_dates:
        date_dt = pd.to_datetime(date_value)
        date_str = date_dt.strftime("%Y-%m-%d")
        packets = load_packets(date_str)
        packet_views = prepare_packet_views(packets)
        team_delta = build_team_delta_map(packet_views["injuries"])
        day_rows = master_df[master_df["date"] == date_value]
        for _, row in day_rows.iterrows():
            away_raw, home_raw = parse_matchup(row["matchup"])
            if not away_raw or not home_raw:
                continue
            away = normalize_team(away_raw)
            home = normalize_team(home_raw)
            if not away or not home:
                continue
            p0 = clamp_prob(float(row["v2c"]))
            y = parse_outcome(row["actual_outcome"])
            if y is None:
                continue

            home_l10, home_streak = extract_team_form(packet_views["form"], home, 1)
            away_l10, away_streak = extract_team_form(packet_views["form"], away, 0)
            form_diff = home_l10 - away_l10
            streak_diff = home_streak - away_streak

            rest_diff, b2b_diff = extract_rest_features(packet_views["schedule"], home, away)
            inj_delta_diff = extract_injury_diff(team_delta, home, away)
            h2h_home_win_pct, h2h_margin_home = extract_h2h_features(
                packet_views["h2h"], home, away
            )

            records.append(
                {
                    "date": date_dt,
                    "matchup": f"{away}@{home}",
                    "away_team": away,
                    "home_team": home,
                    "p0": p0,
                    "y": y,
                    "logit_p0": safe_logit(p0),
                    "form_diff": form_diff,
                    "rest_diff": rest_diff,
                    "b2b_diff": b2b_diff,
                    "inj_delta_diff": inj_delta_diff,
                    "h2h_home_win_pct": h2h_home_win_pct,
                    "h2h_margin_home": h2h_margin_home,
                    "streak_diff": streak_diff,
                    "home_l10_pct": home_l10,
                    "away_l10_pct": away_l10,
                    "home_streak_signed": home_streak,
                    "away_streak_signed": away_streak,
                }
            )
    dataset = pd.DataFrame.from_records(records)
    if dataset.empty:
        return dataset, "", ""
    dataset = dataset.sort_values("date").reset_index(drop=True)
    start_date = pd.to_datetime(dataset["date"].min()).strftime("%Y%m%d")
    end_date = pd.to_datetime(dataset["date"].max()).strftime("%Y%m%d")
    return dataset, start_date, end_date


def baseline_brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(penalty="l2", max_iter=500)),
        ]
    )


def cross_validate(dataset: pd.DataFrame) -> Tuple[float, List[Dict], np.ndarray]:
    groups = dataset["date"].dt.strftime("%Y-%m-%d")
    n_groups = groups.nunique()
    if n_groups < 2:
        # Not enough distinct dates for CV; fall back to full-fit prediction.
        return float("nan"), [], np.full(len(dataset), np.nan)

    gkf = GroupKFold(n_splits=n_groups)
    X = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataset["y"].to_numpy(dtype=float)
    p0 = dataset["p0"].to_numpy(dtype=float)

    oof_preds = np.full_like(y, np.nan, dtype=float)
    fold_results: List[Dict] = []

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        y_train = y[train_idx]
        y_test = y[test_idx]
        p0_test = p0[test_idx]

        if np.unique(y_train).size < 2:
            preds = np.full_like(y_test, np.mean(y_train), dtype=float)
        else:
            model = build_model()
            model.fit(X[train_idx], y_train)
            preds = model.predict_proba(X[test_idx])[:, 1]

        oof_preds[test_idx] = preds
        fold_baseline = baseline_brier(p0_test, y_test)
        fold_adjusted = baseline_brier(preds, y_test)
        fold_dates = sorted(set(groups.iloc[test_idx]))
        fold_results.append(
            {
                "dates": fold_dates,
                "n_games": int(len(test_idx)),
                "baseline_brier": float(fold_baseline),
                "adjusted_brier": float(fold_adjusted),
                "delta": float(fold_adjusted - fold_baseline),
            }
        )

    total_games = len(dataset)
    weighted_adjusted = sum(fr["adjusted_brier"] * fr["n_games"] for fr in fold_results) / total_games
    return float(weighted_adjusted), fold_results, oof_preds


def fit_full_model(dataset: pd.DataFrame) -> Tuple[Pipeline, np.ndarray]:
    X = dataset[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = dataset["y"].to_numpy(dtype=float)
    model = build_model()
    model.fit(X, y)
    preds = model.predict_proba(X)[:, 1]
    return model, preds


def summarize_coefficients(model: Pipeline) -> List[Dict]:
    logreg = model.named_steps["logreg"]
    coefs = logreg.coef_[0]
    return [{"feature": "intercept", "coef": float(logreg.intercept_[0])}] + [
        {"feature": feat, "coef": float(weight)} for feat, weight in zip(FEATURE_COLUMNS, coefs)
    ]


def load_master_ledger() -> pd.DataFrame:
    df = pd.read_csv(MASTER_LEDGER_PATH)
    df = df[df["league"].str.lower() == "nba"]
    df = df[df["actual_outcome"].notna()]
    df["date"] = pd.to_datetime(df["date"])
    df["v2c"] = pd.to_numeric(df["v2c"], errors="coerce")
    df = df[df["v2c"].notna()]
    return df


def main() -> None:
    master_df = load_master_ledger()
    if master_df.empty:
        print("No NBA rows with v2c probabilities found in the master ledger.")
        return

    dataset, start_date, end_date = build_dataset(master_df)
    if dataset.empty:
        print("No usable games after aligning with packets and outcomes.")
        return

    THESIS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = baseline_brier(dataset["p0"].to_numpy(dtype=float), dataset["y"].to_numpy(dtype=float))
    adjusted_cv, fold_results, oof_preds = cross_validate(dataset)
    model, full_preds = fit_full_model(dataset)
    coef_table = summarize_coefficients(model)

    dataset["p_adj_oof"] = oof_preds
    dataset["p_adj_full_fit"] = full_preds
    dataset_out = THESIS_DIR / f"nba_context_features_dataset_{start_date}_{end_date}.csv"
    dataset_out_df = dataset.copy()
    dataset_out_df["date"] = pd.to_datetime(dataset_out_df["date"]).dt.strftime("%Y-%m-%d")
    dataset_out_df.to_csv(dataset_out, index=False)

    summary = {
        "start_date": start_date,
        "end_date": end_date,
        "n_games": int(len(dataset)),
        "n_dates": int(dataset["date"].nunique()),
        "baseline_brier": baseline,
        "adjusted_cv_brier": adjusted_cv,
        "delta_brier": adjusted_cv - baseline if not np.isnan(adjusted_cv) else None,
        "folds": fold_results,
        "coefficients": coef_table,
    }

    summary_json = THESIS_DIR / f"nba_context_eval_summary_{start_date}_{end_date}.json"
    summary_csv = THESIS_DIR / f"nba_context_eval_summary_{start_date}_{end_date}.csv"
    summary_csv_rows = [
        {"metric": "start_date", "value": start_date},
        {"metric": "end_date", "value": end_date},
        {"metric": "n_games", "value": summary["n_games"]},
        {"metric": "n_dates", "value": summary["n_dates"]},
        {"metric": "baseline_brier", "value": summary["baseline_brier"]},
        {"metric": "adjusted_cv_brier", "value": summary["adjusted_cv_brier"]},
        {"metric": "delta_brier", "value": summary["delta_brier"]},
    ]
    for fold in summary["folds"]:
        summary_csv_rows.append(
            {
                "metric": f"fold_{'_'.join(fold['dates'])}_adjusted_brier",
                "value": fold["adjusted_brier"],
            }
        )
        summary_csv_rows.append(
            {
                "metric": f"fold_{'_'.join(fold['dates'])}_baseline_brier",
                "value": fold["baseline_brier"],
            }
        )

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(summary_csv_rows).to_csv(summary_csv, index=False)

    print("NBA context feature evaluation")
    print(f"Date range: {start_date} to {end_date} ({summary['n_dates']} dates, {summary['n_games']} games)")
    print(f"Baseline Brier: {baseline:.4f}")
    if np.isnan(adjusted_cv):
        print("Adjusted CV Brier: unavailable (need at least 2 dates for GroupKFold)")
    else:
        print(f"Adjusted CV Brier: {adjusted_cv:.4f}")
        print(f"Delta (adjusted - baseline): {summary['delta_brier']:.4f}")
    if fold_results:
        print("\nPer-fold results:")
        for fold in fold_results:
            date_label = ",".join(fold["dates"])
            print(
                f"  {date_label}: n={fold['n_games']}, "
                f"baseline={fold['baseline_brier']:.4f}, adjusted={fold['adjusted_brier']:.4f}, "
                f"delta={fold['delta']:.4f}"
            )
    print("\nCoefficients (full fit, standardized features):")
    for coef in coef_table:
        print(f"  {coef['feature']}: {coef['coef']:.4f}")
    print(f"\nDataset -> {dataset_out}")
    print(f"Summary  -> {summary_json} and {summary_csv}")


if __name__ == "__main__":
    main()
