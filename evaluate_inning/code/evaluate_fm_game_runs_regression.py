# -*- coding: utf-8 -*-
"""
FM regression script for team-game run prediction.

Purpose:
- Build a team-game dataset from train_inning_runs_added_9_features.csv.
- Predict how many runs a team scores in the whole game, not in each inning.

Input:
  train_inning_runs_added_9_features.csv

Aggregation:
  one row = one team x one game
  target_game_runs = sum(target_runs_in_inning) over that team's offensive half-innings

Compared feature variants:
  1. all_game_features
  2. selected_game_features
  3. binned_game_features

Default execution:
  python evaluate_fm_game_runs_regression.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
  python evaluate_fm_game_runs_regression.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
Outputs:
  output_fm_game_runs_regression/
    team_game_runs_dataset.csv
    fm_game_runs_regression_metrics.csv
    target_game_runs_distribution.csv
    feature_variant_summary.csv
    <variant>_test_predictions.csv
    <variant>_rounded_prediction_distribution.csv

Notes:
- Split is time-based by game_date: train 80%, valid 10%, test 10% by unique dates.
- To avoid leakage, features are taken from the first offensive inning row for that team/game,
  and in-game outcome columns such as score_diff_before_inning, team_runs_before_inning,
  opponent_runs_before_inning, pitcher_bf_before_inning, and pitcher_runs_allowed_in_game
  are not used as model features.
- The model is FM regression trained with MSELoss.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 42
TARGET_COL = "target_game_runs"

# Game-level raw features. These are mostly pre-game / at-game-start features.
ALL_GAME_CATEGORICAL_FEATURES = [
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
    "current_pitcher_id",
]

ALL_GAME_NUMERIC_FEATURES = [
    "top_bottom",
    "team_avg_runs_7d",
    "opponent_avg_runs_7d",
    "opponent_avg_runs_allowed_7d",
    "opponent_starter_era",
    "pitcher_era_before_inning",
]

# More compact FM version. Removes current_pitcher_id and pitcher_era_before_inning,
# because at game start current_pitcher_id is usually the starter and pitcher_era_before_inning
# overlaps with opponent_starter_era.
SELECTED_GAME_CATEGORICAL_FEATURES = [
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
]

SELECTED_GAME_NUMERIC_FEATURES = [
    "top_bottom",
    "team_avg_runs_7d",
    "opponent_avg_runs_7d",
    "opponent_avg_runs_allowed_7d",
    "opponent_starter_era",
]

BINNED_GAME_CATEGORICAL_FEATURES = [
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
    "top_bottom_cat",
    "team_avg_runs_7d_bin",
    "opponent_avg_runs_7d_bin",
    "opponent_avg_runs_allowed_7d_bin",
    "opponent_starter_era_bin",
    "pitcher_era_before_inning_bin",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def first_non_null(values: pd.Series):
    non_null = values.dropna()
    if len(non_null) == 0:
        return np.nan
    return non_null.iloc[0]


def build_team_game_dataset(inning_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate inning-level rows to one row per team-game."""
    df = inning_df.copy()
    required = [
        "game_date", "game_id", "team_id", "opponent_team_id", "inning", "top_bottom",
        "home_away", "stadium_id", "starting_pitcher_id", "current_pitcher_id",
        "team_avg_runs_7d", "opponent_avg_runs_7d", "opponent_avg_runs_allowed_7d",
        "opponent_starter_era", "pitcher_era_before_inning", "target_runs_in_inning",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for game-level aggregation: {missing}")

    df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["inning"] = pd.to_numeric(df["inning"], errors="coerce")
    df["top_bottom"] = pd.to_numeric(df["top_bottom"], errors="coerce")
    df["target_runs_in_inning"] = pd.to_numeric(df["target_runs_in_inning"], errors="coerce").fillna(0)
    df = df.sort_values(["game_date_dt", "game_id", "team_id", "inning", "top_bottom"]).reset_index(drop=True)

    group_cols = ["game_date", "game_id", "team_id"]
    feature_cols = [
        "opponent_team_id", "top_bottom", "home_away", "stadium_id",
        "starting_pitcher_id", "current_pitcher_id",
        "team_avg_runs_7d", "opponent_avg_runs_7d", "opponent_avg_runs_allowed_7d",
        "opponent_starter_era", "pitcher_era_before_inning",
    ]

    rows = []
    for (game_date, game_id, team_id), g in df.groupby(group_cols, sort=False):
        g = g.sort_values(["inning", "top_bottom"])
        first = g.iloc[0]
        row = {
            "game_date": game_date,
            "game_id": game_id,
            "team_id": team_id,
            "target_game_runs": int(round(float(g["target_runs_in_inning"].sum()))),
            "n_offensive_innings": int(g["inning"].nunique()),
        }
        for col in feature_cols:
            row[col] = first[col] if col in first else np.nan
        rows.append(row)

    out = pd.DataFrame(rows)
    out["game_date_dt"] = pd.to_datetime(out["game_date"], errors="coerce")

    # Fill numeric missing values with train-time medians later; here only create stable bins.
    out = add_binned_game_features(out)
    return out.sort_values(["game_date_dt", "game_id", "top_bottom", "team_id"]).reset_index(drop=True)


def add_binned_game_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["top_bottom_cat"] = safe_numeric(out["top_bottom"], 0).astype(int).map({0: "top", 1: "bottom"}).fillna("unknown")

    def avg_runs_bin(x):
        if pd.isna(x):
            return "unknown"
        if x < 2.5:
            return "low"
        if x < 4.0:
            return "mid"
        if x < 5.5:
            return "high"
        return "very_high"

    def allowed_runs_bin(x):
        if pd.isna(x):
            return "unknown"
        if x < 2.5:
            return "low_allowed"
        if x < 4.0:
            return "mid_allowed"
        if x < 5.5:
            return "high_allowed"
        return "very_high_allowed"

    def era_bin(x):
        if pd.isna(x):
            return "unknown"
        if x < 2.50:
            return "excellent"
        if x < 3.50:
            return "good"
        if x < 4.50:
            return "average"
        if x < 6.00:
            return "high"
        return "very_high"

    out["team_avg_runs_7d_bin"] = safe_numeric(out["team_avg_runs_7d"], np.nan).apply(avg_runs_bin)
    out["opponent_avg_runs_7d_bin"] = safe_numeric(out["opponent_avg_runs_7d"], np.nan).apply(avg_runs_bin)
    out["opponent_avg_runs_allowed_7d_bin"] = safe_numeric(out["opponent_avg_runs_allowed_7d"], np.nan).apply(allowed_runs_bin)
    out["opponent_starter_era_bin"] = safe_numeric(out["opponent_starter_era"], np.nan).apply(era_bin)
    out["pitcher_era_before_inning_bin"] = safe_numeric(out["pitcher_era_before_inning"], np.nan).apply(era_bin)
    return out


def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["game_date_dt", "game_id", "top_bottom", "team_id"]).reset_index(drop=True)

    unique_dates = np.array(sorted(df["game_date_dt"].dropna().unique()))
    n_dates = len(unique_dates)
    train_end = int(n_dates * 0.8)
    valid_end = int(n_dates * 0.9)

    train_dates = set(unique_dates[:train_end])
    valid_dates = set(unique_dates[train_end:valid_end])
    test_dates = set(unique_dates[valid_end:])

    train_df = df[df["game_date_dt"].isin(train_dates)].copy()
    valid_df = df[df["game_date_dt"].isin(valid_dates)].copy()
    test_df = df[df["game_date_dt"].isin(test_dates)].copy()
    return train_df, valid_df, test_df


@dataclass
class FeatureVariant:
    name: str
    categorical_features: List[str]
    numeric_features: List[str]


def get_feature_variants() -> List[FeatureVariant]:
    return [
        FeatureVariant("all_game_features", ALL_GAME_CATEGORICAL_FEATURES, ALL_GAME_NUMERIC_FEATURES),
        FeatureVariant("selected_game_features", SELECTED_GAME_CATEGORICAL_FEATURES, SELECTED_GAME_NUMERIC_FEATURES),
        FeatureVariant("binned_game_features", BINNED_GAME_CATEGORICAL_FEATURES, []),
    ]


@dataclass
class PreprocessArtifacts:
    categorical_features: List[str]
    numeric_features: List[str]
    cat_maps: Dict[str, Dict[str, int]]
    cat_cardinalities: List[int]
    numeric_medians: Dict[str, float]
    scaler: Optional[StandardScaler]


def make_numeric_matrix(df: pd.DataFrame, numeric_features: List[str], numeric_medians: Dict[str, float]) -> np.ndarray:
    arrs = []
    for col in numeric_features:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(numeric_medians[col]).astype(float).values
        arrs.append(vals)
    if not arrs:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.vstack(arrs).T.astype(np.float32)


def fit_preprocess(train_df: pd.DataFrame, variant: FeatureVariant) -> PreprocessArtifacts:
    cat_maps: Dict[str, Dict[str, int]] = {}
    cat_cardinalities: List[int] = []
    for col in variant.categorical_features:
        values = train_df[col].fillna("<NA>").astype(str).unique().tolist()
        mapping = {v: i + 1 for i, v in enumerate(sorted(values))}  # 0 = unknown
        cat_maps[col] = mapping
        cat_cardinalities.append(len(mapping) + 1)

    numeric_medians = {}
    for col in variant.numeric_features:
        median = pd.to_numeric(train_df[col], errors="coerce").median()
        if pd.isna(median):
            median = 0.0
        numeric_medians[col] = float(median)

    X_num_train = make_numeric_matrix(train_df, variant.numeric_features, numeric_medians)
    if X_num_train.shape[1] > 0:
        scaler: Optional[StandardScaler] = StandardScaler()
        scaler.fit(X_num_train)
    else:
        # Some variants, such as binned_game_features, intentionally use only categorical features.
        # In that case StandardScaler cannot be fitted because there are zero numeric columns.
        scaler = None

    return PreprocessArtifacts(
        categorical_features=variant.categorical_features,
        numeric_features=variant.numeric_features,
        cat_maps=cat_maps,
        cat_cardinalities=cat_cardinalities,
        numeric_medians=numeric_medians,
        scaler=scaler,
    )


def make_cat_matrix(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> np.ndarray:
    cols = []
    for col in artifacts.categorical_features:
        mapping = artifacts.cat_maps[col]
        vals = df[col].fillna("<NA>").astype(str).map(mapping).fillna(0).astype(np.int64).values
        cols.append(vals)
    if not cols:
        return np.zeros((len(df), 0), dtype=np.int64)
    return np.vstack(cols).T.astype(np.int64)


def transform_for_fm(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> Tuple[np.ndarray, np.ndarray]:
    X_cat = make_cat_matrix(df, artifacts)
    X_num_raw = make_numeric_matrix(df, artifacts.numeric_features, artifacts.numeric_medians)
    if X_num_raw.shape[1] > 0:
        if artifacts.scaler is None:
            raise ValueError("Scaler is missing even though numeric features exist.")
        X_num = artifacts.scaler.transform(X_num_raw).astype(np.float32)
    else:
        X_num = X_num_raw.astype(np.float32)
    return X_cat, X_num


# -----------------------------------------------------------------------------
# FM Regression model
# -----------------------------------------------------------------------------

class RegressionDataset(Dataset):
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray):
        self.X_cat = X_cat.astype(np.int64)
        self.X_num = X_num.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class RegressionFM(nn.Module):
    def __init__(self, cat_cardinalities: List[int], num_numeric: int, embed_dim: int = 16):
        super().__init__()
        self.num_cat = len(cat_cardinalities)
        self.num_numeric = num_numeric
        self.embed_dim = embed_dim

        offsets = np.cumsum([0] + cat_cardinalities[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        total_cat = int(sum(cat_cardinalities)) if cat_cardinalities else 1

        self.cat_linear = nn.Embedding(total_cat, 1)
        self.cat_embed = nn.Embedding(total_cat, embed_dim)
        self.num_linear = nn.Linear(num_numeric, 1) if num_numeric > 0 else None
        self.num_embed = nn.Parameter(torch.empty(num_numeric, embed_dim)) if num_numeric > 0 else None
        self.interaction_proj = nn.Linear(embed_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.cat_linear.weight, std=0.01)
        nn.init.normal_(self.cat_embed.weight, std=0.01)
        if self.num_linear is not None:
            nn.init.xavier_uniform_(self.num_linear.weight)
            nn.init.zeros_(self.num_linear.bias)
        if self.num_embed is not None:
            nn.init.normal_(self.num_embed, std=0.01)
        nn.init.xavier_uniform_(self.interaction_proj.weight)
        nn.init.zeros_(self.interaction_proj.bias)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size = x_cat.shape[0]
        out = self.bias.unsqueeze(0).expand(batch_size, 1)

        embed_fields = []
        if self.num_cat > 0:
            x_cat_off = x_cat + self.offsets.unsqueeze(0)
            out = out + self.cat_linear(x_cat_off).sum(dim=1)
            embed_fields.append(self.cat_embed(x_cat_off))

        if self.num_numeric > 0:
            out = out + self.num_linear(x_num)
            num_emb = x_num.unsqueeze(-1) * self.num_embed.unsqueeze(0)
            embed_fields.append(num_emb)

        if embed_fields:
            V = torch.cat(embed_fields, dim=1)
            V = self.dropout(V)
            summed = V.sum(dim=1)
            interaction_vec = 0.5 * (summed * summed - (V * V).sum(dim=1))
            out = out + self.interaction_proj(interaction_vec)
        return out.squeeze(1)


def evaluate_mse(model: nn.Module, loader: DataLoader, criterion, device) -> float:
    model.eval()
    losses = []
    total_n = 0
    with torch.no_grad():
        for xb_cat, xb_num, yb in loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            yb = yb.to(device)
            pred = model(xb_cat, xb_num)
            loss = criterion(pred, yb)
            losses.append(loss.item() * len(yb))
            total_n += len(yb)
    return float(np.sum(losses) / max(total_n, 1))


def train_fm_regression(
    X_cat_train: np.ndarray,
    X_num_train: np.ndarray,
    y_train: np.ndarray,
    X_cat_valid: np.ndarray,
    X_num_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_cardinalities: List[int],
    max_epochs: int = 80,
    batch_size: int = 256,
    patience: int = 8,
    lr: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(RegressionDataset(X_cat_train, X_num_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(RegressionDataset(X_cat_valid, X_num_valid, y_valid), batch_size=batch_size, shuffle=False)

    model = RegressionFM(cat_cardinalities, X_num_train.shape[1], embed_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_state = None
    best_valid_mse = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb_cat, xb_num, yb in train_loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb_cat, xb_num)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)

        train_mse = total_loss / max(total_n, 1)
        valid_mse = evaluate_mse(model, valid_loader, criterion, device)
        print(f"epoch={epoch:03d} train_mse={train_mse:.5f} valid_mse={valid_mse:.5f}")

        if valid_mse < best_valid_mse - 1e-5:
            best_valid_mse = valid_mse
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping: best_epoch={best_epoch}, best_valid_mse={best_valid_mse:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": float(best_epoch), "best_valid_mse": float(best_valid_mse)}


def predict_fm_regression(model: nn.Module, X_cat: np.ndarray, X_num: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    dummy_y = np.zeros(len(X_cat), dtype=np.float32)
    loader = DataLoader(RegressionDataset(X_cat, X_num, dummy_y), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb_cat, xb_num, _ in loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            pred = model(xb_cat, xb_num).cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    pred_non_negative = np.clip(pred, 0, None)
    pred_rounded = np.rint(pred_non_negative).astype(int)
    true_int = y_true.astype(int)

    # Poisson deviance requires strictly positive y_pred.
    pred_pos = np.clip(pred_non_negative, 1e-6, None)

    return {
        "mae": float(mean_absolute_error(y_true, pred_non_negative)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred_non_negative))),
        "r2": float(r2_score(y_true, pred_non_negative)),
        "poisson_deviance": float(mean_poisson_deviance(y_true, pred_pos)),
        "exact_game_runs_accuracy": float(np.mean(pred_rounded == true_int)),
        "within_1_run_accuracy": float(np.mean(np.abs(pred_rounded - true_int) <= 1)),
        "within_2_runs_accuracy": float(np.mean(np.abs(pred_rounded - true_int) <= 2)),
        "zero_vs_nonzero_accuracy_after_rounding": float(np.mean((pred_rounded > 0) == (true_int > 0))),
        "true_mean_runs": float(np.mean(y_true)),
        "pred_mean_runs": float(np.mean(pred_non_negative)),
        "pred_rounded_mean_runs": float(np.mean(pred_rounded)),
    }


def run_variant(
    variant: FeatureVariant,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
    fm_epochs: int,
) -> Dict[str, object]:
    print("\n" + "-" * 80)
    print(f"FM game-run regression: variant={variant.name}")
    print("-" * 80)
    print(f"categorical_features ({len(variant.categorical_features)}): {variant.categorical_features}")
    print(f"numeric_features ({len(variant.numeric_features)}): {variant.numeric_features}")

    artifacts = fit_preprocess(train_df, variant)
    X_cat_train, X_num_train = transform_for_fm(train_df, artifacts)
    X_cat_valid, X_num_valid = transform_for_fm(valid_df, artifacts)
    X_cat_test, X_num_test = transform_for_fm(test_df, artifacts)

    y_train = pd.to_numeric(train_df[TARGET_COL], errors="coerce").fillna(0).astype(float).values
    y_valid = pd.to_numeric(valid_df[TARGET_COL], errors="coerce").fillna(0).astype(float).values
    y_test = pd.to_numeric(test_df[TARGET_COL], errors="coerce").fillna(0).astype(float).values

    model, info = train_fm_regression(
        X_cat_train,
        X_num_train,
        y_train,
        X_cat_valid,
        X_num_valid,
        y_valid,
        artifacts.cat_cardinalities,
        max_epochs=fm_epochs,
    )
    pred = predict_fm_regression(model, X_cat_test, X_num_test)
    pred_non_negative = np.clip(pred, 0, None)
    pred_rounded = np.rint(pred_non_negative).astype(int)

    metrics = regression_metrics(y_test, pred)
    metrics.update(info)
    metrics.update({
        "variant": variant.name,
        "model": "fm_regression",
        "n_categorical_features": len(variant.categorical_features),
        "n_numeric_features": len(variant.numeric_features),
        "n_total_features": len(variant.categorical_features) + len(variant.numeric_features),
    })

    pred_df = pd.DataFrame({
        "game_date": test_df["game_date"].values,
        "game_id": test_df["game_id"].values,
        "team_id": test_df["team_id"].values,
        "opponent_team_id": test_df["opponent_team_id"].values,
        "home_away": test_df["home_away"].values,
        "target_game_runs": y_test.astype(int),
        "pred_game_runs_raw": pred,
        "pred_game_runs_non_negative": pred_non_negative,
        "pred_game_runs_rounded": pred_rounded,
        "abs_error": np.abs(pred_non_negative - y_test),
        "rounded_abs_error": np.abs(pred_rounded - y_test),
    })
    pred_df.to_csv(os.path.join(output_dir, f"{variant.name}_test_predictions.csv"), index=False, encoding="utf-8-sig")

    max_runs = int(max(y_test.max(), pred_rounded.max()))
    dist_rows = []
    for runs in range(max_runs + 1):
        dist_rows.append({
            "runs": runs,
            "true_count": int(np.sum(y_test.astype(int) == runs)),
            "pred_rounded_count": int(np.sum(pred_rounded == runs)),
        })
    pd.DataFrame(dist_rows).to_csv(
        os.path.join(output_dir, f"{variant.name}_rounded_prediction_distribution.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    feature_rows = []
    for c in variant.categorical_features:
        feature_rows.append({"feature": c, "feature_type": "categorical"})
    for c in variant.numeric_features:
        feature_rows.append({"feature": c, "feature_type": "numeric"})
    pd.DataFrame(feature_rows).to_csv(
        os.path.join(output_dir, f"{variant.name}_features.csv"), index=False, encoding="utf-8-sig"
    )

    print("[FM Game Regression Test Metrics]")
    for k, v in metrics.items():
        if k not in {"variant", "model"}:
            print(f"{k}: {v}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/train_inning_runs_added_9_features.csv")
    parser.add_argument("--output_dir", type=str, default="output_fm_game_runs_regression")
    parser.add_argument("--fm_epochs", type=int, default=80)
    parser.add_argument("--dry_run", action="store_true", help="Only build dataset/splits and save feature summaries; do not train.")
    args = parser.parse_args()

    set_seed(SEED)
    ensure_dir(args.output_dir)

    print("=" * 80)
    print("FM team-game run-count regression")
    print("=" * 80)
    print(f"csv_path: {args.csv_path}")
    print(f"output_dir: {args.output_dir}")

    inning_df = pd.read_csv(args.csv_path)
    game_df = build_team_game_dataset(inning_df)
    game_df.to_csv(os.path.join(args.output_dir, "team_game_runs_dataset.csv"), index=False, encoding="utf-8-sig")

    train_df, valid_df, test_df = split_by_date(game_df)

    print("\n[Data split: team-game rows]")
    for name, part in [("all", game_df), ("train", train_df), ("valid", valid_df), ("test", test_df)]:
        dt = pd.to_datetime(part["game_date"], errors="coerce")
        if len(part) == 0:
            print(f"{name:5s}: {len(part):6d} rows")
        else:
            print(f"{name:5s}: {len(part):6d} rows, {dt.min().date()} ~ {dt.max().date()}, dates={dt.nunique()}")

    dist = game_df[TARGET_COL].value_counts().sort_index()
    dist.to_csv(os.path.join(args.output_dir, "target_game_runs_distribution.csv"), encoding="utf-8-sig")
    print("\n[Target distribution: target_game_runs]")
    print(dist)

    variants = get_feature_variants()
    summary = pd.DataFrame([
        {
            "variant": v.name,
            "n_categorical_features": len(v.categorical_features),
            "n_numeric_features": len(v.numeric_features),
            "n_total_features": len(v.categorical_features) + len(v.numeric_features),
        }
        for v in variants
    ])
    summary.to_csv(os.path.join(args.output_dir, "feature_variant_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n[Feature variants]")
    for _, row in summary.iterrows():
        print(
            f"{row['variant']}: categorical={row['n_categorical_features']}, "
            f"numeric={row['n_numeric_features']}, total={row['n_total_features']}"
        )

    if args.dry_run:
        print("\nDry run complete. No training was performed.")
        return

    rows = []
    for variant in variants:
        rows.append(run_variant(variant, train_df, valid_df, test_df, args.output_dir, args.fm_epochs))

    comp = pd.DataFrame(rows)
    ordered_cols = [
        "variant", "model", "mae", "rmse", "r2", "poisson_deviance",
        "exact_game_runs_accuracy", "within_1_run_accuracy", "within_2_runs_accuracy",
        "zero_vs_nonzero_accuracy_after_rounding", "true_mean_runs", "pred_mean_runs",
        "pred_rounded_mean_runs", "best_epoch", "best_valid_mse",
        "n_categorical_features", "n_numeric_features", "n_total_features",
    ]
    comp = comp[[c for c in ordered_cols if c in comp.columns] + [c for c in comp.columns if c not in ordered_cols]]
    comp.to_csv(os.path.join(args.output_dir, "fm_game_runs_regression_metrics.csv"), index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("FM team-game run-count regression comparison")
    print("=" * 80)
    print(comp.to_string(index=False))
    print(f"\nSaved: {os.path.join(args.output_dir, 'fm_game_runs_regression_metrics.csv')}")
    print("Done.")


if __name__ == "__main__":
    main()
