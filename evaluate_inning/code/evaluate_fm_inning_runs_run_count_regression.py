# -*- coding: utf-8 -*-
"""
FM-only run-count regression for inning-level run prediction.

Purpose:
- Instead of classifying innings into NO_RUN / RUN_SCORED or 0_RUN / 1_RUN / 2PLUS_RUN,
  directly predict target_runs_in_inning: the number of runs scored in the inning.
- Compare three FM feature settings using the same time-based train / valid / test split:
    1. all_29_features
    2. selected_19_features
    3. binned_features

Example:
  python evaluate_fm_inning_runs_run_count_regression.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
  python evaluate_fm_inning_runs_run_count_regression.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
Outputs:
  output_fm_inning_runs_run_count_regression/
    fm_run_count_regression_metrics.csv
    feature_variant_summary.csv
    each variant's feature list
    each variant's test predictions

Notes:
- The model is a Factorization Machine regressor trained with MSELoss.
- Predictions are continuous values. For exact run-count evaluation, predictions are clipped at 0
  and rounded to the nearest integer.
- This is a strict regression/count-prediction experiment, not threshold tuning.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_poisson_deviance
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


SEED = 42
TARGET_COL = "target_runs_in_inning"
TARGET_COLUMNS = ["target_runs_in_inning", "target_run_scored", "target_run_class_3"]

# -----------------------------------------------------------------------------
# Feature settings
# -----------------------------------------------------------------------------

CATEGORICAL_ALL_29 = [
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
    "current_pitcher_id",
    "batting_order_start_group",
]

NUMERIC_ALL_29 = [
    "inning",
    "top_bottom",
    "score_diff_before_inning",
    "team_runs_before_inning",
    "opponent_runs_before_inning",
    "team_avg_runs_7d",
    "opponent_avg_runs_7d",
    "pitcher_era_before_inning",
    "opponent_avg_runs_allowed_7d",
    "opponent_starter_era",
    "is_late_inning",
    "batting_order_start",
    "pitcher_bf_before_inning",
    "pitcher_runs_allowed_in_game",
    "is_top_order_start",
    "is_cleanup_start",
    "pitcher_is_starter",
    "pitcher_times_through_order",
    "team_run_scored_inning_rate_7d",
    "opponent_run_allowed_inning_rate_7d",
    "team_recent_2plus_inning_rate_7d",
    "opponent_recent_2plus_allowed_rate_7d",
]

CATEGORICAL_SELECTED_19 = CATEGORICAL_ALL_29.copy()
NUMERIC_SELECTED_19 = [
    "inning",
    "top_bottom",
    "score_diff_before_inning",
    "team_runs_before_inning",
    "opponent_runs_before_inning",
    "batting_order_start",
    "pitcher_bf_before_inning",
    "is_late_inning",
    "is_top_order_start",
    "is_cleanup_start",
    "pitcher_is_starter",
    "pitcher_times_through_order",
]

BINNED_FEATURES = [
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
    "current_pitcher_id",
    "inning_group",
    "top_bottom_cat",
    "score_state_bin",
    "team_runs_before_group",
    "opponent_runs_before_group",
    "batting_order_start_group",
    "pitcher_bf_group",
    "pitcher_role_bin",
    "times_through_order_group",
    "top_order_flag",
    "cleanup_flag",
    "team_avg_runs_7d_bin",
    "opponent_avg_runs_7d_bin",
    "opponent_avg_runs_allowed_7d_bin",
    "pitcher_era_before_inning_bin",
    "opponent_starter_era_bin",
    "team_run_scored_inning_rate_7d_bin",
    "opponent_run_allowed_inning_rate_7d_bin",
    "team_recent_2plus_inning_rate_7d_bin",
    "opponent_recent_2plus_allowed_rate_7d_bin",
]

FEATURE_VARIANTS: Dict[str, Dict[str, List[str]]] = {
    "all_29_features": {"categorical": CATEGORICAL_ALL_29, "numeric": NUMERIC_ALL_29},
    "selected_19_features": {"categorical": CATEGORICAL_SELECTED_19, "numeric": NUMERIC_SELECTED_19},
    "binned_features": {"categorical": BINNED_FEATURES, "numeric": []},
}


# -----------------------------------------------------------------------------
# Utilities and binning
# -----------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_num(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def fixed_rate_bin(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 0.0:
        return "zero"
    if x <= 0.15:
        return "low"
    if x <= 0.30:
        return "mid"
    return "high"


def fixed_runs_bin(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 0.0:
        return "zero"
    if x <= 2.5:
        return "low"
    if x <= 4.5:
        return "mid"
    return "high"


def fixed_era_bin(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 0.0:
        return "zero"
    if x <= 2.5:
        return "low"
    if x <= 4.0:
        return "mid"
    if x <= 6.0:
        return "high"
    return "very_high"


def score_state_bin(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= -4:
        return "behind_big"
    if x < 0:
        return "behind"
    if x == 0:
        return "tie"
    if x <= 3:
        return "lead"
    return "lead_big"


def small_count_bin(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 0:
        return "0"
    if x <= 2:
        return "1_2"
    if x <= 5:
        return "3_5"
    return "6plus"


def pitcher_bf_group(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 8:
        return "0_8"
    if x <= 17:
        return "9_17"
    if x <= 26:
        return "18_26"
    return "27plus"


def times_through_order_group(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 1:
        return "1st"
    if x <= 2:
        return "2nd"
    if x <= 3:
        return "3rd"
    return "4th_plus"


def inning_group(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 3:
        return "early"
    if x <= 6:
        return "middle"
    if x <= 9:
        return "late"
    return "extra"


def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["inning_group"] = to_num(df["inning"]).map(inning_group)
    df["top_bottom_cat"] = to_num(df["top_bottom"]).map(lambda x: "top" if int(x) == 0 else "bottom")
    df["score_state_bin"] = to_num(df["score_diff_before_inning"]).map(score_state_bin)
    df["team_runs_before_group"] = to_num(df["team_runs_before_inning"]).map(small_count_bin)
    df["opponent_runs_before_group"] = to_num(df["opponent_runs_before_inning"]).map(small_count_bin)
    df["pitcher_bf_group"] = to_num(df["pitcher_bf_before_inning"]).map(pitcher_bf_group)
    df["pitcher_role_bin"] = to_num(df["pitcher_is_starter"]).map(lambda x: "starter" if int(x) == 1 else "reliever")
    df["times_through_order_group"] = to_num(df["pitcher_times_through_order"]).map(times_through_order_group)
    df["top_order_flag"] = to_num(df["is_top_order_start"]).map(lambda x: "top_order" if int(x) == 1 else "not_top_order")
    df["cleanup_flag"] = to_num(df["is_cleanup_start"]).map(lambda x: "cleanup" if int(x) == 1 else "not_cleanup")
    df["team_avg_runs_7d_bin"] = to_num(df["team_avg_runs_7d"]).map(fixed_runs_bin)
    df["opponent_avg_runs_7d_bin"] = to_num(df["opponent_avg_runs_7d"]).map(fixed_runs_bin)
    df["opponent_avg_runs_allowed_7d_bin"] = to_num(df["opponent_avg_runs_allowed_7d"]).map(fixed_runs_bin)
    df["pitcher_era_before_inning_bin"] = to_num(df["pitcher_era_before_inning"]).map(fixed_era_bin)
    df["opponent_starter_era_bin"] = to_num(df["opponent_starter_era"]).map(fixed_era_bin)
    df["team_run_scored_inning_rate_7d_bin"] = to_num(df["team_run_scored_inning_rate_7d"]).map(fixed_rate_bin)
    df["opponent_run_allowed_inning_rate_7d_bin"] = to_num(df["opponent_run_allowed_inning_rate_7d"]).map(fixed_rate_bin)
    df["team_recent_2plus_inning_rate_7d_bin"] = to_num(df["team_recent_2plus_inning_rate_7d"]).map(fixed_rate_bin)
    df["opponent_recent_2plus_allowed_rate_7d_bin"] = to_num(df["opponent_recent_2plus_allowed_rate_7d"]).map(fixed_rate_bin)
    return df


def split_by_date(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["game_date_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["game_date_dt", "game_id", "inning", "top_bottom"]).reset_index(drop=True)
    unique_dates = np.array(sorted(df["game_date_dt"].dropna().unique()))
    n_dates = len(unique_dates)
    train_end = int(n_dates * 0.8)
    valid_end = int(n_dates * 0.9)
    train_dates = set(unique_dates[:train_end])
    valid_dates = set(unique_dates[train_end:valid_end])
    test_dates = set(unique_dates[valid_end:])
    return (
        df[df["game_date_dt"].isin(train_dates)].copy(),
        df[df["game_date_dt"].isin(valid_dates)].copy(),
        df[df["game_date_dt"].isin(test_dates)].copy(),
    )


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

@dataclass
class PreprocessArtifacts:
    categorical_features: List[str]
    numeric_features: List[str]
    cat_maps: Dict[str, Dict[str, int]]
    cat_cardinalities: List[int]
    numeric_medians: Dict[str, float]
    scaler: StandardScaler


def make_numeric_matrix(df: pd.DataFrame, numeric_features: List[str], numeric_medians: Dict[str, float]) -> np.ndarray:
    if not numeric_features:
        return np.zeros((len(df), 0), dtype=np.float32)
    arrs = []
    for col in numeric_features:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(numeric_medians[col]).astype(float).values
        arrs.append(vals)
    return np.vstack(arrs).T.astype(np.float32)


def fit_preprocess(train_df: pd.DataFrame, categorical_features: List[str], numeric_features: List[str]) -> PreprocessArtifacts:
    cat_maps: Dict[str, Dict[str, int]] = {}
    cat_cardinalities: List[int] = []
    for col in categorical_features:
        values = train_df[col].fillna("<NA>").astype(str).unique().tolist()
        mapping = {v: i + 1 for i, v in enumerate(sorted(values))}  # 0 = unknown
        cat_maps[col] = mapping
        cat_cardinalities.append(len(mapping) + 1)

    numeric_medians: Dict[str, float] = {}
    for col in numeric_features:
        median = pd.to_numeric(train_df[col], errors="coerce").median()
        if pd.isna(median):
            median = 0.0
        numeric_medians[col] = float(median)

    scaler = StandardScaler()
    if numeric_features:
        scaler.fit(make_numeric_matrix(train_df, numeric_features, numeric_medians))
    else:
        scaler.fit(np.zeros((len(train_df), 1), dtype=np.float32))

    return PreprocessArtifacts(categorical_features, numeric_features, cat_maps, cat_cardinalities, numeric_medians, scaler)


def make_cat_matrix(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> np.ndarray:
    if not artifacts.categorical_features:
        return np.zeros((len(df), 0), dtype=np.int64)
    cols = []
    for col in artifacts.categorical_features:
        mapping = artifacts.cat_maps[col]
        vals = df[col].fillna("<NA>").astype(str).map(mapping).fillna(0).astype(np.int64).values
        cols.append(vals)
    return np.vstack(cols).T.astype(np.int64)


def transform_for_fm(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> Tuple[np.ndarray, np.ndarray]:
    X_cat = make_cat_matrix(df, artifacts)
    X_num_raw = make_numeric_matrix(df, artifacts.numeric_features, artifacts.numeric_medians)
    if artifacts.numeric_features:
        X_num = artifacts.scaler.transform(X_num_raw).astype(np.float32)
    else:
        X_num = np.zeros((len(df), 0), dtype=np.float32)
    return X_cat, X_num


# -----------------------------------------------------------------------------
# FM regression model
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


class FMRegressor(nn.Module):
    def __init__(self, cat_cardinalities: List[int], num_numeric: int, embed_dim: int = 16):
        super().__init__()
        self.num_cat = len(cat_cardinalities)
        self.num_numeric = num_numeric
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

    def reset_parameters(self):
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
        out = self.bias.expand(batch_size, 1)
        embed_fields = []
        if self.num_cat > 0:
            x_cat_off = x_cat + self.offsets.unsqueeze(0)
            out = out + self.cat_linear(x_cat_off).sum(dim=1)
            embed_fields.append(self.cat_embed(x_cat_off))
        if self.num_numeric > 0:
            out = out + self.num_linear(x_num)
            embed_fields.append(x_num.unsqueeze(-1) * self.num_embed.unsqueeze(0))
        if embed_fields:
            V = torch.cat(embed_fields, dim=1)
            V = self.dropout(V)
            summed = V.sum(dim=1)
            interaction_vec = 0.5 * (summed * summed - (V * V).sum(dim=1))
            out = out + self.interaction_proj(interaction_vec)
        return out.squeeze(1)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion, device) -> float:
    model.eval()
    losses = []
    n = 0
    with torch.no_grad():
        for xb_cat, xb_num, yb in loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            yb = yb.to(device)
            pred = model(xb_cat, xb_num)
            loss = criterion(pred, yb)
            losses.append(loss.item() * len(yb))
            n += len(yb)
    return float(np.sum(losses) / max(n, 1))


def train_fm_regressor(
    X_cat_train: np.ndarray,
    X_num_train: np.ndarray,
    y_train: np.ndarray,
    X_cat_valid: np.ndarray,
    X_num_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_cardinalities: List[int],
    max_epochs: int = 80,
    batch_size: int = 512,
    patience: int = 8,
    lr: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(RegressionDataset(X_cat_train, X_num_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(RegressionDataset(X_cat_valid, X_num_valid, y_valid), batch_size=batch_size, shuffle=False)
    model = FMRegressor(cat_cardinalities, X_num_train.shape[1], embed_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_state = None
    best_valid_loss = float("inf")
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
        train_loss = total_loss / max(total_n, 1)
        valid_loss = evaluate_loss(model, valid_loader, criterion, device)
        print(f"epoch={epoch:03d} train_mse={train_loss:.5f} valid_mse={valid_loss:.5f}")
        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping: best_epoch={best_epoch}, best_valid_mse={best_valid_loss:.6f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": float(best_epoch), "best_valid_mse": float(best_valid_loss)}


def predict_regression(model: nn.Module, X_cat: np.ndarray, X_num: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    dummy_y = np.zeros(len(X_cat), dtype=np.float32)
    loader = DataLoader(RegressionDataset(X_cat, X_num, dummy_y), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb_cat, xb_num, _ in loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            pred = model(xb_cat, xb_num)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def compute_regression_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray) -> Dict[str, float]:
    y_pred_clip = np.clip(y_pred_raw, 0, None)
    y_pred_round = np.rint(y_pred_clip).astype(int)
    y_true_int = y_true.astype(int)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clip)))
    mae = float(mean_absolute_error(y_true, y_pred_clip))
    r2 = float(r2_score(y_true, y_pred_clip))
    exact_acc = float(np.mean(y_pred_round == y_true_int))
    within1_acc = float(np.mean(np.abs(y_pred_round - y_true_int) <= 1))
    # Poisson deviance needs strictly positive predictions; add small epsilon.
    try:
        poisson_deviance = float(mean_poisson_deviance(y_true, np.maximum(y_pred_clip, 1e-6)))
    except Exception:
        poisson_deviance = float("nan")
    zero_vs_nonzero_acc = float(np.mean((y_pred_round >= 1) == (y_true_int >= 1)))
    pred_mean = float(np.mean(y_pred_clip))
    true_mean = float(np.mean(y_true))
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "poisson_deviance": poisson_deviance,
        "exact_run_accuracy": exact_acc,
        "within_1_run_accuracy": within1_acc,
        "zero_vs_nonzero_accuracy_after_rounding": zero_vs_nonzero_acc,
        "true_mean_runs": true_mean,
        "pred_mean_runs": pred_mean,
    }


def run_variant(
    variant_name: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
    output_dir: str,
    fm_epochs: int,
) -> Dict[str, float]:
    print("\n" + "-" * 80)
    print(f"FM regression: variant={variant_name}")
    print("-" * 80)
    print(f"categorical_features ({len(categorical_features)}): {categorical_features}")
    print(f"numeric_features ({len(numeric_features)}): {numeric_features}")

    artifacts = fit_preprocess(train_df, categorical_features, numeric_features)
    X_cat_train, X_num_train = transform_for_fm(train_df, artifacts)
    X_cat_valid, X_num_valid = transform_for_fm(valid_df, artifacts)
    X_cat_test, X_num_test = transform_for_fm(test_df, artifacts)

    y_train = pd.to_numeric(train_df[TARGET_COL], errors="coerce").fillna(0).astype(float).values
    y_valid = pd.to_numeric(valid_df[TARGET_COL], errors="coerce").fillna(0).astype(float).values
    y_test = pd.to_numeric(test_df[TARGET_COL], errors="coerce").fillna(0).astype(float).values

    model, info = train_fm_regressor(
        X_cat_train,
        X_num_train,
        y_train,
        X_cat_valid,
        X_num_valid,
        y_valid,
        artifacts.cat_cardinalities,
        max_epochs=fm_epochs,
    )
    y_pred_raw = predict_regression(model, X_cat_test, X_num_test)
    y_pred_clip = np.clip(y_pred_raw, 0, None)
    y_pred_round = np.rint(y_pred_clip).astype(int)

    metrics = compute_regression_metrics(y_test, y_pred_raw)
    metrics.update(info)
    metrics.update({
        "n_categorical_features": len(categorical_features),
        "n_numeric_features": len(numeric_features),
        "n_total_features": len(categorical_features) + len(numeric_features),
    })

    pred_df = pd.DataFrame({
        "game_date": test_df["game_date"].values,
        "game_id": test_df["game_id"].values,
        "team_id": test_df["team_id"].values,
        "opponent_team_id": test_df["opponent_team_id"].values,
        "inning": test_df["inning"].values,
        "top_bottom": test_df["top_bottom"].values,
        "y_true_runs": y_test.astype(int),
        "y_pred_raw": y_pred_raw,
        "y_pred_clipped": y_pred_clip,
        "y_pred_rounded": y_pred_round,
        "absolute_error_rounded": np.abs(y_pred_round - y_test.astype(int)),
    })
    pred_df.to_csv(os.path.join(output_dir, f"{variant_name}_test_predictions.csv"), index=False, encoding="utf-8-sig")

    # Save rounded prediction distribution to compare with true run distribution.
    max_run = int(max(np.max(y_test), np.max(y_pred_round))) if len(y_test) else 0
    dist_df = pd.DataFrame({
        "runs": list(range(max_run + 1)),
        "true_count": [int(np.sum(y_test.astype(int) == r)) for r in range(max_run + 1)],
        "pred_rounded_count": [int(np.sum(y_pred_round == r)) for r in range(max_run + 1)],
    })
    dist_df.to_csv(os.path.join(output_dir, f"{variant_name}_rounded_prediction_distribution.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(output_dir, f"{variant_name}_features.txt"), "w", encoding="utf-8") as f:
        f.write("[categorical_features]\n")
        for col in categorical_features:
            f.write(f"{col}\n")
        f.write("\n[numeric_features]\n")
        for col in numeric_features:
            f.write(f"{col}\n")

    print("[FM Regression Test Metrics]")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/train_inning_runs_added_9_features.csv")
    parser.add_argument("--output_dir", type=str, default="output_fm_inning_runs_run_count_regression")
    parser.add_argument("--fm_epochs", type=int, default=80)
    parser.add_argument("--dry_run", action="store_true", help="Only validate data, split, and feature variants; do not train models.")
    args = parser.parse_args()

    set_seed(SEED)
    ensure_dir(args.output_dir)

    print("=" * 80)
    print("FM run-count regression for inning-level run prediction")
    print("=" * 80)
    print(f"csv_path: {args.csv_path}")
    print(f"output_dir: {args.output_dir}")

    df = pd.read_csv(args.csv_path)
    df = add_binned_features(df)

    required_base = set(["game_date", "game_id", TARGET_COL] + TARGET_COLUMNS)
    all_features = set()
    for cfg in FEATURE_VARIANTS.values():
        all_features.update(cfg["categorical"])
        all_features.update(cfg["numeric"])
    required = sorted(required_base | all_features)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    train_df, valid_df, test_df = split_by_date(df)
    print("\n[Data split]")
    for name, part in [("all", df), ("train", train_df), ("valid", valid_df), ("test", test_df)]:
        dt = pd.to_datetime(part["game_date"], errors="coerce")
        print(f"{name:5s}: {len(part):6d} rows, {dt.min().date()} ~ {dt.max().date()}, dates={dt.nunique()}")

    print("\n[Target distribution: target_runs_in_inning]")
    target_dist = df[TARGET_COL].value_counts().sort_index()
    print(target_dist)
    target_dist.to_csv(os.path.join(args.output_dir, "target_runs_in_inning_distribution.csv"), encoding="utf-8-sig")

    print("\n[Feature variants]")
    variant_rows = []
    for name, cfg in FEATURE_VARIANTS.items():
        n_cat = len(cfg["categorical"])
        n_num = len(cfg["numeric"])
        print(f"{name}: categorical={n_cat}, numeric={n_num}, total={n_cat + n_num}")
        variant_rows.append({"variant": name, "n_categorical_features": n_cat, "n_numeric_features": n_num, "n_total_features": n_cat + n_num})
    pd.DataFrame(variant_rows).to_csv(os.path.join(args.output_dir, "feature_variant_summary.csv"), index=False, encoding="utf-8-sig")

    if args.dry_run:
        print("\nDry run complete. No models were trained.")
        return

    results = []
    for variant_name, cfg in FEATURE_VARIANTS.items():
        metrics = run_variant(
            variant_name,
            train_df,
            valid_df,
            test_df,
            cfg["categorical"],
            cfg["numeric"],
            args.output_dir,
            args.fm_epochs,
        )
        results.append({"variant": variant_name, "model": "fm_regression", **metrics})

    comp = pd.DataFrame(results)
    comp.to_csv(os.path.join(args.output_dir, "fm_run_count_regression_metrics.csv"), index=False, encoding="utf-8-sig")
    print("\n" + "=" * 80)
    print("FM run-count regression comparison")
    print("=" * 80)
    print(comp.to_string(index=False))
    print(f"\nSaved: {os.path.join(args.output_dir, 'fm_run_count_regression_metrics.csv')}")
    print("Done.")


if __name__ == "__main__":
    main()
