# -*- coding: utf-8 -*-
"""
LightGBM / Factorization Machine evaluation script for inning-level run prediction.

Targets evaluated in one run:
  1. Binary classification: NO_RUN / RUN_SCORED
     target column: target_run_scored
  2. 3-class classification: 0_RUN / 1_RUN / 2PLUS_RUN
     target column: target_run_class_3

Default execution:
  python evaluate_lightgbm_fm_inning_runs.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_inning_runs.csv"
実行例:
python evaluate_lightgbm_fm_inning_runs.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs.csv"
FMの学習回数を減らしたい場合
python evaluate_lightgbm_fm_inning_runs.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_inning_runs.csv" --fm_epochs 40
LightGBMだけを確認したい場合
python evaluate_lightgbm_fm_inning_runs.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_inning_runs.csv" --skip_fm

Outputs:
  output_lgbm_fm_inning_runs/
    model_comparison_metrics.csv
    target distributions
    classification reports
    confusion matrices
    LightGBM feature importance files

Notes:
- The split is time-based by game_date: train 80%, valid 10%, test 10% by unique dates.
- game_date, game_id, and target columns are not used as model features by default.
- game_id can be used with --use_game_id_feature, but this is usually not recommended
  because it acts mainly as an identifier rather than a generalizable feature.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except Exception:  # pragma: no cover
    torch = None
    nn = None
    Dataset = object
    DataLoader = None


SEED = 42
BINARY_LABELS = ["NO_RUN", "RUN_SCORED"]
THREE_CLASS_LABELS = ["0_RUN", "1_RUN", "2PLUS_RUN"]

CATEGORICAL_FEATURES_BASE = [
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
    "current_pitcher_id",
]

NUMERIC_FEATURES = [
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
]

TARGET_COLUMNS = [
    "target_runs_in_inning",
    "target_run_scored",
    "target_run_class_3",
]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def topk_accuracy(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    k = min(k, proba.shape[1])
    topk = np.argsort(proba, axis=1)[:, -k:]
    return float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))


def safe_metric(fn, *args, default: float = np.nan, **kwargs) -> float:
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return default


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, label_names: List[str]) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "top2_accuracy": topk_accuracy(y_true, proba, k=2),
        "top3_accuracy": topk_accuracy(y_true, proba, k=3),
    }
    metrics["log_loss"] = safe_metric(log_loss, y_true, proba, labels=list(range(len(label_names))))

    # Binary-only probability metrics for RUN_SCORED as positive class.
    if len(label_names) == 2:
        pos_idx = label_names.index("RUN_SCORED") if "RUN_SCORED" in label_names else 1
        y_true_binary = (y_true == pos_idx).astype(int)
        metrics["roc_auc"] = safe_metric(roc_auc_score, y_true_binary, proba[:, pos_idx])
        metrics["pr_auc"] = safe_metric(average_precision_score, y_true_binary, proba[:, pos_idx])
    else:
        metrics["roc_auc_ovr_macro"] = safe_metric(
            roc_auc_score, y_true, proba, multi_class="ovr", average="macro"
        )
    return metrics


def save_report_and_confusion(
    output_dir: str,
    prefix: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: List[str],
) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(label_names))),
        target_names=label_names,
        digits=6,
        zero_division=0,
    )
    with open(os.path.join(output_dir, f"{prefix}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(os.path.join(output_dir, f"{prefix}_confusion_matrix.csv"), encoding="utf-8-sig")


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

    train_df = df[df["game_date_dt"].isin(train_dates)].copy()
    valid_df = df[df["game_date_dt"].isin(valid_dates)].copy()
    test_df = df[df["game_date_dt"].isin(test_dates)].copy()
    return train_df, valid_df, test_df


@dataclass
class PreprocessArtifacts:
    categorical_features: List[str]
    numeric_features: List[str]
    cat_maps: Dict[str, Dict[str, int]]
    cat_cardinalities: List[int]
    numeric_medians: Dict[str, float]
    scaler: StandardScaler


def fit_preprocess(
    train_df: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
) -> PreprocessArtifacts:
    cat_maps: Dict[str, Dict[str, int]] = {}
    cat_cardinalities: List[int] = []
    for col in categorical_features:
        values = train_df[col].fillna("<NA>").astype(str).unique().tolist()
        mapping = {v: i + 1 for i, v in enumerate(sorted(values))}  # 0 = unknown
        cat_maps[col] = mapping
        cat_cardinalities.append(len(mapping) + 1)

    numeric_medians = {}
    for col in numeric_features:
        median = pd.to_numeric(train_df[col], errors="coerce").median()
        if pd.isna(median):
            median = 0.0
        numeric_medians[col] = float(median)

    X_num_train = make_numeric_matrix(train_df, numeric_features, numeric_medians)
    scaler = StandardScaler()
    scaler.fit(X_num_train)

    return PreprocessArtifacts(
        categorical_features=categorical_features,
        numeric_features=numeric_features,
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


def make_numeric_matrix(df: pd.DataFrame, numeric_features: List[str], numeric_medians: Dict[str, float]) -> np.ndarray:
    arrs = []
    for col in numeric_features:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(numeric_medians[col]).astype(float).values
        arrs.append(vals)
    if not arrs:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.vstack(arrs).T.astype(np.float32)


def transform_for_fm(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> Tuple[np.ndarray, np.ndarray]:
    X_cat = make_cat_matrix(df, artifacts)
    X_num_raw = make_numeric_matrix(df, artifacts.numeric_features, artifacts.numeric_medians)
    X_num = artifacts.scaler.transform(X_num_raw).astype(np.float32)
    return X_cat, X_num


def transform_for_lgbm(df: pd.DataFrame, artifacts: PreprocessArtifacts) -> pd.DataFrame:
    data = {}
    for col in artifacts.categorical_features:
        mapping = artifacts.cat_maps[col]
        data[col] = df[col].fillna("<NA>").astype(str).map(mapping).fillna(0).astype("int32")
    for col in artifacts.numeric_features:
        data[col] = pd.to_numeric(df[col], errors="coerce").fillna(artifacts.numeric_medians[col]).astype(float)
    return pd.DataFrame(data, index=df.index)


def make_target(df: pd.DataFrame, target_col: str, label_names: List[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(label_names)}
    y = df[target_col].map(mapping)
    if y.isna().any():
        missing = sorted(df.loc[y.isna(), target_col].astype(str).unique().tolist())
        raise ValueError(f"Unknown labels in {target_col}: {missing}")
    return y.astype(np.int64).values


# -----------------------------------------------------------------------------
# LightGBM
# -----------------------------------------------------------------------------

def run_lightgbm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    target_col: str,
    label_names: List[str],
    output_dir: str,
    prefix: str,
) -> Dict[str, float]:
    if lgb is None:
        raise ImportError("lightgbm is not installed. Please install lightgbm or use --skip_lightgbm.")

    X_train = transform_for_lgbm(train_df, artifacts)
    X_valid = transform_for_lgbm(valid_df, artifacts)
    X_test = transform_for_lgbm(test_df, artifacts)
    y_train = make_target(train_df, target_col, label_names)
    y_valid = make_target(valid_df, target_col, label_names)
    y_test = make_target(test_df, target_col, label_names)

    classes = np.arange(len(label_names))
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    sample_weight = np.array([cw[y] for y in y_train], dtype=np.float32)

    categorical_feature = artifacts.categorical_features

    if len(label_names) == 2:
        params = dict(
            objective="binary",
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            n_estimators=3000,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            eval_metric="binary_logloss",
            categorical_feature=categorical_feature,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        raw_proba = model.predict_proba(X_test)
        if raw_proba.shape[1] == 2:
            proba = raw_proba
        else:
            p1 = raw_proba.reshape(-1)
            proba = np.vstack([1 - p1, p1]).T
    else:
        params = dict(
            objective="multiclass",
            num_class=len(label_names),
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            n_estimators=3000,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=SEED,
            n_jobs=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
            categorical_feature=categorical_feature,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        proba = model.predict_proba(X_test)

    y_pred = np.argmax(proba, axis=1)
    metrics = compute_metrics(y_test, y_pred, proba, label_names)
    save_report_and_confusion(output_dir, prefix, y_test, y_pred, label_names)

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(os.path.join(output_dir, f"{prefix}_feature_importance.csv"), index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "game_date": test_df["game_date"].values,
        "game_id": test_df["game_id"].values,
        "team_id": test_df["team_id"].values,
        "inning": test_df["inning"].values,
        "top_bottom": test_df["top_bottom"].values,
        "y_true": [label_names[i] for i in y_test],
        "y_pred": [label_names[i] for i in y_pred],
    })
    for i, label in enumerate(label_names):
        pred_df[f"proba_{label}"] = proba[:, i]
    pred_df.to_csv(os.path.join(output_dir, f"{prefix}_test_predictions.csv"), index=False, encoding="utf-8-sig")

    return metrics


# -----------------------------------------------------------------------------
# FM
# -----------------------------------------------------------------------------

class TabularDataset(Dataset):
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray):
        self.X_cat = X_cat.astype(np.int64)
        self.X_num = X_num.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class MulticlassFM(nn.Module):
    def __init__(self, cat_cardinalities: List[int], num_numeric: int, num_classes: int, embed_dim: int = 16):
        super().__init__()
        self.num_cat = len(cat_cardinalities)
        self.num_numeric = num_numeric
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        offsets = np.cumsum([0] + cat_cardinalities[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        total_cat = int(sum(cat_cardinalities)) if cat_cardinalities else 1

        self.cat_linear = nn.Embedding(total_cat, num_classes)
        self.cat_embed = nn.Embedding(total_cat, embed_dim)
        self.num_linear = nn.Linear(num_numeric, num_classes) if num_numeric > 0 else None
        self.num_embed = nn.Parameter(torch.empty(num_numeric, embed_dim)) if num_numeric > 0 else None
        self.interaction_proj = nn.Linear(embed_dim, num_classes)
        self.bias = nn.Parameter(torch.zeros(num_classes))
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
        out = self.bias.unsqueeze(0).expand(batch_size, -1)

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
            summed_square = summed * summed
            square_summed = (V * V).sum(dim=1)
            interaction_vec = 0.5 * (summed_square - square_summed)
            out = out + self.interaction_proj(interaction_vec)
        return out


def train_fm_model(
    X_cat_train: np.ndarray,
    X_num_train: np.ndarray,
    y_train: np.ndarray,
    X_cat_valid: np.ndarray,
    X_num_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_cardinalities: List[int],
    num_classes: int,
    class_weights: np.ndarray,
    max_epochs: int = 80,
    batch_size: int = 512,
    patience: int = 8,
    lr: float = 1e-3,
) -> Tuple[nn.Module, Dict[str, float]]:
    if torch is None:
        raise ImportError("torch is not installed. Please install torch or use --skip_fm.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TabularDataset(X_cat_train, X_num_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TabularDataset(X_cat_valid, X_num_valid, y_valid), batch_size=batch_size, shuffle=False)

    model = MulticlassFM(cat_cardinalities, X_num_train.shape[1], num_classes, embed_dim=16).to(device)
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
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
            logits = model(xb_cat, xb_num)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)

        train_loss = total_loss / max(total_n, 1)
        valid_loss, valid_acc, valid_macro_f1 = evaluate_fm_loss(model, valid_loader, criterion, device)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"valid_loss={valid_loss:.5f} valid_acc={valid_acc:.5f} valid_macro_f1={valid_macro_f1:.5f}"
        )

        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping: best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = {"best_epoch": float(best_epoch), "best_valid_loss": float(best_valid_loss)}
    return model, info


def evaluate_fm_loss(model: nn.Module, loader: DataLoader, criterion, device) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    ys = []
    preds = []
    with torch.no_grad():
        for xb_cat, xb_num, yb in loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            yb = yb.to(device)
            logits = model(xb_cat, xb_num)
            loss = criterion(logits, yb)
            losses.append(loss.item() * len(yb))
            pred = torch.argmax(logits, dim=1)
            ys.append(yb.cpu().numpy())
            preds.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    loss_avg = float(np.sum(losses) / max(len(y_true), 1))
    return loss_avg, accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro", zero_division=0)


def predict_fm(model: nn.Module, X_cat: np.ndarray, X_num: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    dummy_y = np.zeros(len(X_cat), dtype=np.int64)
    loader = DataLoader(TabularDataset(X_cat, X_num, dummy_y), batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb_cat, xb_num, _ in loader:
            xb_cat = xb_cat.to(device)
            xb_num = xb_num.to(device)
            logits = model(xb_cat, xb_num)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.vstack(probs)


def run_fm(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    target_col: str,
    label_names: List[str],
    output_dir: str,
    prefix: str,
    max_epochs: int,
) -> Dict[str, float]:
    if torch is None:
        raise ImportError("torch is not installed. Please install torch or use --skip_fm.")

    X_cat_train, X_num_train = transform_for_fm(train_df, artifacts)
    X_cat_valid, X_num_valid = transform_for_fm(valid_df, artifacts)
    X_cat_test, X_num_test = transform_for_fm(test_df, artifacts)
    y_train = make_target(train_df, target_col, label_names)
    y_valid = make_target(valid_df, target_col, label_names)
    y_test = make_target(test_df, target_col, label_names)

    classes = np.arange(len(label_names))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train).astype(np.float32)
    print("[FM] class_weight:")
    for label, w in zip(label_names, class_weights):
        print(f"  {label}: {w:.6f}")

    model, info = train_fm_model(
        X_cat_train,
        X_num_train,
        y_train,
        X_cat_valid,
        X_num_valid,
        y_valid,
        artifacts.cat_cardinalities,
        len(label_names),
        class_weights,
        max_epochs=max_epochs,
    )
    proba = predict_fm(model, X_cat_test, X_num_test)
    y_pred = np.argmax(proba, axis=1)
    metrics = compute_metrics(y_test, y_pred, proba, label_names)
    metrics.update(info)
    save_report_and_confusion(output_dir, prefix, y_test, y_pred, label_names)

    pred_df = pd.DataFrame({
        "game_date": test_df["game_date"].values,
        "game_id": test_df["game_id"].values,
        "team_id": test_df["team_id"].values,
        "inning": test_df["inning"].values,
        "top_bottom": test_df["top_bottom"].values,
        "y_true": [label_names[i] for i in y_test],
        "y_pred": [label_names[i] for i in y_pred],
    })
    for i, label in enumerate(label_names):
        pred_df[f"proba_{label}"] = proba[:, i]
    pred_df.to_csv(os.path.join(output_dir, f"{prefix}_test_predictions.csv"), index=False, encoding="utf-8-sig")
    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_task(
    task_name: str,
    target_col: str,
    label_names: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    output_dir: str,
    skip_lightgbm: bool,
    skip_fm: bool,
    fm_epochs: int,
) -> List[Dict[str, object]]:
    print("\n" + "=" * 80)
    print(f"Task: {task_name} / target={target_col}")
    print("=" * 80)

    rows = []
    dist = pd.concat([train_df, valid_df, test_df])[target_col].value_counts().reindex(label_names, fill_value=0)
    dist.to_csv(os.path.join(output_dir, f"{task_name}_target_distribution.csv"), encoding="utf-8-sig")
    print("[Target distribution]")
    print(dist)

    if not skip_lightgbm:
        print("\n" + "-" * 80)
        print(f"LightGBM evaluation: {task_name}")
        print("-" * 80)
        metrics = run_lightgbm(
            train_df, valid_df, test_df, artifacts, target_col, label_names, output_dir, f"lightgbm_{task_name}"
        )
        print("[LightGBM Test Metrics]")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        rows.append({"task": task_name, "model": "lightgbm", **metrics})

    if not skip_fm:
        print("\n" + "-" * 80)
        print(f"Factorization Machine evaluation: {task_name}")
        print("-" * 80)
        metrics = run_fm(
            train_df, valid_df, test_df, artifacts, target_col, label_names, output_dir, f"fm_{task_name}", fm_epochs
        )
        print("[FM Test Metrics]")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        rows.append({"task": task_name, "model": "fm", **metrics})

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/train_inning_runs.csv")
    parser.add_argument("--output_dir", type=str, default="output_lgbm_fm_inning_runs")
    parser.add_argument("--use_game_id_feature", action="store_true", help="Use game_id as a categorical feature. Usually not recommended.")
    parser.add_argument("--skip_lightgbm", action="store_true")
    parser.add_argument("--skip_fm", action="store_true")
    parser.add_argument("--fm_epochs", type=int, default=80)
    args = parser.parse_args()

    set_seed(SEED)
    ensure_dir(args.output_dir)

    print("=" * 80)
    print("LightGBM / FM comparison for inning-level run prediction")
    print("=" * 80)
    print(f"csv_path: {args.csv_path}")
    print(f"output_dir: {args.output_dir}")

    df = pd.read_csv(args.csv_path)
    required = set(CATEGORICAL_FEATURES_BASE + NUMERIC_FEATURES + ["game_date", "game_id"] + TARGET_COLUMNS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    categorical_features = CATEGORICAL_FEATURES_BASE.copy()
    if args.use_game_id_feature:
        categorical_features = ["game_id"] + categorical_features

    numeric_features = NUMERIC_FEATURES.copy()

    train_df, valid_df, test_df = split_by_date(df)
    print("\n[Data split]")
    for name, part in [("all", df), ("train", train_df), ("valid", valid_df), ("test", test_df)]:
        dt = pd.to_datetime(part["game_date"], errors="coerce")
        if len(part) == 0:
            print(f"{name:5s}: {len(part):6d} rows")
        else:
            print(f"{name:5s}: {len(part):6d} rows, {dt.min().date()} ~ {dt.max().date()}, dates={dt.nunique()}")

    print("\n[Features]")
    print(f"categorical_features: {len(categorical_features)}")
    print(categorical_features)
    print(f"numeric_features: {len(numeric_features)}")
    print(numeric_features)
    print(f"total_features: {len(categorical_features) + len(numeric_features)}")

    pd.Series(categorical_features, name="categorical_features").to_csv(
        os.path.join(args.output_dir, "categorical_features.csv"), index=False, encoding="utf-8-sig"
    )
    pd.Series(numeric_features, name="numeric_features").to_csv(
        os.path.join(args.output_dir, "numeric_features.csv"), index=False, encoding="utf-8-sig"
    )

    artifacts = fit_preprocess(train_df, categorical_features, numeric_features)

    all_rows = []
    all_rows += run_task(
        "binary_run_scored",
        "target_run_scored",
        BINARY_LABELS,
        train_df,
        valid_df,
        test_df,
        artifacts,
        args.output_dir,
        args.skip_lightgbm,
        args.skip_fm,
        args.fm_epochs,
    )
    all_rows += run_task(
        "three_class_runs",
        "target_run_class_3",
        THREE_CLASS_LABELS,
        train_df,
        valid_df,
        test_df,
        artifacts,
        args.output_dir,
        args.skip_lightgbm,
        args.skip_fm,
        args.fm_epochs,
    )

    comp = pd.DataFrame(all_rows)
    comp.to_csv(os.path.join(args.output_dir, "model_comparison_metrics.csv"), index=False, encoding="utf-8-sig")
    print("\n" + "=" * 80)
    print("Model comparison")
    print("=" * 80)
    if len(comp):
        print(comp.to_string(index=False))
    print(f"\nSaved: {os.path.join(args.output_dir, 'model_comparison_metrics.csv')}")
    print("Done.")


if __name__ == "__main__":
    main()
