
# -*- coding: utf-8 -*-
"""
Main task: inning-level run occurrence prediction.

目的:
- 中間発表のメインタスクとして、イニング開始時点で分かる情報から
  そのイニングで得点が入るかを予測する。
- 目的変数: target_run_scored = NO_RUN / RUN_SCORED
- 入力: イニング開始時点の打順、相手投手、チーム状態、球場、直近得点傾向
- 同じ train / valid / test 分割で LightGBM と FM を比較する。

実行例:
python evaluate_main_task_inning_run_scored.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
python evaluate_main_task_inning_run_scored.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
軽い確認:
python evaluate_main_task_inning_run_scored.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" --dry_run

必要ライブラリ:
pip install pandas numpy scikit-learn lightgbm torch tqdm

注意:
- リーク防止のため、イニング中に確定する情報や試合中累積に強く依存する列は使わない。
- score_diff_before_inning, team_runs_before_inning, opponent_runs_before_inning は
  イニング開始時点では分かるが、今回のメイン主張を「打順・投手・チーム状態・直近傾向」に絞るため、標準では除外する。
  必要なら --include_score_context で追加可能。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


LABELS = ["NO_RUN", "RUN_SCORED"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
POS_LABEL = "RUN_SCORED"
POS_ID = LABEL_TO_ID[POS_LABEL]

# ============================================================
# Feature definition for the main task
# ============================================================
# 入力方針:
# - イニング開始時点の打順
# - 相手投手
# - チーム状態
# - 球場
# - 直近得点傾向

BASE_CATEGORICAL_FEATURES = [
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
    "current_pitcher_id",
    "batting_order_start_group",
]

BASE_NUMERIC_FEATURES = [
    # イニング開始時点の基本状況
    "inning",
    "top_bottom",
    "is_late_inning",
    # イニング開始時点の打順
    "batting_order_start",
    "is_top_order_start",
    "is_cleanup_start",
    # 相手投手の状態
    "pitcher_is_starter",
    "pitcher_times_through_order",
    "pitcher_era_before_inning",
    "opponent_starter_era",
    # チーム状態・直近得点傾向
    "team_avg_runs_7d",
    "opponent_avg_runs_7d",
    "opponent_avg_runs_allowed_7d",
    "team_run_scored_inning_rate_7d",
    "opponent_run_allowed_inning_rate_7d",
    "team_recent_2plus_inning_rate_7d",
    "opponent_recent_2plus_allowed_rate_7d",
]

# イニング開始時点で分かるが、主張を絞るため標準では除外する列。
SCORE_CONTEXT_NUMERIC_FEATURES = [
    "score_diff_before_inning",
    "team_runs_before_inning",
    "opponent_runs_before_inning",
]

IGNORE_COLUMNS = [
    "game_date",
    "game_id",
    "target_runs_in_inning",
    "target_run_scored",
    "target_run_class_3",
]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class FeatureConfig:
    name: str
    categorical_features: List[str]
    numeric_features: List[str]

    @property
    def total_features(self) -> int:
        return len(self.categorical_features) + len(self.numeric_features)


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_top_bottom(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if s in ["表", "top", "Top"]:
        return 0.0
    if s in ["裏", "bottom", "Bottom"]:
        return 1.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def prepare_dataframe(csv_path: Path, feature_config: FeatureConfig) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()

    required = set(["game_date"] + feature_config.categorical_features + feature_config.numeric_features)
    if "target_run_scored" not in df.columns and "target_runs_in_inning" not in df.columns:
        required.add("target_run_scored")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"必要な列がCSVに存在しません: {missing}")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    if "target_run_scored" in df.columns:
        df["target_label"] = df["target_run_scored"].astype(str).str.strip()
    else:
        df["target_label"] = np.where(pd.to_numeric(df["target_runs_in_inning"], errors="coerce").fillna(0) > 0, "RUN_SCORED", "NO_RUN")

    unknown_labels = sorted(set(df["target_label"].dropna().unique()) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"target_run_scored に想定外のラベルがあります: {unknown_labels}")

    df["target"] = df["target_label"].map(LABEL_TO_ID).astype(int)

    if "top_bottom" in df.columns:
        df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)

    for col in feature_config.numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in feature_config.categorical_features:
        df[col] = df[col].fillna("<NA>").astype(str)

    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def split_by_game_date(df: pd.DataFrame, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> SplitData:
    unique_dates = sorted(df["game_date"].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("game_date のユニーク数が少なすぎるため、時系列分割できません。")

    n_dates = len(unique_dates)
    train_end = int(n_dates * train_ratio)
    valid_end = int(n_dates * (train_ratio + valid_ratio))

    train_dates = set(unique_dates[:train_end])
    valid_dates = set(unique_dates[train_end:valid_end])
    test_dates = set(unique_dates[valid_end:])

    train_df = df[df["game_date"].isin(train_dates)].copy()
    valid_df = df[df["game_date"].isin(valid_dates)].copy()
    test_df = df[df["game_date"].isin(test_dates)].copy()

    return SplitData(train_df=train_df, valid_df=valid_df, test_df=test_df)


def make_class_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.arange(len(LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return weights.astype(np.float32)


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "no_run_precision": float(per_class_p[0]),
        "no_run_recall": float(per_class_r[0]),
        "no_run_f1": float(per_class_f1[0]),
        "run_scored_precision": float(per_class_p[1]),
        "run_scored_recall": float(per_class_r[1]),
        "run_scored_f1": float(per_class_f1[1]),
        "run_scored_pred_count": int(np.sum(y_pred == POS_ID)),
    }

    if y_prob is not None:
        prob_pos = y_prob[:, POS_ID]
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        except Exception:
            metrics["log_loss"] = np.nan
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
        except Exception:
            metrics["roc_auc"] = np.nan
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, prob_pos))
        except Exception:
            metrics["pr_auc"] = np.nan

    return metrics


def save_binary_outputs(
    output_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics["model"] = model_name

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=LABELS,
        zero_division=0,
        digits=6,
    )
    (output_dir / f"{model_name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        output_dir / f"{model_name}_confusion_matrix.csv", encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame({
        "true_id": y_true,
        "pred_id": y_pred,
        "true_label": [ID_TO_LABEL[int(x)] for x in y_true],
        "pred_label": [ID_TO_LABEL[int(x)] for x in y_pred],
    })
    if y_prob is not None:
        for i, label in enumerate(LABELS):
            pred_df[f"prob_{label}"] = y_prob[:, i]
    pred_df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False, encoding="utf-8-sig")

    return metrics


# ============================================================
# LightGBM
# ============================================================

def train_evaluate_lightgbm(
    split: SplitData,
    feature_config: FeatureConfig,
    output_dir: Path,
    seed: int = 42,
) -> Optional[Dict[str, float]]:
    try:
        import lightgbm as lgb
    except ImportError:
        print("[WARN] lightgbm がインストールされていないため、LightGBM評価をスキップします。")
        return None

    print("\n" + "-" * 80)
    print("LightGBM evaluation: main_binary_run_scored")
    print("-" * 80)

    cat_features = feature_config.categorical_features
    num_features = feature_config.numeric_features
    feature_cols = cat_features + num_features

    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()

    for col in cat_features:
        categories = pd.Index(pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).astype(str).unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train_df[col] = train_df[col].astype(dtype)
        valid_df[col] = valid_df[col].astype(dtype)
        test_df[col] = test_df[col].astype(dtype)

    medians = train_df[num_features].median(numeric_only=True)
    train_df[num_features] = train_df[num_features].fillna(medians)
    valid_df[num_features] = valid_df[num_features].fillna(medians)
    test_df[num_features] = test_df[num_features].fillna(medians)

    X_train = train_df[feature_cols]
    y_train = train_df["target"].values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["target"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target"].values

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        categorical_feature=cat_features,
        callbacks=callbacks,
    )

    prob_pos = model.predict_proba(X_test)[:, 1]
    y_prob = np.column_stack([1.0 - prob_pos, prob_pos])
    y_pred = (prob_pos >= 0.5).astype(np.int64)

    metrics = save_binary_outputs(output_dir, "lightgbm", y_test, y_pred, y_prob)
    metrics.update({
        "n_categorical_features": len(cat_features),
        "n_numeric_features": len(num_features),
        "n_total_features": len(feature_cols),
    })

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(output_dir / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")

    print("[LightGBM Test Metrics]")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")

    return metrics


# ============================================================
# FM
# ============================================================

class FeatureEncoder:
    def __init__(self, categorical_features: List[str], numeric_features: List[str]):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.numeric_median: Dict[str, float] = {}
        self.scaler: Optional[StandardScaler] = None

    def fit(self, df: pd.DataFrame) -> None:
        for col in self.categorical_features:
            values = sorted(df[col].fillna("<NA>").astype(str).unique().tolist())
            self.category_maps[col] = {v: i + 1 for i, v in enumerate(values)}  # 0 is unknown

        if self.numeric_features:
            X = []
            for col in self.numeric_features:
                s = pd.to_numeric(df[col], errors="coerce")
                med = float(s.median()) if not pd.isna(s.median()) else 0.0
                self.numeric_median[col] = med
                X.append(s.fillna(med).astype(float).values)
            X = np.stack(X, axis=1)
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        else:
            self.scaler = None

    def transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col in self.categorical_features:
            mp = self.category_maps[col]
            arr = df[col].fillna("<NA>").astype(str).map(lambda x: mp.get(x, 0)).astype(np.int64).values
            arrays.append(arr)
        if not arrays:
            return np.zeros((len(df), 0), dtype=np.int64)
        return np.stack(arrays, axis=1)

    def transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric_features:
            return np.zeros((len(df), 0), dtype=np.float32)
        X = []
        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_median[col])
            X.append(s.astype(float).values)
        X = np.stack(X, axis=1)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X.astype(np.float32)

    def categorical_cardinalities(self) -> List[int]:
        return [len(self.category_maps[col]) + 1 for col in self.categorical_features]

    def save_cardinality(self, output_path: Path) -> None:
        rows = []
        for col in self.categorical_features:
            rows.append({"feature": col, "cardinality_in_train_plus_unknown": len(self.category_maps[col]) + 1})
        pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")


class FMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: FeatureEncoder):
        self.x_cat = encoder.transform_categorical(df)
        self.x_num = encoder.transform_numeric(df)
        self.y = df["target"].astype(np.int64).values

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
        }


class FactorizationMachineClassifier(nn.Module):
    def __init__(
        self,
        cat_cardinalities: List[int],
        num_numeric: int,
        num_classes: int,
        embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_cat = len(cat_cardinalities)
        self.num_numeric = num_numeric
        self.num_fields = self.num_cat + self.num_numeric
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.cat_embeddings = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cardinalities])
        self.num_embeddings = nn.Parameter(torch.randn(num_numeric, embed_dim) * 0.01)

        self.cat_linear = nn.ModuleList([nn.Embedding(c, num_classes) for c in cat_cardinalities])
        self.num_linear = nn.Parameter(torch.randn(num_numeric, num_classes) * 0.01)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.interaction_linear = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.cat_linear:
            nn.init.zeros_(emb.weight)
        if self.num_numeric > 0:
            nn.init.xavier_uniform_(self.num_embeddings)
            nn.init.xavier_uniform_(self.num_linear)
        nn.init.xavier_uniform_(self.interaction_linear.weight)
        nn.init.zeros_(self.interaction_linear.bias)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size = x_cat.size(0)
        embeddings = []

        for i, emb in enumerate(self.cat_embeddings):
            embeddings.append(emb(x_cat[:, i]))

        if self.num_numeric > 0:
            num_embs = x_num.unsqueeze(-1) * self.num_embeddings.unsqueeze(0)
            for i in range(self.num_numeric):
                embeddings.append(num_embs[:, i, :])

        if embeddings:
            all_embs = torch.stack(embeddings, dim=1)
            all_embs = self.dropout(all_embs)
            summed = torch.sum(all_embs, dim=1)
            interaction = 0.5 * (summed * summed - torch.sum(all_embs * all_embs, dim=1))
            interaction_logits = self.interaction_linear(interaction)
        else:
            interaction_logits = torch.zeros(batch_size, self.num_classes, device=x_cat.device)

        linear_logits = self.bias.unsqueeze(0).expand(batch_size, -1)
        for i, lin in enumerate(self.cat_linear):
            linear_logits = linear_logits + lin(x_cat[:, i])
        if self.num_numeric > 0:
            linear_logits = linear_logits + torch.einsum("bn,nc->bc", x_num, self.num_linear)

        return linear_logits + interaction_logits



def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip: float = 5.0) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="FM train", leave=False):
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x_cat, x_num)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(torch.argmax(logits.detach(), dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred)


@torch.no_grad()
def evaluate_fm(model, loader, criterion, device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    for batch in tqdm(loader, desc="FM eval", leave=False):
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        y = batch["y"].to(device)

        logits = model(x_cat, x_num)
        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)

        total_loss += loss.item() * y.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(torch.argmax(prob, dim=1).cpu().numpy())
        y_prob.append(prob.cpu().numpy())

    return total_loss / len(loader.dataset), np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(y_prob)


def train_evaluate_fm(
    split: SplitData,
    feature_config: FeatureConfig,
    output_dir: Path,
    seed: int = 42,
    batch_size: int = 512,
    epochs: int = 50,
    patience: int = 8,
    embed_dim: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> Dict[str, float]:
    print("\n" + "-" * 80)
    print("Factorization Machine evaluation: main_binary_run_scored")
    print("-" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    encoder = FeatureEncoder(feature_config.categorical_features, feature_config.numeric_features)
    encoder.fit(split.train_df)
    encoder.save_cardinality(output_dir / "fm_categorical_cardinality.csv")

    train_ds = FMDataset(split.train_df, encoder)
    valid_ds = FMDataset(split.valid_df, encoder)
    test_ds = FMDataset(split.test_df, encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FactorizationMachineClassifier(
        cat_cardinalities=encoder.categorical_cardinalities(),
        num_numeric=len(feature_config.numeric_features),
        num_classes=len(LABELS),
        embed_dim=embed_dim,
        dropout=0.1,
    ).to(device)

    y_train_np = split.train_df["target"].values
    class_weights_np = make_class_weights(y_train_np)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    print("[FM] class_weight:")
    for label, weight in zip(LABELS, class_weights_np):
        print(f"  {label}: {weight:.6f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, y_valid, pred_valid, prob_valid = evaluate_fm(model, valid_loader, criterion, device)
        valid_metrics = compute_binary_metrics(y_valid, pred_valid, prob_valid)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "valid_loss": valid_loss,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} "
            f"valid_loss={valid_loss:.5f} "
            f"valid_acc={valid_metrics['accuracy']:.5f} "
            f"valid_macro_f1={valid_metrics['macro_f1']:.5f} "
            f"valid_run_recall={valid_metrics['run_scored_recall']:.5f}"
        )

        if valid_loss < best_valid_loss - 1e-5:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping: best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.6f}")
            break

    pd.DataFrame(history).to_csv(output_dir / "fm_history.csv", index=False, encoding="utf-8-sig")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "categorical_features": feature_config.categorical_features,
            "numeric_features": feature_config.numeric_features,
            "labels": LABELS,
            "class_weights": class_weights_np.tolist(),
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
        },
        output_dir / "fm_best_model.pt",
    )

    test_loss, y_test, y_pred, y_prob = evaluate_fm(model, test_loader, criterion, device)
    metrics = save_binary_outputs(output_dir, "fm", y_test, y_pred, y_prob)
    metrics.update({
        "test_loss": float(test_loss),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "n_categorical_features": len(feature_config.categorical_features),
        "n_numeric_features": len(feature_config.numeric_features),
        "n_total_features": feature_config.total_features,
    })

    print("[FM Test Metrics]")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")

    return metrics


# ============================================================
# Main
# ============================================================

def build_feature_config(include_score_context: bool = False) -> FeatureConfig:
    numeric_features = list(BASE_NUMERIC_FEATURES)
    if include_score_context:
        numeric_features += SCORE_CONTEXT_NUMERIC_FEATURES
    return FeatureConfig(
        name="main_inning_run_scored_features",
        categorical_features=list(BASE_CATEGORICAL_FEATURES),
        numeric_features=numeric_features,
    )


def save_feature_files(output_dir: Path, feature_config: FeatureConfig) -> None:
    rows = []
    for col in feature_config.categorical_features:
        rows.append({"feature": col, "feature_type": "categorical"})
    for col in feature_config.numeric_features:
        rows.append({"feature": col, "feature_type": "numeric"})
    pd.DataFrame(rows).to_csv(output_dir / "main_task_features.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "main_inning_run_scored",
                "target": "target_run_scored / NO_RUN vs RUN_SCORED",
                "feature_policy": "inning-start information: batting order, opponent pitcher, team state, stadium, recent scoring tendency",
                "categorical_features": feature_config.categorical_features,
                "numeric_features": feature_config.numeric_features,
                "excluded_as_default": SCORE_CONTEXT_NUMERIC_FEATURES,
                "labels": LABELS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/train_inning_runs_added_9_features.csv",
        help="イニング得点予測CSVのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_main_task_inning_run_scored",
        help="出力先ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fm_epochs", type=int, default=50)
    parser.add_argument("--fm_batch_size", type=int, default=512)
    parser.add_argument("--fm_embed_dim", type=int, default=32)
    parser.add_argument("--fm_lr", type=float, default=1e-3)
    parser.add_argument("--fm_patience", type=int, default=8)
    parser.add_argument("--skip_lightgbm", action="store_true")
    parser.add_argument("--skip_fm", action="store_true")
    parser.add_argument(
        "--include_score_context",
        action="store_true",
        help="score_diff_before_inning, team_runs_before_inning, opponent_runs_before_inning を追加する",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="データ分割と特徴量確認だけ行い、学習はしない",
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = build_feature_config(include_score_context=args.include_score_context)

    print("=" * 80)
    print("Main task: inning-level run occurrence prediction")
    print("=" * 80)
    print(f"csv_path: {csv_path}")
    print(f"output_dir: {output_dir}")
    print("target: NO_RUN / RUN_SCORED")
    print(f"include_score_context: {args.include_score_context}")

    df = prepare_dataframe(csv_path, feature_config)
    split = split_by_game_date(df)

    save_feature_files(output_dir, feature_config)

    print("\n[Data split]")
    print(f"all  : {len(df):>6} rows, {df['game_date'].min().date()} ~ {df['game_date'].max().date()}, dates={df['game_date'].nunique()}")
    print(f"train: {len(split.train_df):>6} rows, {split.train_df['game_date'].min().date()} ~ {split.train_df['game_date'].max().date()}, dates={split.train_df['game_date'].nunique()}")
    print(f"valid: {len(split.valid_df):>6} rows, {split.valid_df['game_date'].min().date()} ~ {split.valid_df['game_date'].max().date()}, dates={split.valid_df['game_date'].nunique()}")
    print(f"test : {len(split.test_df):>6} rows, {split.test_df['game_date'].min().date()} ~ {split.test_df['game_date'].max().date()}, dates={split.test_df['game_date'].nunique()}")

    print("\n[Features]")
    print(f"categorical_features: {len(feature_config.categorical_features)}")
    print(feature_config.categorical_features)
    print(f"numeric_features: {len(feature_config.numeric_features)}")
    print(feature_config.numeric_features)
    print(f"total_features: {feature_config.total_features}")

    print("\n[Target distribution]")
    dist = df["target_label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print(dist)
    dist.to_csv(output_dir / "target_distribution.csv", encoding="utf-8-sig")

    if args.dry_run:
        print("\nDry run completed. No training executed.")
        return

    all_metrics = []

    if not args.skip_lightgbm:
        lgb_metrics = train_evaluate_lightgbm(split, feature_config, output_dir, seed=args.seed)
        if lgb_metrics is not None:
            all_metrics.append(lgb_metrics)

    if not args.skip_fm:
        fm_metrics = train_evaluate_fm(
            split,
            feature_config,
            output_dir,
            seed=args.seed,
            batch_size=args.fm_batch_size,
            epochs=args.fm_epochs,
            patience=args.fm_patience,
            embed_dim=args.fm_embed_dim,
            lr=args.fm_lr,
        )
        all_metrics.append(fm_metrics)

    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        cols = ["model"] + [c for c in comparison_df.columns if c != "model"]
        comparison_df = comparison_df[cols]
        comparison_df.to_csv(output_dir / "main_task_model_comparison_metrics.csv", index=False, encoding="utf-8-sig")

        print("\n" + "=" * 80)
        print("Main task model comparison")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print(f"\nSaved: {output_dir / 'main_task_model_comparison_metrics.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
