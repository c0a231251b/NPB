# -*- coding: utf-8 -*-
"""
LightGBM + class_weight と Factorization Machine + class_weight を同時に評価するスクリプト。

目的:
- train_all_added_11_features.csv を用いる
- 元の特徴量はそのまま使い、追加済みの左右別8特徴量も数値特徴量として利用する
- 同じ train / valid / test の時系列分割で以下を比較する
    1. LightGBM + class_weight
    2. Factorization Machine + class_weight

実行例:
    python evaluate_lightgbm_fm_11_features.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_11_features.csv"

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm torch tqdm

出力:
    output_lightgbm_fm_11_features/
    ├── model_comparison_metrics.csv
    ├── lightgbm_cw_predictions.csv
    ├── lightgbm_cw_classification_report.txt
    ├── lightgbm_cw_confusion_matrix.csv
    ├── lightgbm_cw_feature_importance.csv
    ├── fm_cw_predictions.csv
    ├── fm_cw_classification_report.txt
    ├── fm_cw_confusion_matrix.csv
    ├── fm_cw_history.csv
    ├── fm_cw_best_model.pt
    ├── target_distribution.csv
    ├── class_weights.csv
    └── feature_config.json

注意:
- FMは一般的なFactorization Machineです。
- previous_result1〜5 は通常のカテゴリ特徴量として扱い、順序構造は明示的には扱いません。
- LightGBMとFMは、同じCSV、同じラベル変換、同じ時系列分割、同じclass_weightで比較します。
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# 1. ラベル・特徴量定義
# ============================================================

LABELS = [
    "SINGLE",
    "DOUBLE",
    "TRIPLE",
    "HR",
    "BB",
    "SO",
    "HBP",
    "OUT",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

SINGLE_SET = {"一安", "二安", "三安", "中安", "右安", "左安", "投安", "捕安", "遊安"}
DOUBLE_SET = {"中２", "二２", "右２", "左２", "投２", "遊２", "三２", "捕２", "一２"}
TRIPLE_SET = {"中３", "右３", "左３", "二３", "三３", "一３", "遊３"}
HR_SET = {"中本", "右本", "左本", "満本", "本塁打"}
BB_SET = {"四球", "敬遠"}
SO_SET = {"三振", "振逃"}
HBP_SET = {"死球"}

# LightGBM / FM 用: すべてのカテゴリ特徴量を通常カテゴリとして扱う
CATEGORICAL_FEATURES = [
    "batter_id",
    "pitcher_id",
    "stadium_id",
    "Home_Away_id",
    "batter_hand",
    "pitcher_hand",
    "previous_result1",
    "previous_result2",
    "previous_result3",
    "previous_result4",
    "previous_result5",
]

# train_all_added_11_features.csv の数値特徴量
# 既存特徴量 + 追加した左右別8特徴量
NUMERIC_FEATURES = [
    "bat_order",
    "inning",
    "top_bottom",
    "out_count",
    "runner1",
    "runner2",
    "runner3",
    "score_diff",
    "batter_avg",
    "batter_ops",
    "batter_obp",
    "batter_slg",
    "batter_iso",
    "batter_vs_rhp_avg",
    "batter_vs_rhp_ops",
    "batter_vs_lhp_avg",
    "batter_vs_lhp_ops",
    "pitcher_era",
    "pitcher_k9",
    "pitcher_whip",
    "pitcher_ops",
    "vs_ops",
]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


# ============================================================
# 2. 共通処理
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_label(raw_label: str) -> str:
    x = str(raw_label).strip()

    if x in SINGLE_SET:
        return "SINGLE"
    if x in DOUBLE_SET:
        return "DOUBLE"
    if x in TRIPLE_SET:
        return "TRIPLE"
    if x in HR_SET:
        return "HR"
    if x in BB_SET:
        return "BB"
    if x in SO_SET:
        return "SO"
    if x in HBP_SET:
        return "HBP"

    return "OUT"


def normalize_top_bottom(value) -> float:
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    if s in ["表", "top", "Top", "TOP", "0"]:
        return 0.0
    if s in ["裏", "bottom", "Bottom", "BOTTOM", "1"]:
        return 1.0

    try:
        return float(s)
    except ValueError:
        return np.nan


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["game_date", "label"])
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"必要な列がCSVに存在しません: {missing}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date"]).copy()

    df["target_label"] = df["label"].apply(convert_label)
    df["target"] = df["target_label"].map(LABEL_TO_ID).astype(int)

    df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
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

    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        raise ValueError("train / valid / test のいずれかが0行です。分割比率を確認してください。")

    return SplitData(train_df=train_df, valid_df=valid_df, test_df=test_df)


def build_class_weights(y_train: np.ndarray, normalize_mean_one: bool = True) -> np.ndarray:
    classes = np.arange(len(LABELS))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    ).astype(np.float32)

    if normalize_mean_one:
        weights = weights / np.mean(weights)

    return weights


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }

    if y_prob is not None:
        for k in [2, 3]:
            topk = np.argsort(y_prob, axis=1)[:, -k:]
            result[f"top{k}_accuracy"] = float(
                np.mean([y_true[i] in topk[i] for i in range(len(y_true))])
            )

    return result


def save_report(
    output_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> Dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob)
    metrics["model"] = model_name

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        digits=6,
    )
    (output_dir / f"{model_name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        output_dir / f"{model_name}_confusion_matrix.csv",
        encoding="utf-8-sig",
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

    pred_df.to_csv(
        output_dir / f"{model_name}_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return metrics


# ============================================================
# 3. LightGBM + class_weight
# ============================================================

def train_evaluate_lightgbm_cw(
    split: SplitData,
    output_dir: Path,
    class_weights: np.ndarray,
    seed: int = 42,
    n_estimators: int = 4000,
    learning_rate: float = 0.03,
    early_stopping_rounds: int = 150,
) -> Optional[Dict[str, float]]:
    try:
        import lightgbm as lgb
    except ImportError:
        print("[WARN] lightgbm がインストールされていないため、LightGBM評価をスキップします。")
        print("       pip install lightgbm")
        return None

    print("\n" + "=" * 80)
    print("LightGBM + class_weight")
    print("=" * 80)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()

    for col in CATEGORICAL_FEATURES:
        categories = pd.Index(
            pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).astype(str).unique()
        )
        dtype = pd.CategoricalDtype(categories=categories)
        train_df[col] = train_df[col].astype(dtype)
        valid_df[col] = valid_df[col].astype(dtype)
        test_df[col] = test_df[col].astype(dtype)

    medians = train_df[NUMERIC_FEATURES].median(numeric_only=True)
    train_df[NUMERIC_FEATURES] = train_df[NUMERIC_FEATURES].fillna(medians)
    valid_df[NUMERIC_FEATURES] = valid_df[NUMERIC_FEATURES].fillna(medians)
    test_df[NUMERIC_FEATURES] = test_df[NUMERIC_FEATURES].fillna(medians)

    X_train = train_df[feature_cols]
    y_train = train_df["target"].values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["target"].values
    X_test = test_df[feature_cols]
    y_test = test_df["target"].values

    class_weight_dict = {i: float(class_weights[i]) for i in range(len(LABELS))}
    print("LightGBM class_weight:")
    print(class_weight_dict)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(LABELS),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
        class_weight=class_weight_dict,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=callbacks,
    )

    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = save_report(output_dir, "lightgbm_cw", y_test, y_pred, y_prob)

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(output_dir / "lightgbm_cw_feature_importance.csv", index=False, encoding="utf-8-sig")

    print("\n[LightGBM + class_weight Test Metrics]")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")

    return metrics


# ============================================================
# 4. Dataset / Encoder
# ============================================================

class CommonEncoder:
    """FM用のカテゴリ/数値エンコーダ。"""

    def __init__(self, categorical_features: List[str], numeric_features: List[str]):
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features

        self.category_maps: Dict[str, Dict[str, int]] = {}
        self.numeric_median: Dict[str, float] = {}
        self.numeric_mean: Dict[str, float] = {}
        self.numeric_std: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col in self.categorical_features:
            values = sorted(df[col].fillna("<NA>").astype(str).unique().tolist())
            self.category_maps[col] = {v: i + 1 for i, v in enumerate(values)}

        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce")
            median_value = s.median()
            med = float(median_value) if not np.isnan(median_value) else 0.0
            filled = s.fillna(med)
            mean = float(filled.mean())
            std = float(filled.std())
            if std == 0 or np.isnan(std):
                std = 1.0

            self.numeric_median[col] = med
            self.numeric_mean[col] = mean
            self.numeric_std[col] = std

    def transform_cat(self, df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        arrays = []
        for col in cols:
            mp = self.category_maps[col]
            arr = (
                df[col]
                .fillna("<NA>")
                .astype(str)
                .map(lambda x: mp.get(x, 0))
                .astype(np.int64)
                .values
            )
            arrays.append(arr)
        return np.stack(arrays, axis=1)

    def transform_num(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_median[col])
            arr = ((s - self.numeric_mean[col]) / self.numeric_std[col]).astype(np.float32).values
            arrays.append(arr)
        return np.stack(arrays, axis=1)

    def cardinality(self, col: str) -> int:
        return len(self.category_maps[col]) + 1

    def cardinalities(self, cols: List[str]) -> List[int]:
        return [self.cardinality(col) for col in cols]


class FMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: CommonEncoder):
        self.x_cat = encoder.transform_cat(df, CATEGORICAL_FEATURES)
        self.x_num = encoder.transform_num(df)
        self.y = df["target"].astype(np.int64).values

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
        }


# ============================================================
# 5. FM + class_weight
# ============================================================

class FactorizationMachineClassifier(nn.Module):
    """
    一般的なFactorization Machineの多クラス分類版。

    実装方針:
    - カテゴリ特徴量と数値特徴量をfieldとして扱う
    - 1次項 + 2次相互作用項
    - 多クラス分類のため、各クラスごとにFMの相互作用重みを持つ
    """

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
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, num_classes * embed_dim)
            for cardinality in cat_cardinalities
        ])

        self.num_embeddings = nn.Parameter(
            torch.randn(num_numeric, num_classes, embed_dim) * 0.01
        )

        self.cat_linear = nn.ModuleList([
            nn.Embedding(cardinality, num_classes)
            for cardinality in cat_cardinalities
        ])

        self.num_linear = nn.Parameter(
            torch.randn(num_numeric, num_classes) * 0.01
        )

        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.cat_linear:
            nn.init.zeros_(emb.weight)
        nn.init.xavier_uniform_(self.num_embeddings)
        nn.init.xavier_uniform_(self.num_linear)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size = x_cat.size(0)

        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            e = emb(x_cat[:, i]).view(batch_size, self.num_classes, self.embed_dim)
            cat_embs.append(e)

        cat_embs = torch.stack(cat_embs, dim=1)  # (B, C, K, D)
        num_embs = x_num.unsqueeze(-1).unsqueeze(-1) * self.num_embeddings.unsqueeze(0)  # (B, N, K, D)

        all_embs = torch.cat([cat_embs, num_embs], dim=1)  # (B, F, K, D)
        all_embs = self.dropout(all_embs)

        summed = torch.sum(all_embs, dim=1)                  # (B, K, D)
        summed_square = summed * summed
        square_summed = torch.sum(all_embs * all_embs, dim=1)
        interaction = 0.5 * (summed_square - square_summed)  # (B, K, D)
        interaction_logits = torch.sum(interaction, dim=2)   # (B, K)

        linear_logits = self.bias.unsqueeze(0).expand(batch_size, -1)

        for i, lin in enumerate(self.cat_linear):
            linear_logits = linear_logits + lin(x_cat[:, i])

        numeric_linear_logits = torch.einsum("bn,nk->bk", x_num, self.num_linear)
        linear_logits = linear_logits + numeric_linear_logits

        return linear_logits + interaction_logits


# ============================================================
# 6. NN学習共通
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 5.0,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc="fm train", leave=False):
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["x_cat"].to(device),
            batch["x_num"].to(device),
        )

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
def evaluate_fm(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    for batch in tqdm(loader, desc="fm eval", leave=False):
        y = batch["y"].to(device)
        logits = model(
            batch["x_cat"].to(device),
            batch["x_num"].to(device),
        )

        loss = criterion(logits, y)
        prob = torch.softmax(logits, dim=1)

        total_loss += loss.item() * y.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(torch.argmax(prob, dim=1).cpu().numpy())
        y_prob.append(prob.cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
    )


def fit_fm_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    monitor: str = "macro_f1",
) -> Dict[str, float]:
    best_score = -float("inf") if monitor == "macro_f1" else float("inf")
    best_state = None
    best_epoch = 0
    best_valid_loss = float("inf")
    bad_epochs = 0

    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        valid_loss, y_valid, pred_valid, prob_valid = evaluate_fm(
            model,
            valid_loader,
            criterion,
            device,
        )
        valid_metrics = compute_metrics(y_valid, pred_valid, prob_valid)

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
            f"valid_macro_f1={valid_metrics['macro_f1']:.5f}"
        )

        if monitor == "macro_f1":
            current_score = valid_metrics["macro_f1"]
            improved = current_score > best_score + 1e-5
        else:
            current_score = valid_loss
            improved = current_score < best_score - 1e-5

        if improved:
            best_score = current_score
            best_epoch = epoch
            best_valid_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(
                f"Early stopping: best_epoch={best_epoch}, "
                f"best_valid_loss={best_valid_loss:.6f}, "
                f"best_{monitor}={best_score:.6f}"
            )
            break

    pd.DataFrame(history).to_csv(
        output_dir / "fm_cw_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            f"best_{monitor}": best_score,
            "labels": LABELS,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
        },
        output_dir / "fm_cw_best_model.pt",
    )

    test_loss, y_test, y_pred, y_prob = evaluate_fm(
        model,
        test_loader,
        criterion,
        device,
    )

    metrics = save_report(output_dir, "fm_cw", y_test, y_pred, y_prob)
    metrics["test_loss"] = float(test_loss)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_valid_loss"] = float(best_valid_loss)
    metrics[f"best_valid_{monitor}"] = float(best_score)

    print("\n[FM + class_weight Test Metrics]")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")

    return metrics


def train_evaluate_fm_cw(
    split: SplitData,
    output_dir: Path,
    class_weights: np.ndarray,
    seed: int = 42,
    batch_size: int = 512,
    epochs: int = 50,
    patience: int = 8,
    embed_dim: int = 32,
    lr: float = 1e-3,
    monitor: str = "macro_f1",
) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print("FM + class_weight")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    encoder = CommonEncoder(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    encoder.fit(split.train_df)

    train_ds = FMDataset(split.train_df, encoder)
    valid_ds = FMDataset(split.valid_df, encoder)
    test_ds = FMDataset(split.test_df, encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FactorizationMachineClassifier(
        cat_cardinalities=encoder.cardinalities(CATEGORICAL_FEATURES),
        num_numeric=len(NUMERIC_FEATURES),
        num_classes=len(LABELS),
        embed_dim=embed_dim,
        dropout=0.1,
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    return fit_fm_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        output_dir=output_dir,
        device=device,
        epochs=epochs,
        patience=patience,
        monitor=monitor,
    )


# ============================================================
# 7. main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/train_all_added_11_features.csv",
        help="11特徴量追加済みCSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_lightgbm_fm_11_features",
        help="出力先ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--fm_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--fm_embed_dim", type=int, default=32)
    parser.add_argument("--fm_lr", type=float, default=1e-3)

    parser.add_argument("--lgb_n_estimators", type=int, default=4000)
    parser.add_argument("--lgb_learning_rate", type=float, default=0.03)
    parser.add_argument("--lgb_early_stopping_rounds", type=int, default=150)

    parser.add_argument(
        "--monitor",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "valid_loss"],
        help="FMモデル保存基準。クラス不均衡対策では macro_f1 推奨。",
    )

    parser.add_argument("--skip_lightgbm", action="store_true")
    parser.add_argument("--skip_fm", action="store_true")

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Class-weight comparison: LightGBM / FM with 11 features")
    print("=" * 80)
    print(f"csv_path: {csv_path}")
    print(f"output_dir: {output_dir}")
    print(f"monitor: {args.monitor}")

    df = prepare_dataframe(csv_path)
    split = split_by_game_date(df, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    print("\n[Data split]")
    print(f"all   : {len(df):>6} rows, dates={df['game_date'].nunique()}")
    print(
        f"train : {len(split.train_df):>6} rows, "
        f"{split.train_df['game_date'].min().date()} ~ {split.train_df['game_date'].max().date()}"
    )
    print(
        f"valid : {len(split.valid_df):>6} rows, "
        f"{split.valid_df['game_date'].min().date()} ~ {split.valid_df['game_date'].max().date()}"
    )
    print(
        f"test  : {len(split.test_df):>6} rows, "
        f"{split.test_df['game_date'].min().date()} ~ {split.test_df['game_date'].max().date()}"
    )

    print("\n[Target distribution]")
    dist = df["target_label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print(dist)
    dist.to_csv(output_dir / "target_distribution.csv", encoding="utf-8-sig")

    class_weights = build_class_weights(split.train_df["target"].values, normalize_mean_one=True)

    class_weight_df = pd.DataFrame({
        "label": LABELS,
        "class_id": list(range(len(LABELS))),
        "class_weight": class_weights,
        "train_count": split.train_df["target_label"].value_counts().reindex(LABELS).fillna(0).astype(int).values,
    })
    class_weight_df.to_csv(output_dir / "class_weights.csv", index=False, encoding="utf-8-sig")

    print("\n[Class weights]")
    print(class_weight_df.to_string(index=False))

    with open(output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "labels": LABELS,
                "monitor": args.monitor,
                "train_ratio": args.train_ratio,
                "valid_ratio": args.valid_ratio,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    all_metrics = []

    if not args.skip_lightgbm:
        metrics = train_evaluate_lightgbm_cw(
            split=split,
            output_dir=output_dir,
            class_weights=class_weights,
            seed=args.seed,
            n_estimators=args.lgb_n_estimators,
            learning_rate=args.lgb_learning_rate,
            early_stopping_rounds=args.lgb_early_stopping_rounds,
        )
        if metrics is not None:
            all_metrics.append(metrics)

    if not args.skip_fm:
        metrics = train_evaluate_fm_cw(
            split=split,
            output_dir=output_dir,
            class_weights=class_weights,
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.fm_epochs,
            patience=args.patience,
            embed_dim=args.fm_embed_dim,
            lr=args.fm_lr,
            monitor=args.monitor,
        )
        all_metrics.append(metrics)

    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        cols = ["model"] + [c for c in comparison_df.columns if c != "model"]
        comparison_df = comparison_df[cols]
        comparison_df.to_csv(
            output_dir / "model_comparison_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print("\n" + "=" * 80)
        print("Model comparison")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print(f"\nSaved: {output_dir / 'model_comparison_metrics.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
