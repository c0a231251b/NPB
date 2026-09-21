# -*- coding: utf-8 -*-
"""
LightGBM + class_weight / FM + class_weight / SeqFM + class_weight・Focal Loss を比較するスクリプト。

目的:
- 7特徴量を追加した train_all_added_7_features.csv を用いる
- 同じ train / valid / test 分割で以下を比較する
    1. LightGBM + class_weight
    2. Factorization Machine + class_weight
    3. SeqFM + CrossEntropyLoss + class_weight
    4. SeqFM + FocalLoss + class_weight

実行例:
    python evaluate_class_weight_models.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_7_features.csv"

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm torch tqdm

出力:
    output_class_weight_models/
    ├── model_comparison_metrics.csv
    ├── lightgbm_cw_predictions.csv
    ├── lightgbm_cw_classification_report.txt
    ├── lightgbm_cw_confusion_matrix.csv
    ├── lightgbm_cw_feature_importance.csv
    ├── fm_cw_predictions.csv
    ├── fm_cw_classification_report.txt
    ├── fm_cw_confusion_matrix.csv
    ├── fm_cw_history.csv
    ├── seqfm_cw_ce_predictions.csv
    ├── seqfm_cw_ce_classification_report.txt
    ├── seqfm_cw_ce_confusion_matrix.csv
    ├── seqfm_cw_ce_history.csv
    ├── seqfm_cw_focal_predictions.csv
    ├── seqfm_cw_focal_classification_report.txt
    ├── seqfm_cw_focal_confusion_matrix.csv
    ├── seqfm_cw_focal_history.csv
    ├── target_distribution.csv
    ├── class_weights.csv
    └── feature_config.json

注意:
- 本スクリプト内のFMは一般的なFactorization Machineです。
  previous_result1〜5 は通常のカテゴリ特徴量として扱い、順序構造は明示的には扱いません。
- SeqFMでは previous_result1〜5 を系列として扱い、TransformerEncoderで時系列情報を処理します。
- SeqFMは研究用の比較実験として、前回作成したSeqFM構造に近い簡潔な実装にしています。
"""

from __future__ import annotations

import argparse
import json
import random
import re
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
DOUBLE_SET = {"中２", "二２", "右２", "左２", "投２", "遊２"}
TRIPLE_SET = {"中３", "右３", "左３"}
HR_SET = {"中本", "右本", "左本"}
BB_SET = {"四球", "敬遠"}
SO_SET = {"三振", "振逃"}
HBP_SET = {"死球"}

# FM / LightGBM用: すべてのカテゴリ特徴量を通常カテゴリとして扱う
FM_CATEGORICAL_FEATURES = [
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

# SeqFM用: static/context/dynamicを分ける
SEQFM_STATIC_CATEGORICAL_FEATURES = [
    "batter_id",
    "pitcher_id",
    "batter_hand",
    "pitcher_hand",
]

SEQFM_CONTEXT_CATEGORICAL_FEATURES = [
    "stadium_id",
    "Home_Away_id",
]

SEQFM_DYNAMIC_FEATURES = [
    "previous_result1",
    "previous_result2",
    "previous_result3",
    "previous_result4",
    "previous_result5",
]

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
    if s in ["表", "top", "Top"]:
        return 0.0
    if s in ["裏", "bottom", "Bottom"]:
        return 1.0

    try:
        return float(s)
    except ValueError:
        return np.nan


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_columns = set(
        FM_CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
        + ["game_date", "label"]
    )
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

    for col in FM_CATEGORICAL_FEATURES:
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


def build_class_weights(y_train: np.ndarray, normalize_mean_one: bool = True) -> np.ndarray:
    """sklearnのbalanced class weightを計算する。"""
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

    feature_cols = FM_CATEGORICAL_FEATURES + NUMERIC_FEATURES

    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()

    for col in FM_CATEGORICAL_FEATURES:
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
        n_estimators=4000,
        learning_rate=0.03,
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
        lgb.early_stopping(stopping_rounds=150, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss",
        categorical_feature=FM_CATEGORICAL_FEATURES,
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
    """FM・SeqFM共通のカテゴリ/数値エンコーダ。"""

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
            med = float(s.median()) if not np.isnan(s.median()) else 0.0
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
        self.x_cat = encoder.transform_cat(df, FM_CATEGORICAL_FEATURES)
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


class SeqFMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: CommonEncoder):
        self.x_static = encoder.transform_cat(df, SEQFM_STATIC_CATEGORICAL_FEATURES)
        self.x_context_cat = encoder.transform_cat(df, SEQFM_CONTEXT_CATEGORICAL_FEATURES)
        self.x_dynamic = encoder.transform_cat(df, SEQFM_DYNAMIC_FEATURES)
        self.x_num = encoder.transform_num(df)
        self.y = df["target"].astype(np.int64).values

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "x_static": torch.tensor(self.x_static[idx], dtype=torch.long),
            "x_context_cat": torch.tensor(self.x_context_cat[idx], dtype=torch.long),
            "x_dynamic": torch.tensor(self.x_dynamic[idx], dtype=torch.long),
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
        }


# ============================================================
# 5. Loss
# ============================================================

class FocalLoss(nn.Module):
    """
    多クラスFocal Loss。

    class_weight:
        少数クラスを重視するためのクラス重み。
    gamma:
        難しいサンプルをどの程度重視するか。
        gamma=0なら重み付きCrossEntropyとほぼ同じ。
    """

    def __init__(
        self,
        class_weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.class_weight = class_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        target_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, target.unsqueeze(1)).squeeze(1)

        focal_factor = (1.0 - target_probs).pow(self.gamma)
        loss = -focal_factor * target_log_probs

        if self.class_weight is not None:
            weights = self.class_weight[target]
            loss = loss * weights

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ============================================================
# 6. FM + class_weight
# ============================================================

class FactorizationMachineClassifier(nn.Module):
    """
    一般的なFactorization Machineの多クラス分類版。

    実装方針:
    - カテゴリ特徴量と数値特徴量をfieldとして扱う
    - 1次項 + 2次相互作用項 + MLPではなく、純粋なFMに近い形
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
            # (B, K*D) -> (B, K, D)
            e = emb(x_cat[:, i]).view(batch_size, self.num_classes, self.embed_dim)
            cat_embs.append(e)

        cat_embs = torch.stack(cat_embs, dim=1)  # (B, C, K, D)

        # (B, N, K, D)
        num_embs = x_num.unsqueeze(-1).unsqueeze(-1) * self.num_embeddings.unsqueeze(0)

        all_embs = torch.cat([cat_embs, num_embs], dim=1)  # (B, F, K, D)
        all_embs = self.dropout(all_embs)

        summed = torch.sum(all_embs, dim=1)               # (B, K, D)
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
# 7. SeqFM
# ============================================================

class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, D)
        scores = self.score(x).squeeze(-1)  # (B, L)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights


class SeqFMClassifier(nn.Module):
    """
    SeqFM風の打席結果予測モデル。

    Static:
        batter_id, pitcher_id, batter_hand, pitcher_hand

    Dynamic:
        previous_result1〜5を系列としてTransformerEncoderへ入力

    Context:
        stadium_id, Home_Away_id と数値特徴量

    Fusion:
        static / dynamic / context の2次相互作用をFM風に結合
    """

    def __init__(
        self,
        static_cardinalities: List[int],
        context_cat_cardinalities: List[int],
        previous_result_cardinality: int,
        num_numeric: int,
        num_classes: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        seq_len: int = 5,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.static_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality in static_cardinalities
        ])
        self.static_proj = nn.Sequential(
            nn.Linear(len(static_cardinalities) * embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.context_cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality in context_cat_cardinalities
        ])

        self.numeric_encoder = nn.Sequential(
            nn.Linear(num_numeric, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        context_input_dim = (len(context_cat_cardinalities) + 1) * embed_dim
        self.context_proj = nn.Sequential(
            nn.Linear(context_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.prev_embedding = nn.Embedding(previous_result_cardinality, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.dynamic_pool = AttentionPooling(embed_dim)

        # static / dynamic / context の3field相互作用
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 6, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(
        self,
        x_static: torch.Tensor,
        x_context_cat: torch.Tensor,
        x_dynamic: torch.Tensor,
        x_num: torch.Tensor,
    ) -> torch.Tensor:
        static_embs = []
        for i, emb in enumerate(self.static_embeddings):
            static_embs.append(emb(x_static[:, i]))
        static_cat = torch.cat(static_embs, dim=1)
        static_vec = self.static_proj(static_cat)

        context_cat_embs = []
        for i, emb in enumerate(self.context_cat_embeddings):
            context_cat_embs.append(emb(x_context_cat[:, i]))
        numeric_vec = self.numeric_encoder(x_num)
        context_cat = torch.cat(context_cat_embs + [numeric_vec], dim=1)
        context_vec = self.context_proj(context_cat)

        dyn = self.prev_embedding(x_dynamic)
        dyn = dyn + self.pos_embedding[:, : dyn.size(1), :]
        dyn = self.transformer(dyn)
        dynamic_vec, _ = self.dynamic_pool(dyn)

        # FM風相互作用
        sd = static_vec * dynamic_vec
        sc = static_vec * context_vec
        dc = dynamic_vec * context_vec

        fused = torch.cat(
            [static_vec, dynamic_vec, context_vec, sd, sc, dc],
            dim=1,
        )
        z = self.fusion(fused)
        logits = self.head(z)

        return logits


# ============================================================
# 8. NN学習共通
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
    grad_clip: float = 5.0,
) -> Tuple[float, float]:
    model.train()

    total_loss = 0.0
    y_true = []
    y_pred = []

    for batch in tqdm(loader, desc=f"{model_type} train", leave=False):
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)

        if model_type == "fm":
            logits = model(
                batch["x_cat"].to(device),
                batch["x_num"].to(device),
            )
        else:
            logits = model(
                batch["x_static"].to(device),
                batch["x_context_cat"].to(device),
                batch["x_dynamic"].to(device),
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
def evaluate_nn(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()

    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    for batch in tqdm(loader, desc=f"{model_type} eval", leave=False):
        y = batch["y"].to(device)

        if model_type == "fm":
            logits = model(
                batch["x_cat"].to(device),
                batch["x_num"].to(device),
            )
        else:
            logits = model(
                batch["x_static"].to(device),
                batch["x_context_cat"].to(device),
                batch["x_dynamic"].to(device),
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


def fit_nn_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    model_name: str,
    model_type: str,
    device: torch.device,
    epochs: int,
    patience: int,
    monitor: str = "macro_f1",
) -> Dict[str, float]:
    """
    NNモデルの共通学習処理。

    monitor:
        "valid_loss" または "macro_f1"
        クラス不均衡対策では macro_f1 保存がおすすめ。
    """
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
            model_type=model_type,
        )

        valid_loss, y_valid, pred_valid, prob_valid = evaluate_nn(
            model,
            valid_loader,
            criterion,
            device,
            model_type=model_type,
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
        output_dir / f"{model_name}_history.csv",
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
        },
        output_dir / f"{model_name}_best_model.pt",
    )

    test_loss, y_test, y_pred, y_prob = evaluate_nn(
        model,
        test_loader,
        criterion,
        device,
        model_type=model_type,
    )

    metrics = save_report(output_dir, model_name, y_test, y_pred, y_prob)
    metrics["test_loss"] = float(test_loss)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_valid_loss"] = float(best_valid_loss)
    metrics[f"best_valid_{monitor}"] = float(best_score)

    print(f"\n[{model_name} Test Metrics]")
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

    encoder = CommonEncoder(FM_CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    encoder.fit(split.train_df)

    train_ds = FMDataset(split.train_df, encoder)
    valid_ds = FMDataset(split.valid_df, encoder)
    test_ds = FMDataset(split.test_df, encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FactorizationMachineClassifier(
        cat_cardinalities=encoder.cardinalities(FM_CATEGORICAL_FEATURES),
        num_numeric=len(NUMERIC_FEATURES),
        num_classes=len(LABELS),
        embed_dim=embed_dim,
        dropout=0.1,
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    return fit_nn_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        output_dir=output_dir,
        model_name="fm_cw",
        model_type="fm",
        device=device,
        epochs=epochs,
        patience=patience,
        monitor=monitor,
    )


def train_evaluate_seqfm(
    split: SplitData,
    output_dir: Path,
    class_weights: np.ndarray,
    model_name: str,
    loss_name: str,
    seed: int = 42,
    batch_size: int = 512,
    epochs: int = 50,
    patience: int = 8,
    embed_dim: int = 64,
    lr: float = 3e-4,
    focal_gamma: float = 2.0,
    monitor: str = "macro_f1",
) -> Dict[str, float]:
    print("\n" + "=" * 80)
    print(f"SeqFM + {loss_name} + class_weight")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    all_seqfm_cat_features = (
        SEQFM_STATIC_CATEGORICAL_FEATURES
        + SEQFM_CONTEXT_CATEGORICAL_FEATURES
        + SEQFM_DYNAMIC_FEATURES
    )

    encoder = CommonEncoder(all_seqfm_cat_features, NUMERIC_FEATURES)
    encoder.fit(split.train_df)

    train_ds = SeqFMDataset(split.train_df, encoder)
    valid_ds = SeqFMDataset(split.valid_df, encoder)
    test_ds = SeqFMDataset(split.test_df, encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # previous_result1〜5 は同じ語彙を想定するが、列ごとにVocabularyを作っているため、
    # 最大cardinalityを使うことで全列を安全に扱う。
    previous_cardinality = max(encoder.cardinalities(SEQFM_DYNAMIC_FEATURES))

    model = SeqFMClassifier(
        static_cardinalities=encoder.cardinalities(SEQFM_STATIC_CATEGORICAL_FEATURES),
        context_cat_cardinalities=encoder.cardinalities(SEQFM_CONTEXT_CATEGORICAL_FEATURES),
        previous_result_cardinality=previous_cardinality,
        num_numeric=len(NUMERIC_FEATURES),
        num_classes=len(LABELS),
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
        seq_len=len(SEQFM_DYNAMIC_FEATURES),
    ).to(device)

    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    elif loss_name == "focal":
        criterion = FocalLoss(class_weight=weight_tensor, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    metrics = fit_nn_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        output_dir=output_dir,
        model_name=model_name,
        model_type="seqfm",
        device=device,
        epochs=epochs,
        patience=patience,
        monitor=monitor,
    )
    metrics["loss_name"] = loss_name
    if loss_name == "focal":
        metrics["focal_gamma"] = float(focal_gamma)

    return metrics


# ============================================================
# 9. main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/train_all_added_7_features.csv",
        help="7特徴量追加済みCSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_class_weight_models",
        help="出力先ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--fm_epochs", type=int, default=50)
    parser.add_argument("--seqfm_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--fm_embed_dim", type=int, default=32)
    parser.add_argument("--seqfm_embed_dim", type=int, default=64)

    parser.add_argument("--fm_lr", type=float, default=1e-3)
    parser.add_argument("--seqfm_lr", type=float, default=3e-4)
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument(
        "--monitor",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "valid_loss"],
        help="NNモデル保存基準。クラス不均衡対策では macro_f1 推奨。",
    )

    parser.add_argument("--skip_lightgbm", action="store_true")
    parser.add_argument("--skip_fm", action="store_true")
    parser.add_argument("--skip_seqfm_ce", action="store_true")
    parser.add_argument("--skip_seqfm_focal", action="store_true")

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Class-weight comparison: LightGBM / FM / SeqFM")
    print("=" * 80)
    print(f"csv_path: {csv_path}")
    print(f"output_dir: {output_dir}")
    print(f"monitor: {args.monitor}")

    df = prepare_dataframe(csv_path)
    split = split_by_game_date(df)

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
                "fm_categorical_features": FM_CATEGORICAL_FEATURES,
                "seqfm_static_categorical_features": SEQFM_STATIC_CATEGORICAL_FEATURES,
                "seqfm_context_categorical_features": SEQFM_CONTEXT_CATEGORICAL_FEATURES,
                "seqfm_dynamic_features": SEQFM_DYNAMIC_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "labels": LABELS,
                "monitor": args.monitor,
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

    if not args.skip_seqfm_ce:
        metrics = train_evaluate_seqfm(
            split=split,
            output_dir=output_dir,
            class_weights=class_weights,
            model_name="seqfm_cw_ce",
            loss_name="cross_entropy",
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.seqfm_epochs,
            patience=args.patience,
            embed_dim=args.seqfm_embed_dim,
            lr=args.seqfm_lr,
            focal_gamma=args.focal_gamma,
            monitor=args.monitor,
        )
        all_metrics.append(metrics)

    if not args.skip_seqfm_focal:
        metrics = train_evaluate_seqfm(
            split=split,
            output_dir=output_dir,
            class_weights=class_weights,
            model_name="seqfm_cw_focal",
            loss_name="focal",
            seed=args.seed,
            batch_size=args.batch_size,
            epochs=args.seqfm_epochs,
            patience=args.patience,
            embed_dim=args.seqfm_embed_dim,
            lr=args.seqfm_lr,
            focal_gamma=args.focal_gamma,
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
