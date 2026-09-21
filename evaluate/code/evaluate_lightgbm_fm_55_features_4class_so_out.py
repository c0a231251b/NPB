# -*- coding: utf-8 -*-
"""
train_all_added_55_features.csv を用いて、提案3の4分類（SINGLE / XBH / BB_HBP / SO_OUT）で LightGBM と Factorization Machine(FM) を評価するスクリプト。

目的:
- 同一データ分割・同一特徴量で LightGBM と一般的なFMを比較する
- 評価指標として Accuracy / Macro F1 / Weighted F1 / Top-k Accuracy を出力する
- 比較結果を output_lgbm_fm_55_features/model_comparison_metrics.csv に保存する

実行例:
python evaluate_lightgbm_fm_55_features_4class_so_out.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_55_features.csv"

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm torch tqdm

注意:
- LightGBM が未インストールの場合は LightGBM 部分をスキップし、FMのみ評価します。
- FMは「一般的なFactorization Machine」に近い構成で、時系列処理は行いません。
- previous_result1〜5 はカテゴリ特徴量として扱います。
- 既存特徴量 + 左右別成績8特徴量 + 直近成績9特徴量 + 試合状況3特徴量 + 累積率10特徴量 + イベント/場面6特徴量 + 出塁/直接対戦/状況12特徴量を扱います。43特徴量版に batter_onbase_event_rate / pitcher_onbase_event_rate_allowed / vs_pa_count / vs_hit_rate / vs_k_rate / vs_bb_rate / vs_xbh_rate / inning_out_state / bat_order_runner_state / is_no_out_runner_on / is_two_out / month を追加した55特徴量版です.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# 1. 設定
# ============================================================

LABELS = [
    "SINGLE",
    "XBH",
    "BB_HBP",
    "SO_OUT",
]

LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# 元の細かい打席結果 → 4クラス（SINGLE / XBH / BB_HBP / SO_OUT）へ変換
SINGLE_SET = {
    "一安", "二安", "三安", "中安", "右安", "左安", "投安", "捕安", "遊安"
}
DOUBLE_SET = {
    "中２", "二２", "右２", "左２", "投２", "遊２"
}
TRIPLE_SET = {
    "中３", "右３", "左３"
}
HR_SET = {
    "中本", "右本", "左本"
}
BB_SET = {
    "四球", "敬遠"
}
SO_SET = {
    "三振", "振逃"
}
HBP_SET = {
    "死球"
}


# 55特徴量版の特徴量定義
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
    "base_out_state",
    "inning_out_state",
    "bat_order_runner_state",
]

BASE_NUMERIC_FEATURES = [
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

HAND_SPLIT_NUMERIC_FEATURES = [
    "batter_vs_rhp_avg",
    "batter_vs_rhp_ops",
    "batter_vs_lhp_avg",
    "batter_vs_lhp_ops",
    "pitcher_vs_rhb_avg",
    "pitcher_vs_rhb_ops",
    "pitcher_vs_lhb_avg",
    "pitcher_vs_lhb_ops",
]

RECENT_NUMERIC_FEATURES = [
    "batter_recent_5pa_ops",
    "batter_recent_10pa_avg",
    "batter_recent_10pa_ops",
    "batter_recent_10pa_k_rate",
    "batter_recent_10pa_bb_rate",
    "pitcher_recent_bf_avg",
    "pitcher_recent_bf_ops",
    "pitcher_recent_bf_k_rate",
    "pitcher_recent_bf_bb_rate",
]

CONTEXT_NUMERIC_FEATURES = [
    "runner_count",
    "is_leadoff_batter_in_inning",
    "batting_turn_in_game",
]

CUMULATIVE_RATE_NUMERIC_FEATURES = [
    "batter_k_rate",
    "batter_bb_rate",
    "batter_hr_rate",
    "batter_xbh_rate",
    "batter_pa_count",
    "pitcher_k_rate",
    "pitcher_bb_rate",
    "pitcher_hr_rate",
    "pitcher_xbh_rate_allowed",
    "pitcher_bf_count",
]

EVENT_CONTEXT_NUMERIC_FEATURES = [
    "batter_hit_rate",
    "batter_out_rate",
    "pitcher_hit_rate_allowed",
    "pitcher_out_rate",
    "pitcher_bf_in_game",
]

ONBASE_VS_CONTEXT_NUMERIC_FEATURES = [
    "batter_onbase_event_rate",
    "pitcher_onbase_event_rate_allowed",
    "vs_pa_count",
    "vs_hit_rate",
    "vs_k_rate",
    "vs_bb_rate",
    "vs_xbh_rate",
    "is_no_out_runner_on",
    "is_two_out",
    "month",
]

ONBASE_VS_CONTEXT_CATEGORICAL_FEATURES = [
    "inning_out_state",
    "bat_order_runner_state",
]

NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + HAND_SPLIT_NUMERIC_FEATURES
    + RECENT_NUMERIC_FEATURES
    + CONTEXT_NUMERIC_FEATURES
    + CUMULATIVE_RATE_NUMERIC_FEATURES
    + EVENT_CONTEXT_NUMERIC_FEATURES
    + ONBASE_VS_CONTEXT_NUMERIC_FEATURES
)

IGNORE_COLUMNS = [
    "game_date",
    "batter_name",
    "pitcher_name",
    "label",
]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


# ============================================================
# 2. 共通ユーティリティ
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_label(raw_label: str) -> str:
    """細かい打席結果を SINGLE / XBH / BB_HBP / SO_OUT の4クラスへ変換する。"""
    x = str(raw_label).strip()

    if x in SINGLE_SET:
        return "SINGLE"
    if x in DOUBLE_SET or x in TRIPLE_SET or x in HR_SET:
        return "XBH"
    if x in BB_SET or x in HBP_SET:
        return "BB_HBP"

    # 三振とその他アウトを統合する
    return "SO_OUT"


def normalize_top_bottom(value) -> float:
    """表/裏 または 0/1 を数値化する。"""
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
    """CSVを読み込み、型変換とラベル変換を行う。"""
    df = pd.read_csv(csv_path)

    required_columns = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["game_date", "label"])
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"必要な列がCSVに存在しません: {missing}")

    df = df.copy()

    # 日付をdatetimeへ
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # ラベル変換
    df["target_label"] = df["label"].apply(convert_label)
    df["target"] = df["target_label"].map(LABEL_TO_ID).astype(int)

    # top_bottomを数値化
    df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)

    # numericをfloatへ
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # categoricalを文字列へ。欠損は <NA> として扱う。
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("<NA>").astype(str)

    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def split_by_game_date(df: pd.DataFrame, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> SplitData:
    """試合日単位で train / valid / test に時系列分割する。"""
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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """共通評価指標を計算する。"""
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
            result[f"top{k}_accuracy"] = float(np.mean([y_true[i] in topk[i] for i in range(len(y_true))]))

    return result


def save_report(
    output_dir: Path,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
) -> Dict[str, float]:
    """評価結果・classification report・confusion matrix・predictionを保存する。"""
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
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.to_csv(output_dir / f"{model_name}_confusion_matrix.csv", encoding="utf-8-sig")

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
# 3. LightGBM
# ============================================================

def train_evaluate_lightgbm(split: SplitData, output_dir: Path, seed: int = 42) -> Optional[Dict[str, float]]:
    """LightGBMを学習・評価する。"""
    try:
        import lightgbm as lgb
    except ImportError:
        print("[WARN] lightgbm がインストールされていないため、LightGBM評価をスキップします。")
        print("       インストールする場合: pip install lightgbm")
        return None

    print("\n" + "=" * 80)
    print("LightGBM evaluation")
    print("=" * 80)

    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()

    # LightGBM用カテゴリ型。train/valid/test全体でカテゴリ集合を合わせる。
    for col in CATEGORICAL_FEATURES:
        categories = pd.Index(
            pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).astype(str).unique()
        )
        dtype = pd.CategoricalDtype(categories=categories)
        train_df[col] = train_df[col].astype(dtype)
        valid_df[col] = valid_df[col].astype(dtype)
        test_df[col] = test_df[col].astype(dtype)

    # 数値欠損はtrain中央値で補完
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

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(LABELS),
        n_estimators=3000,
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
        class_weight=None,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
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

    metrics = save_report(output_dir, "lightgbm", y_test, y_pred, y_prob)

    print("\n[LightGBM Test Metrics]")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")

    # feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(output_dir / "lightgbm_feature_importance.csv", index=False, encoding="utf-8-sig")

    return metrics


# ============================================================
# 4. Factorization Machine
# ============================================================

class FeatureEncoder:
    """FM用にカテゴリ特徴量と数値特徴量をエンコードする。"""

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
            # 0: unknown
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

    def transform_categorical(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col in self.categorical_features:
            mp = self.category_maps[col]
            arr = df[col].fillna("<NA>").astype(str).map(lambda x: mp.get(x, 0)).astype(np.int64).values
            arrays.append(arr)
        return np.stack(arrays, axis=1)

    def transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        arrays = []
        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_median[col])
            arr = ((s - self.numeric_mean[col]) / self.numeric_std[col]).astype(np.float32).values
            arrays.append(arr)
        return np.stack(arrays, axis=1)

    def categorical_cardinalities(self) -> List[int]:
        """各カテゴリ特徴量の語彙数。unknown用に+1する。"""
        return [len(self.category_maps[col]) + 1 for col in self.categorical_features]


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
    """
    一般的なFactorization Machineによる多クラス分類。

    カテゴリ特徴量:
        各カテゴリをembeddingへ変換する。

    数値特徴量:
        各数値特徴量に専用embeddingを持たせ、値を掛けることでFMのfieldとして扱う。

    出力:
        linear項 + 2次相互作用項 をクラス数分出力する。
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
        self.num_fields = self.num_cat + self.num_numeric
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # カテゴリ特徴量のembedding
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])

        # 数値特徴量のembedding。各数値fieldに1つのベクトルを持つ。
        self.num_embeddings = nn.Parameter(torch.randn(num_numeric, embed_dim) * 0.01)

        # 1次項
        self.cat_linear = nn.ModuleList([
            nn.Embedding(cardinality, num_classes) for cardinality in cat_cardinalities
        ])
        self.num_linear = nn.Parameter(torch.randn(num_numeric, num_classes) * 0.01)

        # bias
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

        # 2次項用embedding
        cat_embs = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_embs.append(emb(x_cat[:, i]))
        cat_embs = torch.stack(cat_embs, dim=1)  # (B, C, D)

        # 数値embedding: value * field_embedding
        num_embs = x_num.unsqueeze(-1) * self.num_embeddings.unsqueeze(0)  # (B, N, D)

        all_embs = torch.cat([cat_embs, num_embs], dim=1)  # (B, F, D)
        all_embs = self.dropout(all_embs)

        # FM 2次相互作用: 0.5 * ((sum v)^2 - sum(v^2))
        summed = torch.sum(all_embs, dim=1)               # (B, D)
        summed_square = summed * summed                   # (B, D)
        square_summed = torch.sum(all_embs * all_embs, dim=1)  # (B, D)
        interaction = 0.5 * (summed_square - square_summed)    # (B, D)

        # interactionをクラスlogitへ変換するため、全クラスに同じFM相互作用表現を線形変換
        # 一般的なFMの多クラス版として、相互作用ベクトルをクラス分類に使う。
        interaction_logits = nn.functional.linear(
            interaction,
            torch.ones(self.num_classes, self.embed_dim, device=interaction.device) / self.embed_dim,
            None,
        )

        # 1次項
        linear_logits = self.bias.unsqueeze(0).expand(batch_size, -1)

        for i, lin in enumerate(self.cat_linear):
            linear_logits = linear_logits + lin(x_cat[:, i])

        numeric_linear_logits = torch.einsum("bn,nc->bc", x_num, self.num_linear)
        linear_logits = linear_logits + numeric_linear_logits

        logits = linear_logits + interaction_logits
        return logits


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

    return (
        total_loss / len(loader.dataset),
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(y_prob),
    )


def train_evaluate_fm(
    split: SplitData,
    output_dir: Path,
    seed: int = 42,
    batch_size: int = 512,
    epochs: int = 50,
    patience: int = 8,
    embed_dim: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> Dict[str, float]:
    """FMを学習・評価する。"""
    print("\n" + "=" * 80)
    print("Factorization Machine evaluation")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    encoder = FeatureEncoder(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    encoder.fit(split.train_df)

    train_ds = FMDataset(split.train_df, encoder)
    valid_ds = FMDataset(split.valid_df, encoder)
    test_ds = FMDataset(split.test_df, encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FactorizationMachineClassifier(
        cat_cardinalities=encoder.categorical_cardinalities(),
        num_numeric=len(NUMERIC_FEATURES),
        num_classes=len(LABELS),
        embed_dim=embed_dim,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0

    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, y_valid, pred_valid, prob_valid = evaluate_fm(model, valid_loader, criterion, device)
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
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "labels": LABELS,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
        },
        output_dir / "fm_best_model.pt",
    )

    test_loss, y_test, y_pred, y_prob = evaluate_fm(model, test_loader, criterion, device)
    metrics = save_report(output_dir, "fm", y_test, y_pred, y_prob)
    metrics["test_loss"] = float(test_loss)
    metrics["best_epoch"] = int(best_epoch)
    metrics["best_valid_loss"] = float(best_valid_loss)

    print("\n[FM Test Metrics]")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")

    return metrics


# ============================================================
# 5. main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/train_all_added_55_features.csv",
        help="55特徴量を追加したCSVファイルパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_lgbm_fm_55_features_4class_so_out",
        help="評価結果の出力先ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fm_epochs", type=int, default=50)
    parser.add_argument("--fm_batch_size", type=int, default=512)
    parser.add_argument("--fm_embed_dim", type=int, default=32)
    parser.add_argument("--fm_lr", type=float, default=1e-3)
    parser.add_argument("--fm_patience", type=int, default=8)
    parser.add_argument(
        "--skip_lightgbm",
        action="store_true",
        help="LightGBMをスキップする",
    )
    parser.add_argument(
        "--skip_fm",
        action="store_true",
        help="FMをスキップする",
    )

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LightGBM / FM comparison for 55-feature CSV / 4-class SO_OUT target / no class_weight")
    print("=" * 80)
    print(f"csv_path: {csv_path}")
    print(f"output_dir: {output_dir}")
    print(f"categorical_features: {len(CATEGORICAL_FEATURES)}")
    print(f"numeric_features: {len(NUMERIC_FEATURES)}")
    print(f"total_features: {len(CATEGORICAL_FEATURES) + len(NUMERIC_FEATURES)}")

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

    # 使用特徴量を保存
    with open(output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "base_numeric_features": BASE_NUMERIC_FEATURES,
                "hand_split_numeric_features": HAND_SPLIT_NUMERIC_FEATURES,
                "recent_numeric_features": RECENT_NUMERIC_FEATURES,
                "context_numeric_features": CONTEXT_NUMERIC_FEATURES,
                "cumulative_rate_numeric_features": CUMULATIVE_RATE_NUMERIC_FEATURES,
                "event_context_numeric_features": EVENT_CONTEXT_NUMERIC_FEATURES,
                "event_context_categorical_features": ["base_out_state"],
                "onbase_vs_context_numeric_features": ONBASE_VS_CONTEXT_NUMERIC_FEATURES,
                "onbase_vs_context_categorical_features": ONBASE_VS_CONTEXT_CATEGORICAL_FEATURES,
                "added_from_27_features": CUMULATIVE_RATE_NUMERIC_FEATURES,
                "added_from_37_features": EVENT_CONTEXT_NUMERIC_FEATURES + ["base_out_state"],
                "added_from_43_features": ONBASE_VS_CONTEXT_NUMERIC_FEATURES + ONBASE_VS_CONTEXT_CATEGORICAL_FEATURES,
                "removed_from_31_features": [
                    "is_bases_loaded",
                    "is_late_inning",
                    "is_scoring_position",
                    "score_state",
                ],
                "target_definition": {
                    "SINGLE": ["SINGLE"],
                    "XBH": ["DOUBLE", "TRIPLE", "HR"],
                    "BB_HBP": ["BB", "HBP"],
                    "SO_OUT": ["SO", "OUT"],
                },
                "class_weight": None,
                "labels": LABELS,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    all_metrics = []

    if not args.skip_lightgbm:
        lgb_metrics = train_evaluate_lightgbm(split, output_dir, seed=args.seed)
        if lgb_metrics is not None:
            all_metrics.append(lgb_metrics)

    if not args.skip_fm:
        fm_metrics = train_evaluate_fm(
            split,
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
        # model列を先頭へ
        cols = ["model"] + [c for c in comparison_df.columns if c != "model"]
        comparison_df = comparison_df[cols]
        comparison_df.to_csv(output_dir / "model_comparison_metrics.csv", index=False, encoding="utf-8-sig")

        print("\n" + "=" * 80)
        print("Model comparison")
        print("=" * 80)
        print(comparison_df.to_string(index=False))
        print(f"\nSaved: {output_dir / 'model_comparison_metrics.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
