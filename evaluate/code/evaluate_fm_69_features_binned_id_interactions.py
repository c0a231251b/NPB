# -*- coding: utf-8 -*-
"""
evaluate_fm_69_features_binned_id_interactions.py

train_all_added_69_features.csv を用いて、提案3の4分類
SINGLE / XBH / BB_HBP / SO_OUT に対するFM専用特徴量設計を比較するスクリプト。

比較する特徴量条件:
1. raw_69_features
   - 69特徴量そのままFM
   - カテゴリ特徴量 + 連続値特徴量を標準化して使用

2. binned_numeric_features
   - 連続値カテゴリ化FM
   - 69特徴量の数値特徴量をtrainデータ基準でbin化し、すべてカテゴリとして使用

3. binned_id_interactions
   - 連続値カテゴリ化 + ID相互作用特徴量FM
   - binned_numeric_features に加えて、batter_id×pitcher_id、batter_id×stadium_id などのID相互作用特徴量を追加

実行例:
python evaluate_fm_69_features_binned_id_interactions.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv"
python evaluate_fm_69_features_binned_id_interactions.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv"
軽く動作確認だけする場合:
python evaluate_fm_69_features_binned_id_interactions.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv" --dry_run

必要ライブラリ:
    pip install pandas numpy scikit-learn torch tqdm
"""

from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

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
# 1. ラベル定義
# ============================================================

LABELS = ["SINGLE", "XBH", "BB_HBP", "SO_OUT"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

SINGLE_SET = {"一安", "二安", "三安", "中安", "右安", "左安", "投安", "捕安", "遊安"}
DOUBLE_SET = {"中２", "二２", "右２", "左２", "投２", "遊２"}
TRIPLE_SET = {"中３", "右３", "左３"}
HR_SET = {"中本", "右本", "左本"}
BB_SET = {"四球", "敬遠"}
HBP_SET = {"死球"}


def convert_label(raw_label: str) -> str:
    x = str(raw_label).strip()
    if x in SINGLE_SET:
        return "SINGLE"
    if x in DOUBLE_SET or x in TRIPLE_SET or x in HR_SET:
        return "XBH"
    if x in BB_SET or x in HBP_SET:
        return "BB_HBP"
    return "SO_OUT"


# ============================================================
# 2. 69特徴量定義
# ============================================================

BASE_CATEGORICAL_FEATURES = [
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
    "batter_team_id",
    "pitcher_team_id",
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

BATTED_BALL_TEAM_NUMERIC_FEATURES = [
    "batter_groundout_rate",
    "batter_flyout_rate",
    "batter_gdp_rate",
    "pitcher_groundout_rate",
    "pitcher_flyout_rate",
    "pitcher_gdp_rate",
    "vs_hr_rate",
    "is_cleanup",
    "is_bottom_order",
    "days_since_last_game_batter",
    "days_since_last_game_pitcher",
    "pitcher_batter_times_faced_in_game",
]

INTERLEAGUE_SEASON_PROGRESS_NUMERIC_FEATURES = [
    "is_interleague",
    "season_progress",
]

NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + HAND_SPLIT_NUMERIC_FEATURES
    + RECENT_NUMERIC_FEATURES
    + CONTEXT_NUMERIC_FEATURES
    + CUMULATIVE_RATE_NUMERIC_FEATURES
    + EVENT_CONTEXT_NUMERIC_FEATURES
    + ONBASE_VS_CONTEXT_NUMERIC_FEATURES
    + BATTED_BALL_TEAM_NUMERIC_FEATURES
    + INTERLEAGUE_SEASON_PROGRESS_NUMERIC_FEATURES
)

BINNED_FEATURES = [f"{col}_bin" for col in NUMERIC_FEATURES]

ID_INTERACTION_FEATURES = [
    "batter_pitcher_pair",
    "batter_stadium_pair",
    "pitcher_stadium_pair",
    "batter_pitcher_stadium_triplet",
    "batter_team_pitcher_pair",
    "pitcher_team_batter_pair",
    "batter_team_pitcher_team_pair",
    "batter_hand_pitcher_hand_pair",
    "batter_base_out_pair",
    "pitcher_base_out_pair",
    "batter_inning_out_pair",
    "pitcher_inning_out_pair",
    "batter_previous_result1_pair",
    "pitcher_previous_result1_pair",
    "stadium_base_out_pair",
    "stadium_inning_group_pair",
    "batter_score_state_pair",
    "pitcher_score_state_pair",
    "batter_bat_order_group_pair",
    "pitcher_bat_order_group_pair",
]


# ============================================================
# 3. データ構造
# ============================================================

@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


@dataclass
class FeatureVariant:
    name: str
    categorical_features: List[str]
    numeric_features: List[str]
    description: str


# ============================================================
# 4. 共通処理
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_top_bottom(value: Any) -> float:
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

    required = set(BASE_CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["game_date", "label"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"必要な列がCSVに存在しません: {missing}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df["target_label"] = df["label"].apply(convert_label)
    df["target"] = df["target_label"].map(LABEL_TO_ID).astype(int)
    df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BASE_CATEGORICAL_FEATURES:
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

    return SplitData(
        train_df=df[df["game_date"].isin(train_dates)].copy(),
        valid_df=df[df["game_date"].isin(valid_dates)].copy(),
        test_df=df[df["game_date"].isin(test_dates)].copy(),
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
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


def calc_non_so_out_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    so_idx = LABEL_TO_ID["SO_OUT"]
    true_non = y_true != so_idx
    pred_non = y_pred != so_idx
    tp = int(np.sum(true_non & pred_non))
    fp = int(np.sum(~true_non & pred_non))
    fn = int(np.sum(true_non & ~pred_non))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "non_so_out_precision": float(precision),
        "non_so_out_recall": float(recall),
        "non_so_out_f1": float(f1),
        "non_so_out_pred_count": int(np.sum(pred_non)),
    }


def make_class_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.arange(len(LABELS))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return weights.astype(np.float32)


# ============================================================
# 5. bin化・相互作用特徴量
# ============================================================

def make_score_state_from_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    return np.select(
        [x <= -4, x < 0, x == 0, x <= 3, x >= 4],
        ["behind_big", "behind", "tie", "lead", "lead_big"],
        default="unknown",
    )


def fixed_bin_column(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    """意味が明確な列は固定ルールでbin化する。該当しない場合はNone。"""
    x = pd.to_numeric(df[col], errors="coerce")

    if col == "inning":
        return pd.cut(x, bins=[-np.inf, 3, 6, 9, np.inf], labels=["early", "middle", "late", "extra"]).astype(str)
    if col == "bat_order":
        return pd.cut(x, bins=[-np.inf, 3, 6, 9, np.inf], labels=["top", "middle", "bottom", "unknown"]).astype(str)
    if col == "score_diff":
        return pd.Series(make_score_state_from_series(x), index=df.index).astype(str)
    if col in ["top_bottom", "out_count", "runner1", "runner2", "runner3", "runner_count", "is_leadoff_batter_in_inning", "is_no_out_runner_on", "is_two_out", "is_cleanup", "is_bottom_order", "is_interleague"]:
        return x.fillna(-1).round().astype(int).astype(str)
    if col == "month":
        return x.fillna(-1).round().astype(int).astype(str)
    if col in ["batting_turn_in_game", "pitcher_bf_in_game", "pitcher_batter_times_faced_in_game"]:
        return pd.cut(x, bins=[-np.inf, 1, 2, 3, np.inf], labels=["1", "2", "3", "4plus"]).astype(str)
    if col in ["batter_pa_count", "pitcher_bf_count", "vs_pa_count"]:
        return pd.cut(x, bins=[-np.inf, 0, 10, 30, 80, 150, np.inf], labels=["0", "1_10", "11_30", "31_80", "81_150", "151plus"]).astype(str)
    if col in ["days_since_last_game_batter", "days_since_last_game_pitcher"]:
        return pd.cut(x, bins=[-np.inf, 0, 1, 3, 7, np.inf], labels=["same_day", "1d", "2_3d", "4_7d", "8d_plus"]).astype(str)
    if col == "season_progress":
        return pd.cut(x, bins=[-np.inf, 0.33, 0.66, np.inf], labels=["early", "middle", "late"]).astype(str)

    return None


def fit_quantile_edges(train_s: pd.Series, n_bins: int = 5) -> np.ndarray:
    x = pd.to_numeric(train_s, errors="coerce").dropna().astype(float)
    if x.empty or x.nunique() <= 1:
        return np.array([], dtype=float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) <= 2:
        return np.array([], dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(float)


def apply_quantile_edges(s: pd.Series, edges: np.ndarray) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if len(edges) <= 2:
        return x.fillna(-999999).round(6).astype(str)
    labels = [f"q{i+1}" for i in range(len(edges) - 1)]
    out = pd.cut(x, bins=edges, labels=labels, include_lowest=True).astype(str)
    return out.where(out != "nan", "missing")


def add_binned_features(split: SplitData, n_bins: int = 5) -> Tuple[SplitData, Dict[str, Dict[str, Any]]]:
    """trainだけでbin境界を決め、valid/testへ同じbinを適用する。"""
    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()
    bin_info: Dict[str, Dict[str, Any]] = {}

    for col in NUMERIC_FEATURES:
        bin_col = f"{col}_bin"

        fixed_train = fixed_bin_column(train_df, col)
        if fixed_train is not None:
            train_df[bin_col] = fixed_train.fillna("missing").astype(str)
            valid_df[bin_col] = fixed_bin_column(valid_df, col).fillna("missing").astype(str)
            test_df[bin_col] = fixed_bin_column(test_df, col).fillna("missing").astype(str)
            bin_info[col] = {"method": "fixed_rule", "bin_col": bin_col}
            continue

        edges = fit_quantile_edges(train_df[col], n_bins=n_bins)
        train_df[bin_col] = apply_quantile_edges(train_df[col], edges).astype(str)
        valid_df[bin_col] = apply_quantile_edges(valid_df[col], edges).astype(str)
        test_df[bin_col] = apply_quantile_edges(test_df[col], edges).astype(str)
        bin_info[col] = {"method": "train_quantile", "bin_col": bin_col, "edges": edges.tolist()}

    return SplitData(train_df=train_df, valid_df=valid_df, test_df=test_df), bin_info


def safe_combine(df: pd.DataFrame, cols: List[str], sep: str = "__") -> pd.Series:
    parts = []
    for col in cols:
        if col not in df.columns:
            parts.append(pd.Series("<NA>", index=df.index))
        else:
            parts.append(df[col].fillna("<NA>").astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out + sep + p
    return out.astype(str)


def add_id_interaction_features(split: SplitData) -> SplitData:
    """FMのID×ID相互作用を直接表すカテゴリ特徴量を追加する。"""
    def add(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["batter_pitcher_pair"] = safe_combine(df, ["batter_id", "pitcher_id"])
        df["batter_stadium_pair"] = safe_combine(df, ["batter_id", "stadium_id"])
        df["pitcher_stadium_pair"] = safe_combine(df, ["pitcher_id", "stadium_id"])
        df["batter_pitcher_stadium_triplet"] = safe_combine(df, ["batter_id", "pitcher_id", "stadium_id"])
        df["batter_team_pitcher_pair"] = safe_combine(df, ["batter_team_id", "pitcher_id"])
        df["pitcher_team_batter_pair"] = safe_combine(df, ["pitcher_team_id", "batter_id"])
        df["batter_team_pitcher_team_pair"] = safe_combine(df, ["batter_team_id", "pitcher_team_id"])
        df["batter_hand_pitcher_hand_pair"] = safe_combine(df, ["batter_hand", "pitcher_hand"])
        df["batter_base_out_pair"] = safe_combine(df, ["batter_id", "base_out_state"])
        df["pitcher_base_out_pair"] = safe_combine(df, ["pitcher_id", "base_out_state"])
        df["batter_inning_out_pair"] = safe_combine(df, ["batter_id", "inning_out_state"])
        df["pitcher_inning_out_pair"] = safe_combine(df, ["pitcher_id", "inning_out_state"])
        df["batter_previous_result1_pair"] = safe_combine(df, ["batter_id", "previous_result1"])
        df["pitcher_previous_result1_pair"] = safe_combine(df, ["pitcher_id", "previous_result1"])
        df["stadium_base_out_pair"] = safe_combine(df, ["stadium_id", "base_out_state"])
        df["stadium_inning_group_pair"] = safe_combine(df, ["stadium_id", "inning_bin"])
        df["batter_score_state_pair"] = safe_combine(df, ["batter_id", "score_diff_bin"])
        df["pitcher_score_state_pair"] = safe_combine(df, ["pitcher_id", "score_diff_bin"])
        df["batter_bat_order_group_pair"] = safe_combine(df, ["batter_id", "bat_order_bin"])
        df["pitcher_bat_order_group_pair"] = safe_combine(df, ["pitcher_id", "bat_order_bin"])
        return df

    return SplitData(
        train_df=add(split.train_df),
        valid_df=add(split.valid_df),
        test_df=add(split.test_df),
    )


def create_variants() -> List[FeatureVariant]:
    return [
        FeatureVariant(
            name="raw_69_features",
            categorical_features=BASE_CATEGORICAL_FEATURES,
            numeric_features=NUMERIC_FEATURES,
            description="69特徴量そのままFM。カテゴリ特徴量 + 連続値特徴量を標準化して使用。",
        ),
        FeatureVariant(
            name="binned_numeric_features",
            categorical_features=BASE_CATEGORICAL_FEATURES + BINNED_FEATURES,
            numeric_features=[],
            description="連続値カテゴリ化FM。数値特徴量をbin化し、すべてカテゴリとして使用。",
        ),
        FeatureVariant(
            name="binned_id_interactions",
            categorical_features=BASE_CATEGORICAL_FEATURES + BINNED_FEATURES + ID_INTERACTION_FEATURES,
            numeric_features=[],
            description="連続値カテゴリ化 + ID相互作用特徴量FM。",
        ),
    ]


# ============================================================
# 6. FM用エンコーダ・Dataset・モデル
# ============================================================

class FeatureEncoder:
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
            self.category_maps[col] = {v: i + 1 for i, v in enumerate(values)}  # 0 = unknown

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
        if not self.categorical_features:
            return np.zeros((len(df), 0), dtype=np.int64)
        arrays = []
        for col in self.categorical_features:
            mp = self.category_maps[col]
            arr = df[col].fillna("<NA>").astype(str).map(lambda x: mp.get(x, 0)).astype(np.int64).values
            arrays.append(arr)
        return np.stack(arrays, axis=1)

    def transform_numeric(self, df: pd.DataFrame) -> np.ndarray:
        if not self.numeric_features:
            return np.zeros((len(df), 0), dtype=np.float32)
        arrays = []
        for col in self.numeric_features:
            s = pd.to_numeric(df[col], errors="coerce").fillna(self.numeric_median[col])
            arr = ((s - self.numeric_mean[col]) / self.numeric_std[col]).astype(np.float32).values
            arrays.append(arr)
        return np.stack(arrays, axis=1).astype(np.float32)

    def categorical_cardinalities(self) -> List[int]:
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

        if self.num_fields == 0:
            raise ValueError("FMには少なくとも1つの特徴量が必要です。")

        self.cat_embeddings = nn.ModuleList([nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities])
        self.cat_linear = nn.ModuleList([nn.Embedding(cardinality, num_classes) for cardinality in cat_cardinalities])

        if num_numeric > 0:
            self.num_embeddings = nn.Parameter(torch.randn(num_numeric, embed_dim) * 0.01)
            self.num_linear = nn.Parameter(torch.randn(num_numeric, num_classes) * 0.01)
        else:
            self.num_embeddings = None
            self.num_linear = None

        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.cat_linear:
            nn.init.zeros_(emb.weight)
        if self.num_embeddings is not None:
            nn.init.xavier_uniform_(self.num_embeddings)
        if self.num_linear is not None:
            nn.init.xavier_uniform_(self.num_linear)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size = x_cat.size(0)
        embs = []

        for i, emb in enumerate(self.cat_embeddings):
            embs.append(emb(x_cat[:, i]))

        if self.num_numeric > 0:
            num_embs = x_num.unsqueeze(-1) * self.num_embeddings.unsqueeze(0)  # (B, N, D)
            for j in range(self.num_numeric):
                embs.append(num_embs[:, j, :])

        all_embs = torch.stack(embs, dim=1)  # (B, F, D)
        all_embs = self.dropout(all_embs)

        summed = torch.sum(all_embs, dim=1)
        interaction = 0.5 * (summed * summed - torch.sum(all_embs * all_embs, dim=1))
        interaction_logits = nn.functional.linear(
            interaction,
            torch.ones(self.num_classes, self.embed_dim, device=interaction.device) / self.embed_dim,
            None,
        )

        linear_logits = self.bias.unsqueeze(0).expand(batch_size, -1)
        for i, lin in enumerate(self.cat_linear):
            linear_logits = linear_logits + lin(x_cat[:, i])

        if self.num_numeric > 0:
            linear_logits = linear_logits + torch.einsum("bn,nc->bc", x_num, self.num_linear)

        return linear_logits + interaction_logits


# ============================================================
# 7. 学習・評価
# ============================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, grad_clip: float = 5.0) -> Tuple[float, float]:
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
    return total_loss / len(loader.dataset), accuracy_score(np.concatenate(y_true), np.concatenate(y_pred))


@torch.no_grad()
def evaluate_fm(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
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


def save_variant_report(output_dir: Path, variant_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    variant_dir = output_dir / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics.update(calc_non_so_out_metrics(y_true, y_pred))

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0,
        digits=6,
    )
    (variant_dir / f"{variant_name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        variant_dir / f"{variant_name}_confusion_matrix.csv", encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame({
        "true_id": y_true,
        "pred_id": y_pred,
        "true_label": [ID_TO_LABEL[int(x)] for x in y_true],
        "pred_label": [ID_TO_LABEL[int(x)] for x in y_pred],
    })
    for i, label in enumerate(LABELS):
        pred_df[f"prob_{label}"] = y_prob[:, i]
    pred_df.to_csv(variant_dir / f"{variant_name}_predictions.csv", index=False, encoding="utf-8-sig")

    return metrics


def run_variant(
    variant: FeatureVariant,
    split: SplitData,
    output_dir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    embed_dim: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, float]:
    print("\n" + "-" * 80)
    print(f"FM evaluation: variant={variant.name}")
    print("-" * 80)
    print(f"description: {variant.description}")
    print(f"categorical_features ({len(variant.categorical_features)}): {variant.categorical_features}")
    print(f"numeric_features ({len(variant.numeric_features)}): {variant.numeric_features}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # 特徴量一覧を保存
    feature_rows = []
    for f in variant.categorical_features:
        feature_rows.append({"feature": f, "feature_type": "categorical"})
    for f in variant.numeric_features:
        feature_rows.append({"feature": f, "feature_type": "numeric"})
    pd.DataFrame(feature_rows).to_csv(variant_dir / f"{variant.name}_features.csv", index=False, encoding="utf-8-sig")

    # 高カーディナリティ確認用
    card_rows = []
    for f in variant.categorical_features:
        card_rows.append({"feature": f, "train_unique": int(split.train_df[f].astype(str).nunique())})
    pd.DataFrame(card_rows).sort_values("train_unique", ascending=False).to_csv(
        variant_dir / f"{variant.name}_categorical_cardinality.csv", index=False, encoding="utf-8-sig"
    )

    encoder = FeatureEncoder(variant.categorical_features, variant.numeric_features)
    encoder.fit(split.train_df)

    train_ds = FMDataset(split.train_df, encoder)
    valid_ds = FMDataset(split.valid_df, encoder)
    test_ds = FMDataset(split.test_df, encoder)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FactorizationMachineClassifier(
        cat_cardinalities=encoder.categorical_cardinalities(),
        num_numeric=len(variant.numeric_features),
        num_classes=len(LABELS),
        embed_dim=embed_dim,
        dropout=0.1,
    ).to(device)

    y_train = split.train_df["target"].values
    class_weights_np = make_class_weights(y_train)
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
            f"epoch={epoch:03d} train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} "
            f"valid_acc={valid_metrics['accuracy']:.5f} valid_macro_f1={valid_metrics['macro_f1']:.5f}"
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

    pd.DataFrame(history).to_csv(variant_dir / f"{variant.name}_history.csv", index=False, encoding="utf-8-sig")

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": variant.name,
            "categorical_features": variant.categorical_features,
            "numeric_features": variant.numeric_features,
            "labels": LABELS,
            "class_weights": class_weights_np.tolist(),
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
        },
        variant_dir / f"{variant.name}_fm_best_model.pt",
    )

    test_loss, y_test, y_pred, y_prob = evaluate_fm(model, test_loader, criterion, device)
    metrics = save_variant_report(output_dir, variant.name, y_test, y_pred, y_prob)
    metrics.update({
        "variant": variant.name,
        "model": "fm",
        "test_loss": float(test_loss),
        "best_epoch": int(best_epoch),
        "best_valid_loss": float(best_valid_loss),
        "n_categorical_features": len(variant.categorical_features),
        "n_numeric_features": len(variant.numeric_features),
        "n_total_features": len(variant.categorical_features) + len(variant.numeric_features),
    })

    print("\n[FM Test Metrics]")
    for k, v in metrics.items():
        if k not in ["variant", "model"]:
            print(f"{k}: {v}")

    return metrics


# ============================================================
# 8. main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="data/train_all_added_69_features.csv")
    parser.add_argument("--output_dir", type=str, default="output_fm_69_features_binned_id_interactions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fm_epochs", type=int, default=50)
    parser.add_argument("--fm_batch_size", type=int, default=512)
    parser.add_argument("--fm_embed_dim", type=int, default=32)
    parser.add_argument("--fm_lr", type=float, default=1e-3)
    parser.add_argument("--fm_weight_decay", type=float, default=1e-5)
    parser.add_argument("--fm_patience", type=int, default=8)
    parser.add_argument("--bin_count", type=int, default=5)
    parser.add_argument("--dry_run", action="store_true", help="特徴量生成と分割だけ確認して終了する")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FM comparison for 69-feature PA outcome prediction / binned numeric / ID interactions")
    print("=" * 80)
    print(f"csv_path: {csv_path}")
    print(f"output_dir: {output_dir}")

    df = prepare_dataframe(csv_path)
    split = split_by_game_date(df)

    print("\n[Data split]")
    print(f"all  : {len(df):>6} rows, dates={df['game_date'].nunique()}")
    print(f"train: {len(split.train_df):>6} rows, {split.train_df['game_date'].min().date()} ~ {split.train_df['game_date'].max().date()}")
    print(f"valid: {len(split.valid_df):>6} rows, {split.valid_df['game_date'].min().date()} ~ {split.valid_df['game_date'].max().date()}")
    print(f"test : {len(split.test_df):>6} rows, {split.test_df['game_date'].min().date()} ~ {split.test_df['game_date'].max().date()}")

    dist = df["target_label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print("\n[Target distribution]")
    print(dist)
    dist.to_csv(output_dir / "target_distribution.csv", encoding="utf-8-sig")

    # train基準でbin化 → ID相互作用追加
    split_binned, bin_info = add_binned_features(split, n_bins=args.bin_count)
    split_all = add_id_interaction_features(split_binned)

    with open(output_dir / "bin_info.json", "w", encoding="utf-8") as f:
        json.dump(bin_info, f, ensure_ascii=False, indent=2)

    variants = create_variants()

    variant_summary = []
    for variant in variants:
        variant_summary.append({
            "variant": variant.name,
            "description": variant.description,
            "n_categorical_features": len(variant.categorical_features),
            "n_numeric_features": len(variant.numeric_features),
            "n_total_features": len(variant.categorical_features) + len(variant.numeric_features),
        })
    variant_summary_df = pd.DataFrame(variant_summary)
    variant_summary_df.to_csv(output_dir / "feature_variant_summary.csv", index=False, encoding="utf-8-sig")

    print("\n[Feature variants]")
    print(variant_summary_df.to_string(index=False))

    if args.dry_run:
        print("\nDry run finished. No training executed.")
        return

    all_metrics = []
    for variant in variants:
        metrics = run_variant(
            variant=variant,
            split=split_all,
            output_dir=output_dir,
            seed=args.seed,
            batch_size=args.fm_batch_size,
            epochs=args.fm_epochs,
            patience=args.fm_patience,
            embed_dim=args.fm_embed_dim,
            lr=args.fm_lr,
            weight_decay=args.fm_weight_decay,
        )
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics)
    front_cols = ["variant", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "top2_accuracy", "top3_accuracy", "non_so_out_precision", "non_so_out_recall", "non_so_out_f1", "test_loss", "best_epoch", "best_valid_loss", "n_categorical_features", "n_numeric_features", "n_total_features"]
    cols = [c for c in front_cols if c in comparison_df.columns] + [c for c in comparison_df.columns if c not in front_cols]
    comparison_df = comparison_df[cols]
    comparison_df.to_csv(output_dir / "fm_69_binned_id_interactions_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("FM feature-design comparison")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {output_dir / 'fm_69_binned_id_interactions_comparison_metrics.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
