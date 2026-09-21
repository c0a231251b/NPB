# -*- coding: utf-8 -*-
"""
Step 3: Step2AベースのFM向け bin + クロス特徴量比較スクリプト。

目的:
- イニング得点有無予測（NO_RUN / RUN_SCORED）に対して、Step2A代表特徴量
  lineup_vs_pitcher_hand_32_features をベースにする。
- FMが扱いやすいように連続値をbin化し、カテゴリ同士のクロス特徴量を追加する。
- 同じ時系列分割で以下を比較する。
    1. LightGBM: lgbm_step2a_raw_32_features（参考代表）
    2. FM: fm_step2a_raw_32_features
    3. FM: fm_step2a_binned_features
    4. FM: fm_step2a_binned_cross_features

入力:
1) inning CSV:
   train_inning_runs_added_9_features.csv
2) PA CSV:
   train_all_added_69_features.csv

実行例:
python evaluate_step3_fm_binned_cross_step2a.py --inning_csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" --pa_csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv"
python evaluate_step3_fm_binned_cross_step2a.py \
  --inning_csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" \
  --pa_csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv"

軽い確認:
python evaluate_step3_fm_binned_cross_step2a.py \
  --inning_csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" \
  --pa_csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv" \
  --dry_run

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm torch
"""
from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
    auc,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
LABELS = ["NO_RUN", "RUN_SCORED"]
LABEL_TO_ID = {"NO_RUN": 0, "RUN_SCORED": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# ============================================================
# Step2A feature definitions
# ============================================================

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
    "inning",
    "top_bottom",
    "is_late_inning",
    "batting_order_start",
    "is_top_order_start",
    "is_cleanup_start",
    "pitcher_is_starter",
    "pitcher_times_through_order",
    "pitcher_era_before_inning",
    "opponent_starter_era",
    "team_avg_runs_7d",
    "opponent_avg_runs_7d",
    "opponent_avg_runs_allowed_7d",
    "team_run_scored_inning_rate_7d",
    "opponent_run_allowed_inning_rate_7d",
    "team_recent_2plus_inning_rate_7d",
    "opponent_recent_2plus_allowed_rate_7d",
]

SCORE_CONTEXT_CATEGORICAL_FEATURES = ["score_state_before_inning"]
SCORE_CONTEXT_NUMERIC_FEATURES = ["score_diff_before_inning", "is_close_game", "is_blowout"]

LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES = [
    "next_3_batters_vs_pitcher_hand_avg",
    "next_3_batters_vs_pitcher_hand_ops",
    "next_4_batters_vs_pitcher_hand_ops",
    "next_3_batters_vs_pitcher_hand_xbh_rate",
]

STEP2A_CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES + SCORE_CONTEXT_CATEGORICAL_FEATURES
STEP2A_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + SCORE_CONTEXT_NUMERIC_FEATURES + LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES

PA_LINEUP_VALUE_COLUMNS = [
    "batter_obp",
    "batter_ops",
    "batter_xbh_rate",
    "batter_vs_rhp_avg",
    "batter_vs_rhp_ops",
    "batter_vs_lhp_avg",
    "batter_vs_lhp_ops",
]

@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame

@dataclass
class FeatureVariant:
    name: str
    model: str
    categorical_features: List[str]
    numeric_features: List[str]
    description: str

# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = SEED) -> None:
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


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


def validate_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} に必要な列がありません: {missing}")


def create_score_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    diff = pd.to_numeric(df["score_diff_before_inning"], errors="coerce").fillna(0)
    df["score_diff_before_inning"] = diff
    df["score_state_before_inning"] = np.where(diff > 0, "lead", np.where(diff < 0, "behind", "tie"))
    abs_diff = diff.abs()
    df["is_close_game"] = (abs_diff <= 3).astype(int)
    df["is_blowout"] = (abs_diff >= 6).astype(int)
    return df


def split_by_game_date(df: pd.DataFrame, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> SplitData:
    unique_dates = sorted(df["game_date"].dropna().unique())
    if len(unique_dates) < 3:
        raise ValueError("game_date のユニーク数が少なすぎるため、分割できません。")
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


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: Optional[np.ndarray] = None) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    result: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "no_run_precision": float(per_p[0]),
        "no_run_recall": float(per_r[0]),
        "no_run_f1": float(per_f1[0]),
        "run_scored_precision": float(per_p[1]),
        "run_scored_recall": float(per_r[1]),
        "run_scored_f1": float(per_f1[1]),
        "run_scored_pred_count": int(np.sum(y_pred == 1)),
    }
    if y_prob_pos is not None:
        eps = 1e-15
        y_prob_2d = np.vstack([1.0 - y_prob_pos, y_prob_pos]).T
        y_prob_2d = np.clip(y_prob_2d, eps, 1.0 - eps)
        result["log_loss"] = float(log_loss(y_true, y_prob_2d, labels=[0, 1]))
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_prob_pos))
        except ValueError:
            result["roc_auc"] = float("nan")
        try:
            result["pr_auc"] = float(average_precision_score(y_true, y_prob_pos))
        except ValueError:
            precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
            result["pr_auc"] = float(auc(recall, precision))
    return result

# ============================================================
# Step2A lineup feature engineering
# ============================================================

def build_lineup_lookup(pa_df: pd.DataFrame) -> Tuple[Dict[Tuple[pd.Timestamp, str, int, int], Dict[str, float]], pd.DataFrame]:
    pa = pa_df.copy()
    pa["game_date"] = normalize_date(pa["game_date"])
    pa["top_bottom"] = pa["top_bottom"].apply(normalize_top_bottom).astype("Int64")
    pa["bat_order"] = pd.to_numeric(pa["bat_order"], errors="coerce").astype("Int64")
    for col in PA_LINEUP_VALUE_COLUMNS:
        pa[col] = pd.to_numeric(pa[col], errors="coerce")
    for c in ["inning", "out_count"]:
        if c in pa.columns:
            pa[c] = pd.to_numeric(pa[c], errors="coerce")
        else:
            pa[c] = 0
    sort_cols = ["game_date", "stadium_id", "top_bottom", "bat_order", "inning", "out_count"]
    lineup_cols = ["game_date", "stadium_id", "top_bottom", "bat_order"] + PA_LINEUP_VALUE_COLUMNS
    lineup = (
        pa.sort_values(sort_cols)
        .dropna(subset=["game_date", "stadium_id", "top_bottom", "bat_order"])
        .groupby(["game_date", "stadium_id", "top_bottom", "bat_order"], as_index=False)
        .first()[lineup_cols]
    )
    lookup: Dict[Tuple[pd.Timestamp, str, int, int], Dict[str, float]] = {}
    for row in lineup.itertuples(index=False):
        key = (row.game_date, str(row.stadium_id), int(row.top_bottom), int(row.bat_order))
        lookup[key] = {col: float(getattr(row, col)) if pd.notna(getattr(row, col)) else np.nan for col in PA_LINEUP_VALUE_COLUMNS}
    return lookup, lineup


def next_orders(start_order: int, n: int) -> List[int]:
    return [((start_order - 1 + i) % 9) + 1 for i in range(n)]


def normalize_pitcher_hand(value) -> str:
    if pd.isna(value):
        return "unknown"
    s = str(value).strip().upper()
    if s in ["R", "右", "RIGHT", "RIGHTY"]:
        return "R"
    if s in ["L", "左", "LEFT", "LEFTY"]:
        return "L"
    return "unknown"


def build_pitcher_hand_lookup(pa_df: pd.DataFrame) -> Dict[str, str]:
    validate_required_columns(pa_df, ["pitcher_id", "pitcher_hand"], "PA CSV")
    tmp = pa_df[["pitcher_id", "pitcher_hand"]].copy()
    tmp["pitcher_id"] = tmp["pitcher_id"].fillna("<NA>").astype(str)
    tmp["pitcher_hand"] = tmp["pitcher_hand"].apply(normalize_pitcher_hand)
    tmp = tmp[tmp["pitcher_hand"].isin(["R", "L"])]
    hand_lookup: Dict[str, str] = {}
    for pid, g in tmp.groupby("pitcher_id"):
        mode = g["pitcher_hand"].mode()
        hand_lookup[str(pid)] = str(mode.iloc[0]) if len(mode) > 0 else "unknown"
    return hand_lookup


def add_lineup_vs_pitcher_hand_features(inning_df: pd.DataFrame, pa_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    validate_required_columns(
        pa_df,
        ["game_date", "stadium_id", "top_bottom", "bat_order", "pitcher_id", "pitcher_hand"] + PA_LINEUP_VALUE_COLUMNS,
        "PA CSV",
    )
    validate_required_columns(
        inning_df,
        ["game_date", "stadium_id", "top_bottom", "batting_order_start", "current_pitcher_id"],
        "inning CSV",
    )
    lookup, lineup_table = build_lineup_lookup(pa_df)
    pitcher_hand_lookup = build_pitcher_hand_lookup(pa_df)

    df = inning_df.copy()
    df["game_date"] = normalize_date(df["game_date"])
    df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom).astype(int)
    df["batting_order_start"] = pd.to_numeric(df["batting_order_start"], errors="coerce").fillna(1).astype(int)
    df["current_pitcher_id"] = df["current_pitcher_id"].fillna("<NA>").astype(str)
    df["current_pitcher_hand"] = df["current_pitcher_id"].map(lambda x: pitcher_hand_lookup.get(str(x), "unknown"))

    rows: List[Dict[str, float]] = []
    for row in df.itertuples(index=False):
        game_date = row.game_date
        stadium_id = str(row.stadium_id)
        tb = int(row.top_bottom)
        start_order = int(row.batting_order_start)
        pitcher_hand = str(row.current_pitcher_hand)

        vals3_vs_avg: List[float] = []
        vals3_vs_ops: List[float] = []
        vals4_vs_ops: List[float] = []
        vals3_vs_xbh: List[float] = []

        for i, order in enumerate(next_orders(start_order, 4)):
            key = (game_date, stadium_id, tb, order)
            val = lookup.get(key)
            if val is None:
                if i < 3:
                    vals3_vs_avg.append(np.nan)
                    vals3_vs_ops.append(np.nan)
                    vals3_vs_xbh.append(np.nan)
                vals4_vs_ops.append(np.nan)
                continue

            if pitcher_hand == "R":
                vs_avg = val.get("batter_vs_rhp_avg", np.nan)
                vs_ops = val.get("batter_vs_rhp_ops", np.nan)
            elif pitcher_hand == "L":
                vs_avg = val.get("batter_vs_lhp_avg", np.nan)
                vs_ops = val.get("batter_vs_lhp_ops", np.nan)
            else:
                vs_avg = np.nan
                vs_ops = np.nan

            if i < 3:
                vals3_vs_avg.append(vs_avg)
                vals3_vs_ops.append(vs_ops)
                vals3_vs_xbh.append(val.get("batter_xbh_rate", np.nan))
            vals4_vs_ops.append(vs_ops)

        rows.append({
            "next_3_batters_vs_pitcher_hand_avg": float(np.nanmean(vals3_vs_avg)) if np.isfinite(vals3_vs_avg).any() else np.nan,
            "next_3_batters_vs_pitcher_hand_ops": float(np.nanmean(vals3_vs_ops)) if np.isfinite(vals3_vs_ops).any() else np.nan,
            "next_4_batters_vs_pitcher_hand_ops": float(np.nanmean(vals4_vs_ops)) if np.isfinite(vals4_vs_ops).any() else np.nan,
            "next_3_batters_vs_pitcher_hand_xbh_rate": float(np.nanmean(vals3_vs_xbh)) if np.isfinite(vals3_vs_xbh).any() else np.nan,
            "next_3_batters_vs_hand_known_count": int(np.sum(pd.notna(vals3_vs_ops))),
            "next_4_batters_vs_hand_known_count": int(np.sum(pd.notna(vals4_vs_ops))),
        })
    feat_df = pd.DataFrame(rows)
    for col in feat_df.columns:
        df[col] = feat_df[col].values
    return df, lineup_table


def prepare_dataframe(inning_csv_path: Path, pa_csv_path: Path, output_dir: Path, save_augmented_csv: bool = True) -> pd.DataFrame:
    inning_df = pd.read_csv(inning_csv_path)
    pa_df = pd.read_csv(pa_csv_path)
    required_inning = sorted(set(
        BASE_CATEGORICAL_FEATURES
        + BASE_NUMERIC_FEATURES
        + ["game_date", "game_id", "target_run_scored", "score_diff_before_inning", "stadium_id", "top_bottom", "batting_order_start"]
    ))
    validate_required_columns(inning_df, required_inning, "inning CSV")
    df = inning_df.copy()
    df["game_date"] = normalize_date(df["game_date"])
    df["target"] = df["target_run_scored"].map(LABEL_TO_ID).astype(int)
    df["target_label"] = df["target"].map(ID_TO_LABEL)
    df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)
    df = create_score_context_features(df)
    df, lineup_table = add_lineup_vs_pitcher_hand_features(df, pa_df)

    # Type conversion
    for col in STEP2A_CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("<NA>").astype(str)
    for col in STEP2A_NUMERIC_FEATURES + ["next_3_batters_vs_hand_known_count", "next_4_batters_vs_hand_known_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = [c for c in ["game_date", "game_id", "team_id", "inning", "top_bottom"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    lineup_table.to_csv(output_dir / "step3_lineup_slot_table_from_pa.csv", index=False, encoding="utf-8-sig")
    if save_augmented_csv:
        df.to_csv(output_dir / "train_inning_runs_step3_step2a_lineup_features.csv", index=False, encoding="utf-8-sig")
    return df

# ============================================================
# FM-friendly bins and crosses
# ============================================================

def to_num_series(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def bin_by_edges(x: float, edges: List[float], labels: List[str]) -> str:
    if pd.isna(x):
        return "missing"
    for edge, label in zip(edges, labels):
        if x <= edge:
            return label
    return labels[-1]


def fixed_rate_bin(x: float) -> str:
    return bin_by_edges(x, [0.0, 0.05, 0.10, 0.20], ["zero", "very_low", "low", "mid", "high"])


def fixed_ops_bin(x: float) -> str:
    return bin_by_edges(x, [0.50, 0.65, 0.75, 0.85], ["very_low", "low", "mid", "high", "elite"])


def fixed_avg_bin(x: float) -> str:
    return bin_by_edges(x, [0.18, 0.23, 0.27, 0.31], ["very_low", "low", "mid", "high", "elite"])


def fixed_runs_bin(x: float) -> str:
    return bin_by_edges(x, [0.0, 2.5, 4.5, 6.0], ["zero", "low", "mid", "high", "very_high"])


def fixed_era_bin(x: float) -> str:
    return bin_by_edges(x, [0.0, 2.5, 4.0, 6.0], ["zero", "low", "mid", "high", "very_high"])


def score_diff_bin(x: float) -> str:
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


def batting_order_group(x: float) -> str:
    if pd.isna(x):
        return "missing"
    x = int(x)
    if x in [1, 2, 3]:
        return "top"
    if x in [4, 5, 6]:
        return "middle"
    return "bottom"


def times_through_group(x: float) -> str:
    if pd.isna(x):
        return "missing"
    if x <= 1:
        return "1st"
    if x <= 2:
        return "2nd"
    if x <= 3:
        return "3rd"
    return "4th_plus"


def flag_cat(x: float, true_label: str, false_label: str) -> str:
    try:
        return true_label if int(float(x)) == 1 else false_label
    except Exception:
        return "missing"


def make_pair(a, b) -> str:
    return f"{str(a)}__{str(b)}"


def add_step3_binned_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Core context bins
    df["inning_group"] = to_num_series(df["inning"]).map(inning_group)
    df["top_bottom_cat"] = to_num_series(df["top_bottom"]).map(lambda x: "top" if int(x) == 0 else "bottom")
    df["score_diff_bin"] = to_num_series(df["score_diff_before_inning"]).map(score_diff_bin)
    df["batting_order_start_bin"] = to_num_series(df["batting_order_start"]).map(batting_order_group)
    df["is_top_order_start_cat"] = to_num_series(df["is_top_order_start"]).map(lambda x: flag_cat(x, "top_order", "not_top_order"))
    df["is_cleanup_start_cat"] = to_num_series(df["is_cleanup_start"]).map(lambda x: flag_cat(x, "cleanup", "not_cleanup"))
    df["pitcher_role_bin"] = to_num_series(df["pitcher_is_starter"]).map(lambda x: flag_cat(x, "starter", "reliever"))
    df["times_through_order_bin"] = to_num_series(df["pitcher_times_through_order"]).map(times_through_group)
    df["close_blowout_state"] = np.where(
        to_num_series(df["is_blowout"]).astype(int) == 1,
        "blowout",
        np.where(to_num_series(df["is_close_game"]).astype(int) == 1, "close", "normal"),
    )

    # Numeric skill/context bins
    rate_cols = [
        "team_run_scored_inning_rate_7d",
        "opponent_run_allowed_inning_rate_7d",
        "team_recent_2plus_inning_rate_7d",
        "opponent_recent_2plus_allowed_rate_7d",
        "next_3_batters_vs_pitcher_hand_xbh_rate",
    ]
    for col in rate_cols:
        df[f"{col}_bin"] = to_num_series(df[col]).map(fixed_rate_bin)
    for col in ["team_avg_runs_7d", "opponent_avg_runs_7d", "opponent_avg_runs_allowed_7d"]:
        df[f"{col}_bin"] = to_num_series(df[col]).map(fixed_runs_bin)
    for col in ["pitcher_era_before_inning", "opponent_starter_era"]:
        df[f"{col}_bin"] = to_num_series(df[col]).map(fixed_era_bin)
    for col in ["next_3_batters_vs_pitcher_hand_avg"]:
        df[f"{col}_bin"] = to_num_series(df[col]).map(fixed_avg_bin)
    for col in ["next_3_batters_vs_pitcher_hand_ops", "next_4_batters_vs_pitcher_hand_ops"]:
        df[f"{col}_bin"] = to_num_series(df[col]).map(fixed_ops_bin)

    # Cross features for FM interactions
    df["team_current_pitcher_pair"] = [make_pair(a, b) for a, b in zip(df["team_id"], df["current_pitcher_id"])]
    df["opponent_team_current_pitcher_pair"] = [make_pair(a, b) for a, b in zip(df["opponent_team_id"], df["current_pitcher_id"])]
    df["team_stadium_pair"] = [make_pair(a, b) for a, b in zip(df["team_id"], df["stadium_id"])]
    df["batting_order_group_pitcher_pair"] = [make_pair(a, b) for a, b in zip(df["batting_order_start_group"], df["current_pitcher_id"])]
    df["batting_order_group_lineup_ops_bin_pair"] = [make_pair(a, b) for a, b in zip(df["batting_order_start_group"], df["next_3_batters_vs_pitcher_hand_ops_bin"])]
    df["current_pitcher_lineup_ops_bin_pair"] = [make_pair(a, b) for a, b in zip(df["current_pitcher_id"], df["next_3_batters_vs_pitcher_hand_ops_bin"])]
    df["current_pitcher_lineup_xbh_bin_pair"] = [make_pair(a, b) for a, b in zip(df["current_pitcher_id"], df["next_3_batters_vs_pitcher_hand_xbh_rate_bin"])]
    df["score_state_lineup_ops_bin_pair"] = [make_pair(a, b) for a, b in zip(df["score_state_before_inning"], df["next_3_batters_vs_pitcher_hand_ops_bin"])]
    df["stadium_lineup_xbh_bin_pair"] = [make_pair(a, b) for a, b in zip(df["stadium_id"], df["next_3_batters_vs_pitcher_hand_xbh_rate_bin"])]
    df["team_lineup_ops_bin_pair"] = [make_pair(a, b) for a, b in zip(df["team_id"], df["next_3_batters_vs_pitcher_hand_ops_bin"])]
    df["pitcher_era_lineup_ops_bin_pair"] = [make_pair(a, b) for a, b in zip(df["pitcher_era_before_inning_bin"], df["next_3_batters_vs_pitcher_hand_ops_bin"])]
    df["inning_group_lineup_ops_bin_pair"] = [make_pair(a, b) for a, b in zip(df["inning_group"], df["next_3_batters_vs_pitcher_hand_ops_bin"])]

    # Ensure categorical strings
    new_cols = [c for c in df.columns if c.endswith("_bin") or c.endswith("_pair") or c in [
        "inning_group", "top_bottom_cat", "batting_order_start_bin", "is_top_order_start_cat", "is_cleanup_start_cat",
        "pitcher_role_bin", "times_through_order_bin", "close_blowout_state",
    ]]
    for col in new_cols:
        df[col] = df[col].fillna("missing").astype(str)
    return df

BINNED_CATEGORICAL_FEATURES = [
    # ID / context categories
    "team_id",
    "opponent_team_id",
    "home_away",
    "stadium_id",
    "starting_pitcher_id",
    "current_pitcher_id",
    "batting_order_start_group",
    "score_state_before_inning",
    # bins
    "inning_group",
    "top_bottom_cat",
    "score_diff_bin",
    "batting_order_start_bin",
    "is_top_order_start_cat",
    "is_cleanup_start_cat",
    "pitcher_role_bin",
    "times_through_order_bin",
    "close_blowout_state",
    "team_avg_runs_7d_bin",
    "opponent_avg_runs_7d_bin",
    "opponent_avg_runs_allowed_7d_bin",
    "pitcher_era_before_inning_bin",
    "opponent_starter_era_bin",
    "team_run_scored_inning_rate_7d_bin",
    "opponent_run_allowed_inning_rate_7d_bin",
    "team_recent_2plus_inning_rate_7d_bin",
    "opponent_recent_2plus_allowed_rate_7d_bin",
    "next_3_batters_vs_pitcher_hand_avg_bin",
    "next_3_batters_vs_pitcher_hand_ops_bin",
    "next_4_batters_vs_pitcher_hand_ops_bin",
    "next_3_batters_vs_pitcher_hand_xbh_rate_bin",
]

CROSS_CATEGORICAL_FEATURES = [
    "team_current_pitcher_pair",
    "opponent_team_current_pitcher_pair",
    "team_stadium_pair",
    "batting_order_group_pitcher_pair",
    "batting_order_group_lineup_ops_bin_pair",
    "current_pitcher_lineup_ops_bin_pair",
    "current_pitcher_lineup_xbh_bin_pair",
    "score_state_lineup_ops_bin_pair",
    "stadium_lineup_xbh_bin_pair",
    "team_lineup_ops_bin_pair",
    "pitcher_era_lineup_ops_bin_pair",
    "inning_group_lineup_ops_bin_pair",
]

# ============================================================
# Preprocess for FM
# ============================================================

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
    X_num_train = make_numeric_matrix(train_df, numeric_features, numeric_medians)
    scaler = StandardScaler()
    if numeric_features:
        scaler.fit(X_num_train)
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

# ============================================================
# FM model
# ============================================================

class TabularDataset(Dataset):
    def __init__(self, X_cat: np.ndarray, X_num: np.ndarray, y: np.ndarray):
        self.X_cat = X_cat.astype(np.int64)
        self.X_num = X_num.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self) -> int:
        return len(self.y)
    def __getitem__(self, idx: int):
        return self.X_cat[idx], self.X_num[idx], self.y[idx]


class BinaryFM(nn.Module):
    def __init__(self, cat_cardinalities: List[int], num_numeric: int, embed_dim: int = 16, dropout: float = 0.1):
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
        self.bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.cat_linear.weight, std=0.01)
        nn.init.normal_(self.cat_embed.weight, std=0.01)
        if self.num_linear is not None:
            nn.init.xavier_uniform_(self.num_linear.weight)
            nn.init.zeros_(self.num_linear.bias)
        if self.num_embed is not None:
            nn.init.normal_(self.num_embed, std=0.01)
    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size = x_cat.shape[0]
        out = self.bias.expand(batch_size)
        fields = []
        if self.num_cat > 0:
            x_cat_off = x_cat + self.offsets.unsqueeze(0)
            out = out + self.cat_linear(x_cat_off).squeeze(-1).sum(dim=1)
            fields.append(self.cat_embed(x_cat_off))
        if self.num_numeric > 0:
            assert self.num_linear is not None and self.num_embed is not None
            out = out + self.num_linear(x_num).squeeze(-1)
            fields.append(x_num.unsqueeze(-1) * self.num_embed.unsqueeze(0))
        if fields:
            V = torch.cat(fields, dim=1)
            V = self.dropout(V)
            interaction = 0.5 * ((V.sum(dim=1) ** 2) - (V ** 2).sum(dim=1)).sum(dim=1)
            out = out + interaction
        return out


def evaluate_fm_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    losses, ys, preds = [], [], []
    with torch.no_grad():
        for xb_cat, xb_num, yb in loader:
            xb_cat, xb_num, yb = xb_cat.to(device), xb_num.to(device), yb.float().to(device)
            logits = model(xb_cat, xb_num)
            loss = criterion(logits, yb)
            losses.append(loss.item() * len(yb))
            prob = torch.sigmoid(logits)
            pred = (prob >= 0.5).long()
            ys.append(yb.cpu().numpy().astype(int))
            preds.append(pred.cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(preds)
    return float(np.sum(losses) / max(len(y_true), 1)), float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def train_fm_model(
    X_cat_train: np.ndarray,
    X_num_train: np.ndarray,
    y_train: np.ndarray,
    X_cat_valid: np.ndarray,
    X_num_valid: np.ndarray,
    y_valid: np.ndarray,
    cat_cardinalities: List[int],
    pos_weight: float,
    max_epochs: int = 80,
    batch_size: int = 512,
    patience: int = 8,
    lr: float = 1e-3,
    embed_dim: int = 16,
) -> Tuple[nn.Module, Dict[str, float]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(TabularDataset(X_cat_train, X_num_train, y_train), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TabularDataset(X_cat_valid, X_num_valid, y_valid), batch_size=batch_size, shuffle=False)
    model = BinaryFM(cat_cardinalities, X_num_train.shape[1], embed_dim=embed_dim).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_valid_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0
        for xb_cat, xb_num, yb in train_loader:
            xb_cat, xb_num, yb = xb_cat.to(device), xb_num.to(device), yb.float().to(device)
            optimizer.zero_grad()
            logits = model(xb_cat, xb_num)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            total_n += len(yb)
        train_loss = total_loss / max(total_n, 1)
        valid_loss, valid_acc, valid_macro_f1 = evaluate_fm_loss(model, valid_loader, criterion, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} valid_acc={valid_acc:.5f} valid_macro_f1={valid_macro_f1:.5f}")
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
    return model, {"best_epoch": float(best_epoch), "best_valid_loss": float(best_valid_loss)}


def predict_fm(model: nn.Module, X_cat: np.ndarray, X_num: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    dummy_y = np.zeros(len(X_cat), dtype=np.int64)
    loader = DataLoader(TabularDataset(X_cat, X_num, dummy_y), batch_size=batch_size, shuffle=False)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb_cat, xb_num, _ in loader:
            xb_cat, xb_num = xb_cat.to(device), xb_num.to(device)
            logits = model(xb_cat, xb_num)
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs)

# ============================================================
# Save/report helpers
# ============================================================

def save_feature_list(output_dir: Path, variant: FeatureVariant) -> None:
    rows = []
    for col in variant.categorical_features:
        rows.append({"feature": col, "feature_type": "categorical"})
    for col in variant.numeric_features:
        rows.append({"feature": col, "feature_type": "numeric"})
    pd.DataFrame(rows).to_csv(output_dir / f"{variant.name}_features.csv", index=False, encoding="utf-8-sig")


def save_report_confusion_predictions(output_dir: Path, prefix: str, test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> None:
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=LABELS, zero_division=0, digits=6)
    (output_dir / f"{prefix}_classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(output_dir / f"{prefix}_confusion_matrix.csv", encoding="utf-8-sig")
    pred_df = pd.DataFrame({
        "game_date": test_df["game_date"].values,
        "game_id": test_df["game_id"].values if "game_id" in test_df.columns else np.arange(len(test_df)),
        "team_id": test_df["team_id"].values,
        "inning": test_df["inning"].values,
        "top_bottom": test_df["top_bottom"].values,
        "true_id": y_true,
        "pred_id": y_pred,
        "true_label": [ID_TO_LABEL[int(x)] for x in y_true],
        "pred_label": [ID_TO_LABEL[int(x)] for x in y_pred],
        "prob_RUN_SCORED": y_prob,
    })
    pred_df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")

# ============================================================
# Model evaluation
# ============================================================

def train_evaluate_lightgbm(split: SplitData, variant: FeatureVariant, output_dir: Path, seed: int) -> Dict[str, Any]:
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm が必要です。pip install lightgbm を実行してください。") from e
    print("\n" + "-" * 80)
    print(f"LightGBM evaluation: variant={variant.name}")
    print("-" * 80)
    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_feature_list(variant_dir, variant)
    feature_cols = variant.categorical_features + variant.numeric_features
    train_df, valid_df, test_df = split.train_df.copy(), split.valid_df.copy(), split.test_df.copy()
    for col in variant.categorical_features:
        cats = pd.Index(pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).fillna("<NA>").astype(str).unique())
        dtype = pd.CategoricalDtype(categories=cats)
        for part in [train_df, valid_df, test_df]:
            part[col] = part[col].fillna("<NA>").astype(str).astype(dtype)
    medians = train_df[variant.numeric_features].median(numeric_only=True).fillna(0.0)
    for part in [train_df, valid_df, test_df]:
        part[variant.numeric_features] = part[variant.numeric_features].fillna(medians)
    X_train, y_train = train_df[feature_cols], train_df["target"].values
    X_valid, y_valid = valid_df[feature_cols], valid_df["target"].values
    X_test, y_test = test_df[feature_cols], test_df["target"].values
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.03, num_leaves=31,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=seed, n_jobs=-1, class_weight="balanced",
    )
    callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=True), lgb.log_evaluation(period=100)]
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="binary_logloss", categorical_feature=variant.categorical_features, callbacks=callbacks)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_test, y_pred, y_prob)
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(variant_dir / f"{variant.name}_feature_importance.csv", index=False, encoding="utf-8-sig")
    save_report_confusion_predictions(variant_dir, variant.name, test_df, y_test, y_pred, y_prob)
    row: Dict[str, Any] = {
        "variant": variant.name, "model": "lightgbm",
        "n_categorical_features": len(variant.categorical_features),
        "n_numeric_features": len(variant.numeric_features),
        "n_total_features": len(feature_cols),
        "best_iteration": int(model.best_iteration_) if model.best_iteration_ is not None else None,
        **metrics,
    }
    print(f"[LightGBM Test Metrics: {variant.name}]")
    for k, v in row.items():
        if k not in ["variant", "model"]:
            print(f"{k}: {v}")
    return row


def train_evaluate_fm(split: SplitData, variant: FeatureVariant, output_dir: Path, args) -> Dict[str, Any]:
    print("\n" + "-" * 80)
    print(f"FM evaluation: variant={variant.name}")
    print("-" * 80)
    print(f"categorical_features ({len(variant.categorical_features)}): {variant.categorical_features}")
    print(f"numeric_features ({len(variant.numeric_features)}): {variant.numeric_features}")
    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_feature_list(variant_dir, variant)
    train_df, valid_df, test_df = split.train_df.copy(), split.valid_df.copy(), split.test_df.copy()
    artifacts = fit_preprocess(train_df, variant.categorical_features, variant.numeric_features)
    X_cat_train, X_num_train = transform_for_fm(train_df, artifacts)
    X_cat_valid, X_num_valid = transform_for_fm(valid_df, artifacts)
    X_cat_test, X_num_test = transform_for_fm(test_df, artifacts)
    y_train = train_df["target"].values.astype(np.int64)
    y_valid = valid_df["target"].values.astype(np.int64)
    y_test = test_df["target"].values.astype(np.int64)
    neg = max(int(np.sum(y_train == 0)), 1)
    pos = max(int(np.sum(y_train == 1)), 1)
    pos_weight = neg / pos
    print(f"[FM] pos_weight={pos_weight:.6f}  (negative={neg}, positive={pos})")
    model, info = train_fm_model(
        X_cat_train, X_num_train, y_train, X_cat_valid, X_num_valid, y_valid,
        artifacts.cat_cardinalities, pos_weight=pos_weight,
        max_epochs=args.fm_epochs, batch_size=args.batch_size,
        patience=args.patience, lr=args.lr, embed_dim=args.embed_dim,
    )
    y_prob = predict_fm(model, X_cat_test, X_num_test, batch_size=args.batch_size)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_test, y_pred, y_prob)
    save_report_confusion_predictions(variant_dir, variant.name, test_df, y_test, y_pred, y_prob)
    row: Dict[str, Any] = {
        "variant": variant.name, "model": "fm",
        "n_categorical_features": len(variant.categorical_features),
        "n_numeric_features": len(variant.numeric_features),
        "n_total_features": len(variant.categorical_features) + len(variant.numeric_features),
        **info, **metrics,
    }
    print(f"[FM Test Metrics: {variant.name}]")
    for k, v in row.items():
        if k not in ["variant", "model"]:
            print(f"{k}: {v}")
    return row


def get_variants() -> Tuple[FeatureVariant, List[FeatureVariant]]:
    lgbm_raw = FeatureVariant(
        name="lgbm_step2a_raw_32_features",
        model="lightgbm",
        categorical_features=STEP2A_CATEGORICAL_FEATURES,
        numeric_features=STEP2A_NUMERIC_FEATURES,
        description="LightGBM reference: Step2A raw 32 features.",
    )
    fm_variants = [
        FeatureVariant(
            name="fm_step2a_raw_32_features",
            model="fm",
            categorical_features=STEP2A_CATEGORICAL_FEATURES,
            numeric_features=STEP2A_NUMERIC_FEATURES,
            description="FM baseline: Step2A raw categorical + scaled numeric features.",
        ),
        FeatureVariant(
            name="fm_step2a_binned_features",
            model="fm",
            categorical_features=BINNED_CATEGORICAL_FEATURES,
            numeric_features=[],
            description="FM-friendly: Step2A numeric/context features converted to categorical bins.",
        ),
        FeatureVariant(
            name="fm_step2a_binned_cross_features",
            model="fm",
            categorical_features=BINNED_CATEGORICAL_FEATURES + CROSS_CATEGORICAL_FEATURES,
            numeric_features=[],
            description="FM-friendly: bins + explicit cross features.",
        ),
    ]
    return lgbm_raw, fm_variants

# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inning_csv_path", type=str, default="data/train_inning_runs_added_9_features.csv")
    parser.add_argument("--pa_csv_path", type=str, default="data/train_all_added_69_features.csv")
    parser.add_argument("--output_dir", type=str, default="output_step3_fm_binned_cross_step2a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fm_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_lightgbm", action="store_true", help="LightGBM reference evaluationをスキップする")
    parser.add_argument("--no_save_augmented_csv", action="store_true")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inning_csv_path = Path(args.inning_csv_path)
    pa_csv_path = Path(args.pa_csv_path)

    print("=" * 80)
    print("Step 3: FM bin + cross features based on Step2A lineup_vs_pitcher_hand_32_features")
    print("=" * 80)
    print(f"inning_csv_path: {inning_csv_path}")
    print(f"pa_csv_path: {pa_csv_path}")
    print(f"output_dir: {output_dir}")
    print("target: NO_RUN / RUN_SCORED")

    df = prepare_dataframe(inning_csv_path, pa_csv_path, output_dir, save_augmented_csv=not args.no_save_augmented_csv)
    df = add_step3_binned_cross_features(df)

    # Ensure all model features are valid dtypes
    all_cat_features = sorted(set(STEP2A_CATEGORICAL_FEATURES + BINNED_CATEGORICAL_FEATURES + CROSS_CATEGORICAL_FEATURES))
    all_num_features = sorted(set(STEP2A_NUMERIC_FEATURES))
    for col in all_cat_features:
        df[col] = df[col].fillna("<NA>").astype(str)
    for col in all_num_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if not args.no_save_augmented_csv:
        df.to_csv(output_dir / "train_inning_runs_step3_binned_cross_step2a.csv", index=False, encoding="utf-8-sig")

    split = split_by_game_date(df)
    print("\n[Data split]")
    for name, part in [("all", df), ("train", split.train_df), ("valid", split.valid_df), ("test", split.test_df)]:
        print(f"{name:5s}: {len(part):6d} rows, {part['game_date'].min().date()} ~ {part['game_date'].max().date()}, dates={part['game_date'].nunique()}")

    print("\n[Target distribution]")
    target_dist = df["target_label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print(target_dist)
    target_dist.to_csv(output_dir / "target_distribution.csv", encoding="utf-8-sig")

    # Coverage for Step2A and Step3 categorical features
    coverage_rows = []
    for col in LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES:
        s = pd.to_numeric(df[col], errors="coerce")
        coverage_rows.append({"feature": col, "missing_count": int(s.isna().sum()), "missing_rate": float(s.isna().mean()), "mean": float(s.mean())})
    pd.DataFrame(coverage_rows).to_csv(output_dir / "step3_step2a_lineup_feature_coverage.csv", index=False, encoding="utf-8-sig")

    lgbm_raw, fm_variants = get_variants()
    variants = ([lgbm_raw] if not args.skip_lightgbm else []) + fm_variants
    print("\n[Feature variants]")
    summary_rows = []
    for v in variants:
        row = {
            "variant": v.name,
            "model": v.model,
            "description": v.description,
            "n_categorical_features": len(v.categorical_features),
            "n_numeric_features": len(v.numeric_features),
            "n_total_features": len(v.categorical_features) + len(v.numeric_features),
        }
        summary_rows.append(row)
        print(f"{v.name}: model={v.model}, categorical={row['n_categorical_features']}, numeric={row['n_numeric_features']}, total={row['n_total_features']}")
    pd.DataFrame(summary_rows).to_csv(output_dir / "step3_feature_variant_summary.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "feature_config_step3.json", "w", encoding="utf-8") as f:
        json.dump({
            "step2a_categorical_features": STEP2A_CATEGORICAL_FEATURES,
            "step2a_numeric_features": STEP2A_NUMERIC_FEATURES,
            "binned_categorical_features": BINNED_CATEGORICAL_FEATURES,
            "cross_categorical_features": CROSS_CATEGORICAL_FEATURES,
            "target": "NO_RUN / RUN_SCORED",
            "lineup_join_key": ["game_date", "stadium_id", "top_bottom", "bat_order"],
        }, f, ensure_ascii=False, indent=2)

    if args.dry_run:
        print("\nDry run finished. No model training executed.")
        return

    all_rows = []
    if not args.skip_lightgbm:
        all_rows.append(train_evaluate_lightgbm(split, lgbm_raw, output_dir, args.seed))
    for variant in fm_variants:
        all_rows.append(train_evaluate_fm(split, variant, output_dir, args))

    comp = pd.DataFrame(all_rows)
    preferred_cols = ["variant", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "no_run_precision", "no_run_recall", "no_run_f1", "run_scored_precision", "run_scored_recall", "run_scored_f1", "run_scored_pred_count", "log_loss", "roc_auc", "pr_auc", "n_categorical_features", "n_numeric_features", "n_total_features", "best_epoch", "best_valid_loss", "best_iteration"]
    cols = [c for c in preferred_cols if c in comp.columns] + [c for c in comp.columns if c not in preferred_cols]
    comp = comp[cols]
    comp.to_csv(output_dir / "step3_fm_binned_cross_step2a_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Step 3 comparison")
    print("=" * 80)
    print(comp.to_string(index=False))
    print(f"\nSaved: {output_dir / 'step3_fm_binned_cross_step2a_comparison_metrics.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
