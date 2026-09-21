# -*- coding: utf-8 -*-
"""
Step 2C: LightGBM inning-level RUN_SCORED prediction with enhanced lineup-structure features.

目的:
- イニング得点有無予測（NO_RUN / RUN_SCORED）に対して、
  batting_order_start を「打者能力ベースの特徴量」に拡張する。
- PA単位CSVから、各試合・表裏・打順スロットの打者能力を抽出し、
  イニング開始打順から次の3人/4人の打者能力平均を作成する。
- Step1 の score_context_28_features と、Step2 の lineup_strength_features を同一分割で比較する。

入力:
1) inning CSV:
   train_inning_runs_added_9_features.csv
2) PA CSV:
   train_all_added_69_features.csv

実行例:
python evaluate_step2c_lightgbm_enhanced_lineup_structure.py --inning_csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" --pa_csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv"
python evaluate_step2c_lightgbm_enhanced_lineup_structure.py \
  --inning_csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" \
  --pa_csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv"

軽い確認:
python evaluate_step2c_lightgbm_enhanced_lineup_structure.py \
  --inning_csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" \
  --pa_csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv" \
  --dry_run

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm
"""
from __future__ import annotations

import argparse
import json
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
    roc_auc_score,
    auc,
)

LABELS = ["NO_RUN", "RUN_SCORED"]
LABEL_TO_ID = {"NO_RUN": 0, "RUN_SCORED": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

# ============================================================
# Feature definitions
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

SCORE_CONTEXT_CATEGORICAL_FEATURES = [
    "score_state_before_inning",
]

SCORE_CONTEXT_NUMERIC_FEATURES = [
    "score_diff_before_inning",
    "is_close_game",
    "is_blowout",
]

# Step2 original features: overall batter quality from the next 3/4 batting-order slots
LINEUP_STRENGTH_NUMERIC_FEATURES = [
    "next_3_batters_obp",
    "next_3_batters_ops",
    "next_3_batters_xbh_rate",
    "next_4_batters_ops",
]

# Step2A features: choose batter_vs_rhp_* or batter_vs_lhp_* according to current pitcher hand
LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES = [
    "next_3_batters_vs_pitcher_hand_avg",
    "next_3_batters_vs_pitcher_hand_ops",
    "next_4_batters_vs_pitcher_hand_ops",
    "next_3_batters_vs_pitcher_hand_xbh_rate",
]

# Step2C features: Step2A + leadoff / chain / role / variance-max lineup structure
STEP2C_ADDED_NUMERIC_FEATURES = [
    # 1. 先頭打者特徴量
    "leadoff_batter_vs_pitcher_hand_avg",
    "leadoff_batter_vs_pitcher_hand_ops",
    "leadoff_batter_obp",
    "leadoff_batter_bb_rate",
    "leadoff_batter_k_rate",
    # 2. 連鎖特徴量
    "chain_leadoff_obp_batter2_xbh",
    "chain_leadoff_obp_batter2_obp_batter3_xbh",
    "chain_leadoff_bb_batter2_xbh",
    # 3. 役割特徴量
    "slot1_onbase_score",
    "slot2_contact_score",
    "slot3_power_score",
    # 4. 分散・最大値特徴量
    "ops_std_next3",
    "xbh_std_next3",
    "bb_rate_std_next3",
    "k_rate_std_next3",
    "ops_max_next3",
    "xbh_max_next3",
    "hr_rate_max_next3",
]

# Engineered support features created for diagnostics / formulas, not used directly in Step2C model by default.
LINEUP_SUPPORT_NUMERIC_FEATURES = [
    "next_3_batters_obp",
    "next_3_batters_ops",
    "next_3_batters_bb_rate",
    "next_3_batters_k_rate",
    "next_3_batters_hr_rate",
    "next_3_batters_xbh_rate",
    "next_4_batters_ops",
]

# Optional diagnostics; not used as model features by default
LINEUP_DIAGNOSTIC_FEATURES = [
    "next_3_batters_known_count",
    "next_4_batters_known_count",
    "next_3_batters_vs_hand_known_count",
    "next_4_batters_vs_hand_known_count",
]

PA_LINEUP_VALUE_COLUMNS = [
    "batter_obp",
    "batter_ops",
    "batter_k_rate",
    "batter_bb_rate",
    "batter_hr_rate",
    "batter_xbh_rate",
    "batter_vs_rhp_avg",
    "batter_vs_rhp_ops",
    "batter_vs_lhp_avg",
    "batter_vs_lhp_ops",
]


@dataclass
class FeatureVariant:
    name: str
    description: str
    categorical_features: List[str]
    numeric_features: List[str]


@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


# ============================================================
# Utilities
# ============================================================

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob_pos: Optional[np.ndarray] = None) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

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
        precision, recall, _ = precision_recall_curve(y_true, y_prob_pos)
        result["pr_auc"] = float(auc(recall, precision))

    return result


# ============================================================
# Step2 feature engineering
# ============================================================

def validate_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} に必要な列がありません: {missing}")


def build_lineup_lookup(pa_df: pd.DataFrame) -> Tuple[Dict[Tuple[pd.Timestamp, str, int, int], Dict[str, float]], pd.DataFrame]:
    """
    PA単位CSVから、各試合・チーム・打順スロットの代表打者能力を作る。

    key:
        (game_date, stadium_id, top_bottom, bat_order)

    値:
        batter_obp, batter_ops, batter_xbh_rate

    注意:
    - PA CSVにgame_idが無いため、game_date + stadium_id + top_bottom + bat_orderで結合する。
      NPBの通常日程では同一日・同一球場で同時に複数試合が行われない前提。
    - batter_team_id はPA CSV側で不安定なケースがあるため、結合キーには使わない。
    - 各打順スロットは、その試合で最初に現れたPA行を採用する。
      同一試合の後続結果を集計して作る特徴量ではない。
    """
    pa = pa_df.copy()
    pa["game_date"] = normalize_date(pa["game_date"])
    pa["top_bottom"] = pa["top_bottom"].apply(normalize_top_bottom).astype("Int64")
    pa["bat_order"] = pd.to_numeric(pa["bat_order"], errors="coerce").astype("Int64")

    for col in PA_LINEUP_VALUE_COLUMNS:
        pa[col] = pd.to_numeric(pa[col], errors="coerce")

    sort_cols = ["game_date", "stadium_id", "top_bottom", "bat_order", "inning", "out_count"]
    for c in ["inning", "out_count"]:
        if c in pa.columns:
            pa[c] = pd.to_numeric(pa[c], errors="coerce")
        else:
            pa[c] = 0

    lineup_cols = ["game_date", "stadium_id", "top_bottom", "bat_order"] + PA_LINEUP_VALUE_COLUMNS
    lineup = (
        pa.sort_values(sort_cols)
        .dropna(subset=["game_date", "stadium_id", "top_bottom", "bat_order"])
        .groupby(["game_date", "stadium_id", "top_bottom", "bat_order"], as_index=False)
        .first()[lineup_cols]
    )

    lookup: Dict[Tuple[pd.Timestamp, str, int, int], Dict[str, float]] = {}
    for row in lineup.itertuples(index=False):
        key = (
            row.game_date,
            str(row.stadium_id),
            int(row.top_bottom),
            int(row.bat_order),
        )
        lookup[key] = {
            "batter_obp": float(row.batter_obp) if pd.notna(row.batter_obp) else np.nan,
            "batter_ops": float(row.batter_ops) if pd.notna(row.batter_ops) else np.nan,
            "batter_k_rate": float(row.batter_k_rate) if pd.notna(row.batter_k_rate) else np.nan,
            "batter_bb_rate": float(row.batter_bb_rate) if pd.notna(row.batter_bb_rate) else np.nan,
            "batter_hr_rate": float(row.batter_hr_rate) if pd.notna(row.batter_hr_rate) else np.nan,
            "batter_xbh_rate": float(row.batter_xbh_rate) if pd.notna(row.batter_xbh_rate) else np.nan,
            "batter_vs_rhp_avg": float(row.batter_vs_rhp_avg) if pd.notna(row.batter_vs_rhp_avg) else np.nan,
            "batter_vs_rhp_ops": float(row.batter_vs_rhp_ops) if pd.notna(row.batter_vs_rhp_ops) else np.nan,
            "batter_vs_lhp_avg": float(row.batter_vs_lhp_avg) if pd.notna(row.batter_vs_lhp_avg) else np.nan,
            "batter_vs_lhp_ops": float(row.batter_vs_lhp_ops) if pd.notna(row.batter_vs_lhp_ops) else np.nan,
        }

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
    """PA CSVから pitcher_id -> pitcher_hand の代表値を作る。"""
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


def add_lineup_strength_features(inning_df: pd.DataFrame, pa_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """通常の打線強度・投手左右対応・Step2C得点期待proxy特徴量を追加する。"""
    validate_required_columns(
        pa_df,
        ["game_date", "stadium_id", "top_bottom", "bat_order", "pitcher_id", "pitcher_hand"] + PA_LINEUP_VALUE_COLUMNS,
        "PA CSV",
    )
    validate_required_columns(
        inning_df,
        ["game_date", "stadium_id", "team_id", "top_bottom", "batting_order_start", "current_pitcher_id"],
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

    def nanmean_or_nan(values: List[float]) -> float:
        arr = np.array(values, dtype=float)
        if np.all(np.isnan(arr)):
            return np.nan
        return float(np.nanmean(arr))

    def safe_std(values: List[float]) -> float:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan
        return float(np.std(arr, ddof=0))

    def safe_max(values: List[float]) -> float:
        arr = np.array(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            return np.nan
        return float(np.max(arr))

    def safe_mul(*values: float) -> float:
        arr = np.array(values, dtype=float)
        if np.any(np.isnan(arr)):
            return np.nan
        return float(np.prod(arr))

    def safe_add(*values: float) -> float:
        arr = np.array(values, dtype=float)
        if np.any(np.isnan(arr)):
            return np.nan
        return float(np.sum(arr))

    def choose_vs_hand_values(val: Dict[str, float], pitcher_hand: str) -> Tuple[float, float]:
        if pitcher_hand == "R":
            return val.get("batter_vs_rhp_avg", np.nan), val.get("batter_vs_rhp_ops", np.nan)
        if pitcher_hand == "L":
            return val.get("batter_vs_lhp_avg", np.nan), val.get("batter_vs_lhp_ops", np.nan)
        return np.nan, val.get("batter_ops", np.nan)

    rows = []
    for row in df.itertuples(index=False):
        game_date = row.game_date
        stadium_id = str(row.stadium_id)
        tb = int(row.top_bottom)
        start_order = int(row.batting_order_start)
        pitcher_hand = str(row.current_pitcher_hand)

        vals3_obp: List[float] = []
        vals3_ops: List[float] = []
        vals3_k: List[float] = []
        vals3_bb: List[float] = []
        vals3_hr: List[float] = []
        vals3_xbh: List[float] = []
        vals4_ops: List[float] = []

        vals3_vs_avg: List[float] = []
        vals3_vs_ops: List[float] = []
        vals4_vs_ops: List[float] = []
        vals3_vs_xbh: List[float] = []

        leadoff_vals = {
            "leadoff_batter_vs_pitcher_hand_avg": np.nan,
            "leadoff_batter_vs_pitcher_hand_ops": np.nan,
            "leadoff_batter_obp": np.nan,
            "leadoff_batter_bb_rate": np.nan,
            "leadoff_batter_k_rate": np.nan,
        }
        # slot_metrics[0] = イニング先頭打者, slot_metrics[1] = 2人目, slot_metrics[2] = 3人目
        slot_metrics = [
            {"obp": np.nan, "ops": np.nan, "k_rate": np.nan, "bb_rate": np.nan, "hr_rate": np.nan, "xbh_rate": np.nan},
            {"obp": np.nan, "ops": np.nan, "k_rate": np.nan, "bb_rate": np.nan, "hr_rate": np.nan, "xbh_rate": np.nan},
            {"obp": np.nan, "ops": np.nan, "k_rate": np.nan, "bb_rate": np.nan, "hr_rate": np.nan, "xbh_rate": np.nan},
        ]

        first3 = set(next_orders(start_order, 3))
        for idx, order in enumerate(next_orders(start_order, 4)):
            key = (game_date, stadium_id, tb, order)
            val = lookup.get(key)
            if val is None:
                if order in first3:
                    vals3_obp.append(np.nan)
                    vals3_ops.append(np.nan)
                    vals3_k.append(np.nan)
                    vals3_bb.append(np.nan)
                    vals3_hr.append(np.nan)
                    vals3_xbh.append(np.nan)
                    vals3_vs_avg.append(np.nan)
                    vals3_vs_ops.append(np.nan)
                    vals3_vs_xbh.append(np.nan)
                vals4_ops.append(np.nan)
                vals4_vs_ops.append(np.nan)
                continue

            hand_avg, hand_ops = choose_vs_hand_values(val, pitcher_hand)

            if idx == 0:
                leadoff_vals = {
                    "leadoff_batter_vs_pitcher_hand_avg": hand_avg,
                    "leadoff_batter_vs_pitcher_hand_ops": hand_ops,
                    "leadoff_batter_obp": val.get("batter_obp", np.nan),
                    "leadoff_batter_bb_rate": val.get("batter_bb_rate", np.nan),
                    "leadoff_batter_k_rate": val.get("batter_k_rate", np.nan),
                }

            if idx < 3:
                slot_metrics[idx] = {
                    "obp": val.get("batter_obp", np.nan),
                    "ops": val.get("batter_ops", np.nan),
                    "k_rate": val.get("batter_k_rate", np.nan),
                    "bb_rate": val.get("batter_bb_rate", np.nan),
                    "hr_rate": val.get("batter_hr_rate", np.nan),
                    "xbh_rate": val.get("batter_xbh_rate", np.nan),
                }

            if order in first3:
                vals3_obp.append(val.get("batter_obp", np.nan))
                vals3_ops.append(val.get("batter_ops", np.nan))
                vals3_k.append(val.get("batter_k_rate", np.nan))
                vals3_bb.append(val.get("batter_bb_rate", np.nan))
                vals3_hr.append(val.get("batter_hr_rate", np.nan))
                vals3_xbh.append(val.get("batter_xbh_rate", np.nan))
                vals3_vs_avg.append(hand_avg)
                vals3_vs_ops.append(hand_ops)
                # xbh_rate is not split by pitcher hand in the current PA CSV, so use overall xbh_rate as a proxy.
                vals3_vs_xbh.append(val.get("batter_xbh_rate", np.nan))

            vals4_ops.append(val.get("batter_ops", np.nan))
            vals4_vs_ops.append(hand_ops)

        # Step2C: order-aware chain / role / dispersion features
        s1, s2, s3 = slot_metrics
        chain_leadoff_obp_batter2_xbh = safe_mul(s1["obp"], s2["xbh_rate"])
        chain_leadoff_obp_batter2_obp_batter3_xbh = safe_mul(s1["obp"], s2["obp"], s3["xbh_rate"])
        chain_leadoff_bb_batter2_xbh = safe_mul(s1["bb_rate"], s2["xbh_rate"])

        # 役割特徴量は、イニング開始から何人目かの役割として定義する。
        # slot1: 出塁役、slot2: つなぎ役、slot3: 返す役
        slot1_onbase_score = safe_add(s1["obp"], s1["bb_rate"], -s1["k_rate"] if pd.notna(s1["k_rate"]) else np.nan)
        slot2_contact_score = safe_add(s2["obp"], 1.0 - s2["k_rate"] if pd.notna(s2["k_rate"]) else np.nan)
        slot3_power_score = safe_add(s3["ops"], s3["xbh_rate"], s3["hr_rate"])

        out = {
            # Step2 original / support
            "next_3_batters_obp": nanmean_or_nan(vals3_obp),
            "next_3_batters_ops": nanmean_or_nan(vals3_ops),
            "next_3_batters_xbh_rate": nanmean_or_nan(vals3_xbh),
            "next_4_batters_ops": nanmean_or_nan(vals4_ops),
            "next_3_batters_bb_rate": nanmean_or_nan(vals3_bb),
            "next_3_batters_k_rate": nanmean_or_nan(vals3_k),
            "next_3_batters_hr_rate": nanmean_or_nan(vals3_hr),
            # Step2A
            "next_3_batters_vs_pitcher_hand_avg": nanmean_or_nan(vals3_vs_avg),
            "next_3_batters_vs_pitcher_hand_ops": nanmean_or_nan(vals3_vs_ops),
            "next_4_batters_vs_pitcher_hand_ops": nanmean_or_nan(vals4_vs_ops),
            "next_3_batters_vs_pitcher_hand_xbh_rate": nanmean_or_nan(vals3_vs_xbh),
            # Step2C: 先頭打者特徴量
            **leadoff_vals,
            # Step2C: 連鎖特徴量
            "chain_leadoff_obp_batter2_xbh": chain_leadoff_obp_batter2_xbh,
            "chain_leadoff_obp_batter2_obp_batter3_xbh": chain_leadoff_obp_batter2_obp_batter3_xbh,
            "chain_leadoff_bb_batter2_xbh": chain_leadoff_bb_batter2_xbh,
            # Step2C: 役割特徴量
            "slot1_onbase_score": slot1_onbase_score,
            "slot2_contact_score": slot2_contact_score,
            "slot3_power_score": slot3_power_score,
            # Step2C: 分散・最大値特徴量
            "ops_std_next3": safe_std(vals3_ops),
            "xbh_std_next3": safe_std(vals3_xbh),
            "bb_rate_std_next3": safe_std(vals3_bb),
            "k_rate_std_next3": safe_std(vals3_k),
            "ops_max_next3": safe_max(vals3_ops),
            "xbh_max_next3": safe_max(vals3_xbh),
            "hr_rate_max_next3": safe_max(vals3_hr),
            # diagnostics
            "next_3_batters_known_count": int(np.sum(~np.isnan(np.array(vals3_ops, dtype=float)))),
            "next_4_batters_known_count": int(np.sum(~np.isnan(np.array(vals4_ops, dtype=float)))),
            "next_3_batters_vs_hand_known_count": int(np.sum(~np.isnan(np.array(vals3_vs_ops, dtype=float)))),
            "next_4_batters_vs_hand_known_count": int(np.sum(~np.isnan(np.array(vals4_vs_ops, dtype=float)))),
        }
        rows.append(out)

    feat_df = pd.DataFrame(rows)
    all_added = (
        LINEUP_SUPPORT_NUMERIC_FEATURES
        + LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES
        + STEP2C_ADDED_NUMERIC_FEATURES
        + LINEUP_DIAGNOSTIC_FEATURES
    )
    for col in dict.fromkeys(all_added):
        df[col] = feat_df[col].values

    return df, lineup_table

def prepare_inning_dataframe(inning_csv_path: Path, pa_csv_path: Path, output_dir: Path, save_augmented_csv: bool = True) -> pd.DataFrame:
    inning_df = pd.read_csv(inning_csv_path)
    pa_df = pd.read_csv(pa_csv_path)

    required_inning = sorted(
        set(
            BASE_CATEGORICAL_FEATURES
            + BASE_NUMERIC_FEATURES
            + ["game_date", "target_run_scored", "score_diff_before_inning", "team_id", "stadium_id", "top_bottom", "batting_order_start"]
        )
    )
    validate_required_columns(inning_df, required_inning, "inning CSV")

    df = inning_df.copy()
    df["game_date"] = normalize_date(df["game_date"])
    df["target"] = df["target_run_scored"].map(LABEL_TO_ID).astype(int)
    df["target_label"] = df["target"].map(ID_TO_LABEL)
    df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)

    df = create_score_context_features(df)
    df, lineup_table = add_lineup_strength_features(df, pa_df)

    # type conversion
    categorical_features = BASE_CATEGORICAL_FEATURES + SCORE_CONTEXT_CATEGORICAL_FEATURES
    numeric_features = BASE_NUMERIC_FEATURES + SCORE_CONTEXT_NUMERIC_FEATURES + LINEUP_SUPPORT_NUMERIC_FEATURES + LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES + STEP2C_ADDED_NUMERIC_FEATURES + LINEUP_DIAGNOSTIC_FEATURES

    for col in categorical_features:
        df[col] = df[col].fillna("<NA>").astype(str)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["game_date", "game_id", "team_id", "inning", "top_bottom"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    lineup_table.to_csv(output_dir / "step2c_lineup_slot_table_from_pa.csv", index=False, encoding="utf-8-sig")

    if save_augmented_csv:
        df.to_csv(output_dir / "train_inning_runs_step2c_enhanced_lineup_structure.csv", index=False, encoding="utf-8-sig")

    return df


# ============================================================
# LightGBM evaluation
# ============================================================

def unique_features(features: List[str]) -> List[str]:
    return list(dict.fromkeys(features))


def get_feature_variants() -> List[FeatureVariant]:
    score_cat = BASE_CATEGORICAL_FEATURES + SCORE_CONTEXT_CATEGORICAL_FEATURES
    score_num = BASE_NUMERIC_FEATURES + SCORE_CONTEXT_NUMERIC_FEATURES
    step2a_num = score_num + LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES
    step2c_num = unique_features(score_num + LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES + STEP2C_ADDED_NUMERIC_FEATURES)
    return [
        FeatureVariant(
            name="score_context_28_features",
            description="Step1: score contextあり。",
            categorical_features=score_cat,
            numeric_features=score_num,
        ),
        FeatureVariant(
            name="lineup_vs_pitcher_hand_32_features",
            description="Step2A: score context + 相手投手左右に応じた次の3/4打者能力特徴量。",
            categorical_features=score_cat,
            numeric_features=step2a_num,
        ),
        FeatureVariant(
            name="step2c_enhanced_lineup_50_features",
            description="Step2C: Step2A + 先頭打者 + 連鎖 + 役割 + 分散/最大値特徴量。",
            categorical_features=score_cat,
            numeric_features=step2c_num,
        ),
    ]

def save_feature_list(output_dir: Path, variant: FeatureVariant) -> None:
    rows = []
    for col in variant.categorical_features:
        rows.append({"feature": col, "feature_type": "categorical"})
    for col in variant.numeric_features:
        rows.append({"feature": col, "feature_type": "numeric"})
    pd.DataFrame(rows).to_csv(output_dir / f"{variant.name}_features.csv", index=False, encoding="utf-8-sig")


def train_evaluate_lightgbm_variant(
    split: SplitData,
    variant: FeatureVariant,
    output_dir: Path,
    seed: int,
) -> Dict[str, float]:
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm が必要です。pip install lightgbm を実行してください。") from e

    print("\n" + "-" * 80)
    print(f"LightGBM evaluation: variant={variant.name}")
    print("-" * 80)
    print(f"categorical_features ({len(variant.categorical_features)}): {variant.categorical_features}")
    print(f"numeric_features ({len(variant.numeric_features)}): {variant.numeric_features}")

    variant_dir = output_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_feature_list(variant_dir, variant)

    feature_cols = variant.categorical_features + variant.numeric_features
    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()

    # Align categorical dtypes across splits
    for col in variant.categorical_features:
        cats = pd.Index(pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).fillna("<NA>").astype(str).unique())
        dtype = pd.CategoricalDtype(categories=cats)
        train_df[col] = train_df[col].fillna("<NA>").astype(str).astype(dtype)
        valid_df[col] = valid_df[col].fillna("<NA>").astype(str).astype(dtype)
        test_df[col] = test_df[col].fillna("<NA>").astype(str).astype(dtype)

    # Fill numeric by train median only
    medians = train_df[variant.numeric_features].median(numeric_only=True)
    medians = medians.fillna(0.0)
    train_df[variant.numeric_features] = train_df[variant.numeric_features].fillna(medians)
    valid_df[variant.numeric_features] = valid_df[variant.numeric_features].fillna(medians)
    test_df[variant.numeric_features] = test_df[variant.numeric_features].fillna(medians)

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
        categorical_feature=variant.categorical_features,
        callbacks=callbacks,
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics.update(
        {
            "variant": variant.name,
            "model": "lightgbm",
            "n_categorical_features": len(variant.categorical_features),
            "n_numeric_features": len(variant.numeric_features),
            "n_total_features": len(feature_cols),
            "best_iteration": int(model.best_iteration_) if model.best_iteration_ is not None else None,
        }
    )

    print(f"\n[LightGBM Test Metrics: {variant.name}]")
    for k, v in metrics.items():
        if k not in ["variant", "model"]:
            print(f"{k}: {v}")

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=LABELS,
        zero_division=0,
        digits=6,
    )
    (variant_dir / f"{variant.name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        variant_dir / f"{variant.name}_confusion_matrix.csv", encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame(
        {
            "true_id": y_test,
            "pred_id": y_pred,
            "true_label": [ID_TO_LABEL[int(x)] for x in y_test],
            "pred_label": [ID_TO_LABEL[int(x)] for x in y_pred],
            "prob_RUN_SCORED": y_prob,
        }
    )
    pred_df.to_csv(variant_dir / f"{variant.name}_predictions.csv", index=False, encoding="utf-8-sig")

    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    fi.to_csv(variant_dir / f"{variant.name}_feature_importance.csv", index=False, encoding="utf-8-sig")

    return metrics


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inning_csv_path",
        type=str,
        default="data/train_inning_runs_added_9_features.csv",
        help="イニング単位CSVのパス",
    )
    parser.add_argument(
        "--pa_csv_path",
        type=str,
        default="data/train_all_added_69_features.csv",
        help="PA単位69特徴量CSVのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_step2c_lightgbm_enhanced_lineup_structure",
        help="出力先ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="特徴量作成と分割確認のみ行う")
    parser.add_argument("--no_save_augmented_csv", action="store_true", help="拡張CSVを保存しない")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    inning_csv_path = Path(args.inning_csv_path)
    pa_csv_path = Path(args.pa_csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 2C: LightGBM enhanced lineup-structure feature comparison")
    print("=" * 80)
    print(f"inning_csv_path: {inning_csv_path}")
    print(f"pa_csv_path: {pa_csv_path}")
    print(f"output_dir: {output_dir}")
    print("target: NO_RUN / RUN_SCORED")

    df = prepare_inning_dataframe(
        inning_csv_path,
        pa_csv_path,
        output_dir,
        save_augmented_csv=not args.no_save_augmented_csv,
    )
    split = split_by_game_date(df)

    print("\n[Data split]")
    print(f"all  : {len(df):>6} rows, {df['game_date'].min().date()} ~ {df['game_date'].max().date()}, dates={df['game_date'].nunique()}")
    print(f"train: {len(split.train_df):>6} rows, {split.train_df['game_date'].min().date()} ~ {split.train_df['game_date'].max().date()}, dates={split.train_df['game_date'].nunique()}")
    print(f"valid: {len(split.valid_df):>6} rows, {split.valid_df['game_date'].min().date()} ~ {split.valid_df['game_date'].max().date()}, dates={split.valid_df['game_date'].nunique()}")
    print(f"test : {len(split.test_df):>6} rows, {split.test_df['game_date'].min().date()} ~ {split.test_df['game_date'].max().date()}, dates={split.test_df['game_date'].nunique()}")

    print("\n[Target distribution]")
    target_dist = df["target_label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print(target_dist)
    target_dist.to_csv(output_dir / "target_distribution.csv", encoding="utf-8-sig")

    print("\n[Lineup-strength feature coverage]")
    coverage_rows = []
    for col in unique_features(LINEUP_SUPPORT_NUMERIC_FEATURES + LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES + STEP2C_ADDED_NUMERIC_FEATURES + LINEUP_DIAGNOSTIC_FEATURES):
        coverage_rows.append(
            {
                "feature": col,
                "missing_count": int(df[col].isna().sum()),
                "missing_rate": float(df[col].isna().mean()),
                "mean": float(pd.to_numeric(df[col], errors="coerce").mean()) if pd.to_numeric(df[col], errors="coerce").notna().any() else np.nan,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    print(coverage_df.to_string(index=False))
    coverage_df.to_csv(output_dir / "step2c_lineup_feature_coverage.csv", index=False, encoding="utf-8-sig")

    variants = get_feature_variants()
    print("\n[Feature variants]")
    summary_rows = []
    for v in variants:
        row = {
            "variant": v.name,
            "description": v.description,
            "n_categorical_features": len(v.categorical_features),
            "n_numeric_features": len(v.numeric_features),
            "n_total_features": len(v.categorical_features) + len(v.numeric_features),
        }
        summary_rows.append(row)
        print(f"{v.name}: categorical={row['n_categorical_features']}, numeric={row['n_numeric_features']}, total={row['n_total_features']}")
    pd.DataFrame(summary_rows).to_csv(output_dir / "step2c_feature_variant_summary.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_categorical_features": BASE_CATEGORICAL_FEATURES,
                "base_numeric_features": BASE_NUMERIC_FEATURES,
                "score_context_categorical_features": SCORE_CONTEXT_CATEGORICAL_FEATURES,
                "score_context_numeric_features": SCORE_CONTEXT_NUMERIC_FEATURES,
                "lineup_support_numeric_features": LINEUP_SUPPORT_NUMERIC_FEATURES,
                "lineup_vs_pitcher_hand_numeric_features": LINEUP_VS_PITCHER_HAND_NUMERIC_FEATURES,
                "step2c_added_numeric_features": STEP2C_ADDED_NUMERIC_FEATURES,
                "lineup_diagnostic_features": LINEUP_DIAGNOSTIC_FEATURES,
                "target": "NO_RUN / RUN_SCORED",
                "lineup_join_key": ["game_date", "stadium_id", "top_bottom", "bat_order"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.dry_run:
        print("\nDry run finished. No model training executed.")
        return

    all_metrics = []
    for variant in variants:
        metrics = train_evaluate_lightgbm_variant(split, variant, output_dir, args.seed)
        all_metrics.append(metrics)

    comparison_df = pd.DataFrame(all_metrics)
    cols = ["variant", "model"] + [c for c in comparison_df.columns if c not in ["variant", "model"]]
    comparison_df = comparison_df[cols]
    comparison_df.to_csv(output_dir / "step2c_lightgbm_enhanced_lineup_structure_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Step 2C comparison")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {output_dir / 'step2c_lightgbm_enhanced_lineup_structure_comparison_metrics.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
