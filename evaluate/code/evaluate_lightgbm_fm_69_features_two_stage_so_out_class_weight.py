# -*- coding: utf-8 -*-
"""
train_all_added_69_features.csv を用いて、2段階分類で LightGBM と Factorization Machine(FM) を評価するスクリプト。

目的:
- 第1段階: 出塁するか / アウトか を2分類する
    ON_BASE = SINGLE + DOUBLE + TRIPLE + HR + BB + HBP
    OUT     = SO + OTHER_OUT
- 第2段階:
    第1段階でON_BASEと予測された場合: SINGLE / XBH / BB_HBP を分類する
    第1段階でOUTと予測された場合    : SO / OTHER_OUT を分類する
- 最終的に SINGLE / XBH / BB_HBP / SO / OTHER_OUT の5分類として評価する
- class_weight を導入し、クラス不均衡への対応を行う

実行例:
    python evaluate_lightgbm_fm_69_features_two_stage_so_out_class_weight.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_69_features.csv"

必要ライブラリ:
    pip install pandas numpy scikit-learn lightgbm torch tqdm

出力:
- output_lgbm_fm_69_features_two_stage_so_out_class_weight/model_comparison_metrics.csv
- LightGBM / FM の最終5分類 classification_report / confusion_matrix / predictions
- 各ステージの classification_report
- LightGBM の各ステージ特徴量重要度
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ============================================================
# 1. 設定
# ============================================================

FINAL_LABELS = [
    "SINGLE",
    "XBH",
    "BB_HBP",
    "SO",
    "OTHER_OUT",
]
FINAL_LABEL_TO_ID = {label: i for i, label in enumerate(FINAL_LABELS)}
FINAL_ID_TO_LABEL = {i: label for label, i in FINAL_LABEL_TO_ID.items()}

STAGE1_LABELS = ["ON_BASE", "OUT"]
STAGE1_LABEL_TO_ID = {label: i for i, label in enumerate(STAGE1_LABELS)}
STAGE1_ID_TO_LABEL = {i: label for label, i in STAGE1_LABEL_TO_ID.items()}

ONBASE_LABELS = ["SINGLE", "XBH", "BB_HBP"]
ONBASE_LABEL_TO_ID = {label: i for i, label in enumerate(ONBASE_LABELS)}
ONBASE_ID_TO_LABEL = {i: label for label, i in ONBASE_LABEL_TO_ID.items()}

OUT_LABELS = ["SO", "OTHER_OUT"]
OUT_LABEL_TO_ID = {label: i for i, label in enumerate(OUT_LABELS)}
OUT_ID_TO_LABEL = {i: label for label, i in OUT_LABEL_TO_ID.items()}

# 元の細かい打席結果 → 5クラスへ変換
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

# 69特徴量版の特徴量定義
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


def convert_final_label(raw_label: str) -> str:
    """細かい打席結果を SINGLE / XBH / BB_HBP / SO / OTHER_OUT の5クラスへ変換する。"""
    x = str(raw_label).strip()

    if x in SINGLE_SET:
        return "SINGLE"
    if x in DOUBLE_SET or x in TRIPLE_SET or x in HR_SET:
        return "XBH"
    if x in BB_SET or x in HBP_SET:
        return "BB_HBP"
    if x in SO_SET:
        return "SO"
    return "OTHER_OUT"


def final_to_stage1(final_label: str) -> str:
    """最終5分類ラベルを第1段階の ON_BASE / OUT に変換する。"""
    if final_label in {"SINGLE", "XBH", "BB_HBP"}:
        return "ON_BASE"
    return "OUT"


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
    """CSVを読み込み、型変換と2段階分類用ラベルを作成する。"""
    df = pd.read_csv(csv_path)

    required_columns = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["game_date", "label"])
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"必要な列がCSVに存在しません: {missing}")

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # 最終5分類
    df["final_label"] = df["label"].apply(convert_final_label)
    df["final_target"] = df["final_label"].map(FINAL_LABEL_TO_ID).astype(int)

    # 第1段階: ON_BASE / OUT
    df["stage1_label"] = df["final_label"].apply(final_to_stage1)
    df["stage1_target"] = df["stage1_label"].map(STAGE1_LABEL_TO_ID).astype(int)

    # 第2段階: ON_BASE側 / OUT側
    df["onbase_label"] = np.where(df["stage1_label"] == "ON_BASE", df["final_label"], pd.NA)
    df["out_label"] = np.where(df["stage1_label"] == "OUT", df["final_label"], pd.NA)
    df["onbase_target"] = df["onbase_label"].map(ONBASE_LABEL_TO_ID)
    df["out_target"] = df["out_label"].map(OUT_LABEL_TO_ID)

    df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

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


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """共通評価指標を計算する。"""
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
        max_k = min(3, y_prob.shape[1])
        for k in range(2, max_k + 1):
            topk = np.argsort(y_prob, axis=1)[:, -k:]
            result[f"top{k}_accuracy"] = float(
                np.mean([y_true[i] in topk[i] for i in range(len(y_true))])
            )

    return result


def save_classification_outputs(
    output_dir: Path,
    model_name: str,
    labels: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """classification_report / confusion_matrix / predictions を保存する。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob)
    metrics["model"] = model_name

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        digits=6,
    )
    (output_dir / f"{model_name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(output_dir / f"{model_name}_confusion_matrix.csv", encoding="utf-8-sig")

    pred_df = pd.DataFrame({
        "true_id": y_true,
        "pred_id": y_pred,
        "true_label": [labels[int(x)] for x in y_true],
        "pred_label": [labels[int(x)] for x in y_pred],
    })
    if y_prob is not None:
        for i, label in enumerate(labels):
            pred_df[f"prob_{label}"] = y_prob[:, i]

    pred_df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False, encoding="utf-8-sig")
    return metrics


def save_stage_report(
    output_dir: Path,
    model_name: str,
    stage_name: str,
    labels: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """各ステージ単体のclassification reportを保存する。"""
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        digits=6,
    )
    (output_dir / f"{model_name}_{stage_name}_classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        output_dir / f"{model_name}_{stage_name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )


def make_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return weights.astype(np.float32)


def combine_two_stage_prob(
    p_stage1: np.ndarray,
    p_onbase: np.ndarray,
    p_out: np.ndarray,
) -> np.ndarray:
    """各ステージの確率から最終5分類の確率を合成する。"""
    final_prob = np.zeros((len(p_stage1), len(FINAL_LABELS)), dtype=np.float32)
    p_on = p_stage1[:, STAGE1_LABEL_TO_ID["ON_BASE"]]
    p_o = p_stage1[:, STAGE1_LABEL_TO_ID["OUT"]]

    final_prob[:, FINAL_LABEL_TO_ID["SINGLE"]] = p_on * p_onbase[:, ONBASE_LABEL_TO_ID["SINGLE"]]
    final_prob[:, FINAL_LABEL_TO_ID["XBH"]] = p_on * p_onbase[:, ONBASE_LABEL_TO_ID["XBH"]]
    final_prob[:, FINAL_LABEL_TO_ID["BB_HBP"]] = p_on * p_onbase[:, ONBASE_LABEL_TO_ID["BB_HBP"]]
    final_prob[:, FINAL_LABEL_TO_ID["SO"]] = p_o * p_out[:, OUT_LABEL_TO_ID["SO"]]
    final_prob[:, FINAL_LABEL_TO_ID["OTHER_OUT"]] = p_o * p_out[:, OUT_LABEL_TO_ID["OTHER_OUT"]]

    # 念のため正規化
    denom = final_prob.sum(axis=1, keepdims=True)
    final_prob = np.divide(final_prob, denom, out=np.zeros_like(final_prob), where=denom != 0)
    return final_prob


# ============================================================
# 3. LightGBM 2段階分類
# ============================================================

def _prepare_lgbm_frames(split: SplitData) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
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

    return train_df, valid_df, test_df, feature_cols


def _fit_lgbm_classifier(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    num_classes: int,
    stage_name: str,
    output_dir: Path,
    seed: int,
):
    import lightgbm as lgb

    X_train = train_df[feature_cols]
    y_train = train_df[target_col].astype(int).values
    X_valid = valid_df[feature_cols]
    y_valid = valid_df[target_col].astype(int).values

    if num_classes == 2:
        model = lgb.LGBMClassifier(
            objective="binary",
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
            class_weight="balanced",
        )
        eval_metric = "binary_logloss"
    else:
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
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
            class_weight="balanced",
        )
        eval_metric = "multi_logloss"

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    print(f"\n[LightGBM] stage={stage_name}, target={target_col}, class_weight=balanced")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=eval_metric,
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=callbacks,
    )

    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    fi.to_csv(output_dir / f"lightgbm_{stage_name}_feature_importance.csv", index=False, encoding="utf-8-sig")

    return model


def train_evaluate_lightgbm_two_stage(split: SplitData, output_dir: Path, seed: int = 42) -> Optional[Dict[str, float]]:
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        print("[WARN] lightgbm がインストールされていないため、LightGBM評価をスキップします。")
        print("       インストールする場合: pip install lightgbm")
        return None

    print("\n" + "=" * 80)
    print("LightGBM two-stage evaluation")
    print("=" * 80)

    train_df, valid_df, test_df, feature_cols = _prepare_lgbm_frames(split)

    # 第1段階: ON_BASE / OUT
    stage1_model = _fit_lgbm_classifier(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        target_col="stage1_target",
        num_classes=len(STAGE1_LABELS),
        stage_name="stage1_onbase_vs_out",
        output_dir=output_dir,
        seed=seed,
    )

    # 第2段階: ON_BASE側 SINGLE / XBH / BB_HBP
    train_on = train_df[train_df["stage1_label"] == "ON_BASE"].copy()
    valid_on = valid_df[valid_df["stage1_label"] == "ON_BASE"].copy()
    stage2_on_model = _fit_lgbm_classifier(
        train_df=train_on,
        valid_df=valid_on,
        feature_cols=feature_cols,
        target_col="onbase_target",
        num_classes=len(ONBASE_LABELS),
        stage_name="stage2_onbase_single_xbh_bbhbp",
        output_dir=output_dir,
        seed=seed,
    )

    # 第2段階: OUT側 SO / OTHER_OUT
    train_out = train_df[train_df["stage1_label"] == "OUT"].copy()
    valid_out = valid_df[valid_df["stage1_label"] == "OUT"].copy()
    stage2_out_model = _fit_lgbm_classifier(
        train_df=train_out,
        valid_df=valid_out,
        feature_cols=feature_cols,
        target_col="out_target",
        num_classes=len(OUT_LABELS),
        stage_name="stage2_out_so_otherout",
        output_dir=output_dir,
        seed=seed,
    )

    # 各ステージ単体レポート
    p_stage1_test = stage1_model.predict_proba(test_df[feature_cols])
    pred_stage1 = np.argmax(p_stage1_test, axis=1)
    save_stage_report(
        output_dir, "lightgbm", "stage1_onbase_vs_out", STAGE1_LABELS,
        test_df["stage1_target"].astype(int).values, pred_stage1,
    )

    test_on = test_df[test_df["stage1_label"] == "ON_BASE"].copy()
    if len(test_on) > 0:
        pred_on = np.argmax(stage2_on_model.predict_proba(test_on[feature_cols]), axis=1)
        save_stage_report(
            output_dir, "lightgbm", "stage2_onbase_single_xbh_bbhbp", ONBASE_LABELS,
            test_on["onbase_target"].astype(int).values, pred_on,
        )

    test_out = test_df[test_df["stage1_label"] == "OUT"].copy()
    if len(test_out) > 0:
        pred_out = np.argmax(stage2_out_model.predict_proba(test_out[feature_cols]), axis=1)
        save_stage_report(
            output_dir, "lightgbm", "stage2_out_so_otherout", OUT_LABELS,
            test_out["out_target"].astype(int).values, pred_out,
        )

    # 最終5分類確率へ合成
    p_onbase_test = stage2_on_model.predict_proba(test_df[feature_cols])
    p_out_test = stage2_out_model.predict_proba(test_df[feature_cols])
    final_prob = combine_two_stage_prob(p_stage1_test, p_onbase_test, p_out_test)
    final_pred = np.argmax(final_prob, axis=1)
    y_true = test_df["final_target"].astype(int).values

    metrics = save_classification_outputs(
        output_dir, "lightgbm_two_stage", FINAL_LABELS, y_true, final_pred, final_prob
    )

    print("\n[LightGBM Two-stage Final Test Metrics]")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v}")

    return metrics


# ============================================================
# 4. FM 2段階分類
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
            self.category_maps[col] = {v: i + 1 for i, v in enumerate(values)}  # 0: unknown

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
        return [len(self.category_maps[col]) + 1 for col in self.categorical_features]


class FMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: FeatureEncoder, target_col: str):
        self.x_cat = encoder.transform_categorical(df)
        self.x_num = encoder.transform_numeric(df)
        self.y = df[target_col].astype(np.int64).values

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.long),
        }


class FMInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, encoder: FeatureEncoder):
        self.x_cat = encoder.transform_categorical(df)
        self.x_num = encoder.transform_numeric(df)

    def __len__(self) -> int:
        return len(self.x_cat)

    def __getitem__(self, idx: int):
        return {
            "x_cat": torch.tensor(self.x_cat[idx], dtype=torch.long),
            "x_num": torch.tensor(self.x_num[idx], dtype=torch.float32),
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
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in cat_cardinalities
        ])
        self.num_embeddings = nn.Parameter(torch.randn(num_numeric, embed_dim) * 0.01)

        self.cat_linear = nn.ModuleList([
            nn.Embedding(cardinality, num_classes) for cardinality in cat_cardinalities
        ])
        self.num_linear = nn.Parameter(torch.randn(num_numeric, num_classes) * 0.01)
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

        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embs = torch.stack(cat_embs, dim=1)
        num_embs = x_num.unsqueeze(-1) * self.num_embeddings.unsqueeze(0)
        all_embs = torch.cat([cat_embs, num_embs], dim=1)
        all_embs = self.dropout(all_embs)

        summed = torch.sum(all_embs, dim=1)
        interaction = 0.5 * ((summed * summed) - torch.sum(all_embs * all_embs, dim=1))
        interaction_logits = nn.functional.linear(
            interaction,
            torch.ones(self.num_classes, self.embed_dim, device=interaction.device) / self.embed_dim,
            None,
        )

        linear_logits = self.bias.unsqueeze(0).expand(batch_size, -1)
        for i, lin in enumerate(self.cat_linear):
            linear_logits = linear_logits + lin(x_cat[:, i])
        linear_logits = linear_logits + torch.einsum("bn,nc->bc", x_num, self.num_linear)
        return linear_logits + interaction_logits


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
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
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_fm(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []
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


@torch.no_grad()
def predict_fm_proba(
    model: nn.Module,
    df: pd.DataFrame,
    encoder: FeatureEncoder,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    ds = FMInferenceDataset(df, encoder)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs = []
    for batch in tqdm(loader, desc="FM predict", leave=False):
        x_cat = batch["x_cat"].to(device)
        x_num = batch["x_num"].to(device)
        logits = model(x_cat, x_num)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_fm_stage(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    output_dir: Path,
    stage_name: str,
    target_col: str,
    label_names: List[str],
    seed: int,
    batch_size: int,
    epochs: int,
    patience: int,
    embed_dim: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
):
    print(f"\n[FM] stage={stage_name}, target={target_col}")

    encoder = FeatureEncoder(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    encoder.fit(train_df)

    train_ds = FMDataset(train_df, encoder, target_col)
    valid_ds = FMDataset(valid_df, encoder, target_col)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = FactorizationMachineClassifier(
        cat_cardinalities=encoder.categorical_cardinalities(),
        num_numeric=len(NUMERIC_FEATURES),
        num_classes=len(label_names),
        embed_dim=embed_dim,
        dropout=0.1,
    ).to(device)

    y_train = train_df[target_col].astype(int).values
    weights_np = make_class_weights(y_train, len(label_names))
    print("[FM] class_weight:")
    for label, weight in zip(label_names, weights_np):
        print(f"  {label}: {weight:.6f}")

    class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_valid_loss = float("inf")
    best_state = None
    best_epoch = 0
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, y_valid, pred_valid, prob_valid = evaluate_fm(model, valid_loader, criterion, device)
        valid_metrics = compute_metrics(y_valid, pred_valid, prob_valid)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
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

    pd.DataFrame(history).to_csv(output_dir / f"fm_{stage_name}_history.csv", index=False, encoding="utf-8-sig")
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "label_names": label_names,
            "target_col": target_col,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
        },
        output_dir / f"fm_{stage_name}_best_model.pt",
    )

    return model, encoder, criterion, best_epoch, best_valid_loss


def train_evaluate_fm_two_stage(
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
    print("\n" + "=" * 80)
    print("Factorization Machine two-stage evaluation")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()

    stage1_model, stage1_encoder, stage1_criterion, stage1_best_epoch, stage1_best_loss = train_fm_stage(
        train_df, valid_df, output_dir, "stage1_onbase_vs_out", "stage1_target", STAGE1_LABELS,
        seed, batch_size, epochs, patience, embed_dim, lr, weight_decay, device,
    )

    train_on = train_df[train_df["stage1_label"] == "ON_BASE"].copy()
    valid_on = valid_df[valid_df["stage1_label"] == "ON_BASE"].copy()
    on_model, on_encoder, on_criterion, on_best_epoch, on_best_loss = train_fm_stage(
        train_on, valid_on, output_dir, "stage2_onbase_single_xbh_bbhbp", "onbase_target", ONBASE_LABELS,
        seed, batch_size, epochs, patience, embed_dim, lr, weight_decay, device,
    )

    train_out = train_df[train_df["stage1_label"] == "OUT"].copy()
    valid_out = valid_df[valid_df["stage1_label"] == "OUT"].copy()
    out_model, out_encoder, out_criterion, out_best_epoch, out_best_loss = train_fm_stage(
        train_out, valid_out, output_dir, "stage2_out_so_otherout", "out_target", OUT_LABELS,
        seed, batch_size, epochs, patience, embed_dim, lr, weight_decay, device,
    )

    # ステージ単体レポート
    stage1_test_ds = FMDataset(test_df, stage1_encoder, "stage1_target")
    stage1_test_loader = DataLoader(stage1_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    _, y_s1, pred_s1, p_s1 = evaluate_fm(stage1_model, stage1_test_loader, stage1_criterion, device)
    save_stage_report(output_dir, "fm", "stage1_onbase_vs_out", STAGE1_LABELS, y_s1, pred_s1)

    test_on = test_df[test_df["stage1_label"] == "ON_BASE"].copy()
    if len(test_on) > 0:
        on_test_ds = FMDataset(test_on, on_encoder, "onbase_target")
        on_test_loader = DataLoader(on_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        _, y_on, pred_on, _ = evaluate_fm(on_model, on_test_loader, on_criterion, device)
        save_stage_report(output_dir, "fm", "stage2_onbase_single_xbh_bbhbp", ONBASE_LABELS, y_on, pred_on)

    test_out = test_df[test_df["stage1_label"] == "OUT"].copy()
    if len(test_out) > 0:
        out_test_ds = FMDataset(test_out, out_encoder, "out_target")
        out_test_loader = DataLoader(out_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        _, y_out, pred_out, _ = evaluate_fm(out_model, out_test_loader, out_criterion, device)
        save_stage_report(output_dir, "fm", "stage2_out_so_otherout", OUT_LABELS, y_out, pred_out)

    # 全テスト行に対する第2段階確率を計算して最終5分類へ合成
    p_stage1 = predict_fm_proba(stage1_model, test_df, stage1_encoder, device, batch_size=batch_size)
    p_onbase = predict_fm_proba(on_model, test_df, on_encoder, device, batch_size=batch_size)
    p_out = predict_fm_proba(out_model, test_df, out_encoder, device, batch_size=batch_size)
    final_prob = combine_two_stage_prob(p_stage1, p_onbase, p_out)
    final_pred = np.argmax(final_prob, axis=1)
    y_true = test_df["final_target"].astype(int).values

    metrics = save_classification_outputs(
        output_dir, "fm_two_stage", FINAL_LABELS, y_true, final_pred, final_prob
    )
    metrics["stage1_best_epoch"] = int(stage1_best_epoch)
    metrics["stage1_best_valid_loss"] = float(stage1_best_loss)
    metrics["stage2_onbase_best_epoch"] = int(on_best_epoch)
    metrics["stage2_onbase_best_valid_loss"] = float(on_best_loss)
    metrics["stage2_out_best_epoch"] = int(out_best_epoch)
    metrics["stage2_out_best_valid_loss"] = float(out_best_loss)

    print("\n[FM Two-stage Final Test Metrics]")
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
        default="data/train_all_added_69_features.csv",
        help="69特徴量を追加したCSVファイルパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_lgbm_fm_69_features_two_stage_so_out_class_weight",
        help="評価結果の出力先ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fm_epochs", type=int, default=50)
    parser.add_argument("--fm_batch_size", type=int, default=512)
    parser.add_argument("--fm_embed_dim", type=int, default=32)
    parser.add_argument("--fm_lr", type=float, default=1e-3)
    parser.add_argument("--fm_patience", type=int, default=8)
    parser.add_argument("--skip_lightgbm", action="store_true", help="LightGBMをスキップする")
    parser.add_argument("--skip_fm", action="store_true", help="FMをスキップする")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    set_seed(args.seed)

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LightGBM / FM comparison for 69-feature CSV / two-stage target / class_weight")
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

    print("\n[Final target distribution]")
    final_dist = df["final_label"].value_counts().reindex(FINAL_LABELS).fillna(0).astype(int)
    print(final_dist)
    final_dist.to_csv(output_dir / "final_target_distribution.csv", encoding="utf-8-sig")

    print("\n[Stage1 target distribution]")
    stage1_dist = df["stage1_label"].value_counts().reindex(STAGE1_LABELS).fillna(0).astype(int)
    print(stage1_dist)
    stage1_dist.to_csv(output_dir / "stage1_target_distribution.csv", encoding="utf-8-sig")

    print("\n[Stage2 ON_BASE target distribution]")
    onbase_dist = df.loc[df["stage1_label"] == "ON_BASE", "onbase_label"].value_counts().reindex(ONBASE_LABELS).fillna(0).astype(int)
    print(onbase_dist)
    onbase_dist.to_csv(output_dir / "stage2_onbase_target_distribution.csv", encoding="utf-8-sig")

    print("\n[Stage2 OUT target distribution]")
    out_dist = df.loc[df["stage1_label"] == "OUT", "out_label"].value_counts().reindex(OUT_LABELS).fillna(0).astype(int)
    print(out_dist)
    out_dist.to_csv(output_dir / "stage2_out_target_distribution.csv", encoding="utf-8-sig")

    with open(output_dir / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "task": "two_stage_classification",
                "stage1_labels": STAGE1_LABELS,
                "stage2_onbase_labels": ONBASE_LABELS,
                "stage2_out_labels": OUT_LABELS,
                "onbase_vs_context_numeric_features": ONBASE_VS_CONTEXT_NUMERIC_FEATURES,
                "batted_ball_team_numeric_features": BATTED_BALL_TEAM_NUMERIC_FEATURES,
                "interleague_season_progress_numeric_features": INTERLEAGUE_SEASON_PROGRESS_NUMERIC_FEATURES,
                "team_categorical_features": ["batter_team_id", "pitcher_team_id"],
                "final_labels": FINAL_LABELS,
                "categorical_features": CATEGORICAL_FEATURES,
                "numeric_features": NUMERIC_FEATURES,
                "base_numeric_features": BASE_NUMERIC_FEATURES,
                "hand_split_numeric_features": HAND_SPLIT_NUMERIC_FEATURES,
                "recent_numeric_features": RECENT_NUMERIC_FEATURES,
                "context_numeric_features": CONTEXT_NUMERIC_FEATURES,
                "cumulative_rate_numeric_features": CUMULATIVE_RATE_NUMERIC_FEATURES,
                "event_context_numeric_features": EVENT_CONTEXT_NUMERIC_FEATURES,
                "class_weight": {
                    "lightgbm": "balanced for each stage",
                    "fm": "balanced weights computed from train split for each stage",
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    all_metrics = []

    if not args.skip_lightgbm:
        lgb_metrics = train_evaluate_lightgbm_two_stage(split, output_dir, seed=args.seed)
        if lgb_metrics is not None:
            all_metrics.append(lgb_metrics)

    if not args.skip_fm:
        fm_metrics = train_evaluate_fm_two_stage(
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
