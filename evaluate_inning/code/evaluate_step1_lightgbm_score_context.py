# -*- coding: utf-8 -*-
"""
Step 1: LightGBM score-context evaluation for inning-level run occurrence prediction.

目的:
- メインタスク NO_RUN / RUN_SCORED に対して、score context を追加した場合に
  得点イニング検出力、とくに RUN_SCORED Recall / F1 が改善するかを確認する。
- 同じ train / valid / test 分割で以下を比較する。
    1. base_24_features
       イニング開始時点の打順、相手投手、チーム状態、球場、直近得点傾向
    2. score_context_28_features
       base_24_features + score_diff_before_inning + score_state_before_inning
       + is_close_game + is_blowout

実行例:
python evaluate_step1_lightgbm_score_context.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
python evaluate_step1_lightgbm_score_context.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv"
軽い確認:
python evaluate_step1_lightgbm_score_context.py --csv_path "C:/Users/Admin/Desktop/NPB/evaluate_inning/data/train_inning_runs_added_9_features.csv" --dry_run

必要ライブラリ:
pip install pandas numpy scikit-learn lightgbm

注意:
- score_state_before_inning / is_close_game / is_blowout は、score_diff_before_inning から作成する。
- score_diff_before_inning は攻撃チーム視点の点差として扱う。
    > 0: lead
    = 0: tie
    < 0: behind
- is_close_game は abs(score_diff_before_inning) <= 3
- is_blowout は abs(score_diff_before_inning) >= 6
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
    average_precision_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

LABELS = ["NO_RUN", "RUN_SCORED"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}
POS_ID = LABEL_TO_ID["RUN_SCORED"]

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


@dataclass
class FeatureConfig:
    name: str
    categorical_features: List[str]
    numeric_features: List[str]

    @property
    def total_features(self) -> int:
        return len(self.categorical_features) + len(self.numeric_features)


@dataclass
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    test_df: pd.DataFrame


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


def add_score_context_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "score_diff_before_inning" not in df.columns:
        raise ValueError("score_diff_before_inning がCSVに存在しません。")

    score_diff = pd.to_numeric(df["score_diff_before_inning"], errors="coerce").fillna(0)
    df["score_diff_before_inning"] = score_diff

    df["score_state_before_inning"] = np.select(
        [score_diff > 0, score_diff == 0, score_diff < 0],
        ["lead", "tie", "behind"],
        default="unknown",
    )
    df["is_close_game"] = (score_diff.abs() <= 3).astype(int)
    df["is_blowout"] = (score_diff.abs() >= 6).astype(int)
    return df


def build_feature_configs() -> List[FeatureConfig]:
    base = FeatureConfig(
        name="base_24_features",
        categorical_features=BASE_CATEGORICAL_FEATURES.copy(),
        numeric_features=BASE_NUMERIC_FEATURES.copy(),
    )
    score_context = FeatureConfig(
        name="score_context_28_features",
        categorical_features=BASE_CATEGORICAL_FEATURES.copy() + SCORE_CONTEXT_CATEGORICAL_FEATURES.copy(),
        numeric_features=BASE_NUMERIC_FEATURES.copy() + SCORE_CONTEXT_NUMERIC_FEATURES.copy(),
    )
    return [base, score_context]


def prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.copy()

    if "game_date" not in df.columns:
        raise ValueError("game_date がCSVに存在しません。")

    if "target_run_scored" not in df.columns and "target_runs_in_inning" not in df.columns:
        raise ValueError("target_run_scored または target_runs_in_inning がCSVに存在しません。")

    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = add_score_context_features(df)

    if "target_run_scored" in df.columns:
        df["target_label"] = df["target_run_scored"].astype(str).str.strip()
    else:
        runs = pd.to_numeric(df["target_runs_in_inning"], errors="coerce").fillna(0)
        df["target_label"] = np.where(runs > 0, "RUN_SCORED", "NO_RUN")

    unknown = sorted(set(df["target_label"].dropna().unique()) - set(LABELS))
    if unknown:
        raise ValueError(f"想定外のtarget_labelがあります: {unknown}")

    df["target"] = df["target_label"].map(LABEL_TO_ID).astype(int)

    if "top_bottom" in df.columns:
        df["top_bottom"] = df["top_bottom"].apply(normalize_top_bottom)

    all_cats = sorted(set(BASE_CATEGORICAL_FEATURES + SCORE_CONTEXT_CATEGORICAL_FEATURES))
    all_nums = sorted(set(BASE_NUMERIC_FEATURES + SCORE_CONTEXT_NUMERIC_FEATURES))

    missing = sorted(set(all_cats + all_nums) - set(df.columns))
    if missing:
        raise ValueError(f"必要な特徴量列がCSVに存在しません: {missing}")

    for col in all_cats:
        df[col] = df[col].fillna("<NA>").astype(str)
    for col in all_nums:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("game_date").reset_index(drop=True)
    return df


def split_by_game_date(df: pd.DataFrame, train_ratio: float = 0.8, valid_ratio: float = 0.1) -> SplitData:
    dates = sorted(df["game_date"].dropna().unique())
    if len(dates) < 3:
        raise ValueError("game_dateのユニーク数が少なすぎます。")

    n_dates = len(dates)
    train_end = int(n_dates * train_ratio)
    valid_end = int(n_dates * (train_ratio + valid_ratio))

    train_dates = set(dates[:train_end])
    valid_dates = set(dates[train_end:valid_end])
    test_dates = set(dates[valid_end:])

    return SplitData(
        train_df=df[df["game_date"].isin(train_dates)].copy(),
        valid_df=df[df["game_date"].isin(valid_dates)].copy(),
        test_df=df[df["game_date"].isin(test_dates)].copy(),
    )


def class_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray]) -> Dict[str, float]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average=None,
        zero_division=0,
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
        "run_scored_pred_count": int(np.sum(y_pred == POS_ID)),
    }

    if y_prob is not None:
        result["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
        pos_prob = y_prob[:, POS_ID]
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, pos_prob))
        except ValueError:
            result["roc_auc"] = np.nan
        try:
            result["pr_auc"] = float(average_precision_score(y_true, pos_prob))
        except ValueError:
            result["pr_auc"] = np.nan
    return result


def save_feature_list(output_dir: Path, config: FeatureConfig) -> None:
    rows = []
    for col in config.categorical_features:
        rows.append({"feature": col, "feature_type": "categorical"})
    for col in config.numeric_features:
        rows.append({"feature": col, "feature_type": "numeric"})
    pd.DataFrame(rows).to_csv(output_dir / f"{config.name}_features.csv", index=False, encoding="utf-8-sig")


def train_evaluate_lightgbm(split: SplitData, config: FeatureConfig, output_dir: Path, seed: int) -> Dict[str, float]:
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError("lightgbm が必要です。pip install lightgbm を実行してください。") from e

    variant_dir = output_dir / config.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    save_feature_list(variant_dir, config)

    train_df = split.train_df.copy()
    valid_df = split.valid_df.copy()
    test_df = split.test_df.copy()

    feature_cols = config.categorical_features + config.numeric_features

    for col in config.categorical_features:
        categories = pd.Index(pd.concat([train_df[col], valid_df[col], test_df[col]], axis=0).astype(str).unique())
        dtype = pd.CategoricalDtype(categories=categories)
        train_df[col] = train_df[col].astype(dtype)
        valid_df[col] = valid_df[col].astype(dtype)
        test_df[col] = test_df[col].astype(dtype)

    medians = train_df[config.numeric_features].median(numeric_only=True)
    train_df[config.numeric_features] = train_df[config.numeric_features].fillna(medians)
    valid_df[config.numeric_features] = valid_df[config.numeric_features].fillna(medians)
    test_df[config.numeric_features] = test_df[config.numeric_features].fillna(medians)

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

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        categorical_feature=config.categorical_features,
        callbacks=callbacks,
    )

    y_prob = model.predict_proba(X_test)
    y_pred = np.argmax(y_prob, axis=1)
    metrics = class_metrics(y_test, y_pred, y_prob)
    metrics.update(
        {
            "variant": config.name,
            "model": "lightgbm",
            "n_categorical_features": len(config.categorical_features),
            "n_numeric_features": len(config.numeric_features),
            "n_total_features": config.total_features,
        }
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=LABELS,
        zero_division=0,
        digits=6,
    )
    (variant_dir / f"{config.name}_classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(
        variant_dir / f"{config.name}_confusion_matrix.csv", encoding="utf-8-sig"
    )

    pred_df = pd.DataFrame(
        {
            "true_id": y_test,
            "pred_id": y_pred,
            "true_label": [ID_TO_LABEL[int(x)] for x in y_test],
            "pred_label": [ID_TO_LABEL[int(x)] for x in y_pred],
            "prob_NO_RUN": y_prob[:, 0],
            "prob_RUN_SCORED": y_prob[:, 1],
        }
    )
    pred_df.to_csv(variant_dir / f"{config.name}_predictions.csv", index=False, encoding="utf-8-sig")

    fi = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    fi.to_csv(variant_dir / f"{config.name}_feature_importance.csv", index=False, encoding="utf-8-sig")

    print(f"\n[LightGBM Test Metrics: {config.name}]")
    for key, value in metrics.items():
        if key not in {"variant", "model"}:
            print(f"{key}: {value}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/train_inning_runs_added_9_features.csv",
        help="train_inning_runs_added_9_features.csv のパス",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_step1_lightgbm_score_context",
        help="出力先ディレクトリ",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="学習せずに分割・特徴量だけ確認する")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Step 1: LightGBM score-context comparison")
    print("=" * 80)
    print(f"csv_path: {csv_path}")
    print(f"output_dir: {output_dir}")
    print("target: NO_RUN / RUN_SCORED")

    df = prepare_dataframe(csv_path)
    split = split_by_game_date(df)
    configs = build_feature_configs()

    print("\n[Data split]")
    print(f"all  : {len(df):>6} rows, {df['game_date'].min().date()} ~ {df['game_date'].max().date()}, dates={df['game_date'].nunique()}")
    print(f"train: {len(split.train_df):>6} rows, {split.train_df['game_date'].min().date()} ~ {split.train_df['game_date'].max().date()}, dates={split.train_df['game_date'].nunique()}")
    print(f"valid: {len(split.valid_df):>6} rows, {split.valid_df['game_date'].min().date()} ~ {split.valid_df['game_date'].max().date()}, dates={split.valid_df['game_date'].nunique()}")
    print(f"test : {len(split.test_df):>6} rows, {split.test_df['game_date'].min().date()} ~ {split.test_df['game_date'].max().date()}, dates={split.test_df['game_date'].nunique()}")

    print("\n[Target distribution]")
    dist = df["target_label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    print(dist)
    dist.to_csv(output_dir / "target_distribution.csv", encoding="utf-8-sig")

    feature_summary_rows = []
    print("\n[Feature variants]")
    for config in configs:
        print(f"{config.name}: categorical={len(config.categorical_features)}, numeric={len(config.numeric_features)}, total={config.total_features}")
        feature_summary_rows.append(
            {
                "variant": config.name,
                "n_categorical_features": len(config.categorical_features),
                "n_numeric_features": len(config.numeric_features),
                "n_total_features": config.total_features,
                "categorical_features": json.dumps(config.categorical_features, ensure_ascii=False),
                "numeric_features": json.dumps(config.numeric_features, ensure_ascii=False),
            }
        )
    pd.DataFrame(feature_summary_rows).to_csv(output_dir / "step1_feature_variant_summary.csv", index=False, encoding="utf-8-sig")

    if args.dry_run:
        print("\nDry run completed. No training executed.")
        return

    rows = []
    for config in configs:
        print("\n" + "-" * 80)
        print(f"LightGBM evaluation: variant={config.name}")
        print("-" * 80)
        print(f"categorical_features ({len(config.categorical_features)}): {config.categorical_features}")
        print(f"numeric_features ({len(config.numeric_features)}): {config.numeric_features}")
        rows.append(train_evaluate_lightgbm(split, config, output_dir, seed=args.seed))

    comparison_df = pd.DataFrame(rows)
    preferred_cols = [
        "variant",
        "model",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "no_run_precision",
        "no_run_recall",
        "no_run_f1",
        "run_scored_precision",
        "run_scored_recall",
        "run_scored_f1",
        "run_scored_pred_count",
        "log_loss",
        "roc_auc",
        "pr_auc",
        "n_categorical_features",
        "n_numeric_features",
        "n_total_features",
    ]
    comparison_df = comparison_df[[c for c in preferred_cols if c in comparison_df.columns]]
    comparison_df.to_csv(output_dir / "step1_lightgbm_score_context_comparison_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Step 1 comparison")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {output_dir / 'step1_lightgbm_score_context_comparison_metrics.csv'}")
    print("Done.")


if __name__ == "__main__":
    main()
