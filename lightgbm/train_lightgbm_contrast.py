import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import os

# ============================================================
# contrast9 対応版データロード
# ============================================================
def load_dataset(pkl_path):
    """contrast9 形式（展開済み DataFrame）を読み込む"""

    df = pickle.load(open(pkl_path, "rb"))

    # 特徴量（target 以外）
    X = df.drop(columns=["target"])

    # 目的変数
    y = df["target"]

    return X, y, X.columns.tolist()



# ============================================================
# LightGBM 学習
# ============================================================
def train_lightgbm(X, y, feature_cols, dataset_name, test_size=0.2, random_state=42):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": 42,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )

    # 予測
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    # 評価指標
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n==============================")
    print(f"LightGBM Evaluation Results：{dataset_name}")
    print("==============================")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")
    print("==============================\n")

    # Feature Importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance()
    }).sort_values("importance", ascending=False)

    print("Top 20 Feature Importance")
    print(importance.head(20))

    return model, importance, {
        "Dataset": dataset_name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2
    }


# ============================================================
# メイン処理（contrast9 データセット一覧）
# ============================================================
if __name__ == "__main__":

    datasets = {
        "打率-被打率(7日間)"      : "fm_dataset_7d_contrast9.pkl",
        "打率-被打率(30日間)"     : "fm_dataset_30d_contrast9.pkl",

        "出塁率-被出塁率(7日間)"      : "fm_dataset_7d_OBP_contrast9.pkl",
        "出塁率-被出塁率(30日間)"     : "fm_dataset_30d_OBP_contrast9.pkl",

        "長打率-被長打率(7日間)"      : "fm_dataset_7d_SLG_contrast9.pkl",
        "長打率-被長打率(30日間)"     : "fm_dataset_30d_SLG_contrast9.pkl",

        "OPS-被OPS(7日間)"           : "fm_dataset_7d_OPS_contrast9.pkl",
        "OPS-被OPS(30日間)"          : "fm_dataset_30d_OPS_contrast9.pkl",

        "ISO-被ISO(7日間)"           : "fm_dataset_7d_ISO_contrast9.pkl",
        "ISO-被ISO(30日間)"          : "fm_dataset_30d_ISO_contrast9.pkl",

        "三振率-奪三振率(7日間)"      : "fm_dataset_7d_K_contrast9.pkl",
        "三振率-奪三振率(30日間)"     : "fm_dataset_30d_K_contrast9.pkl",

        "四球率-与四球率(7日間)"      : "fm_dataset_7d_BB_contrast9.pkl",
        "四球率-与四球率(30日間)"     : "fm_dataset_30d_BB_contrast9.pkl",

        "本塁打率-被本塁打率(7日間)"  : "fm_dataset_7d_HR_contrast9.pkl",
        "本塁打率-被本塁打率(30日間)" : "fm_dataset_30d_HR_contrast9.pkl",
    }

    results = []

    for dataset_name, pkl_path in datasets.items():

        print("\n")
        print("=" * 70)
        print(f"学習開始 : {dataset_name}")
        print("=" * 70)

        if not os.path.exists(pkl_path):
            print(f"{pkl_path} が存在しません。スキップします。")
            continue

        X, y, feature_cols = load_dataset(pkl_path)

        model, importance, metrics = train_lightgbm(
            X, y, feature_cols, dataset_name
        )

        results.append(metrics)

    # ==========================
    # 全データセット比較
    # ==========================
    result_df = (
        pd.DataFrame(results)
        .sort_values("RMSE")
        .reset_index(drop=True)
    )

    print("\n\n")
    print("=" * 80)
    print("全データセット比較（RMSE昇順）")
    print("=" * 80)
    print(result_df)
