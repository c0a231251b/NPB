import pickle
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import os

def load_dataset(pkl_path):
    """FMデータセットをLightGBM用に展開"""

    with open(pkl_path, "rb") as f:
        df = pickle.load(f)

    # cat_featuresを展開
    cat_df = pd.DataFrame(
        df["cat_features"].tolist(),
        columns=[
            "batter1", "batter2", "batter3",
            "batter4", "batter5", "batter6",
            "batter7", "batter8", "batter9",
            "pitcher"
        ]
    )

    # num_featuresを展開
    num_df = pd.DataFrame(
        df["num_features"].tolist(),
        columns=[
            "contrast_mean",
            "contrast_std"
        ]
    )

    # 特徴量結合
    X = pd.concat([cat_df, num_df], axis=1)

    y = df["target"]

    return X, y, X.columns.tolist()


def train_lightgbm(X, y, feature_cols, dataset_name, test_size=0.2, random_state=42):
    """LightGBM で学習し、評価指標を返す"""

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
    y_pred = model.predict(
        X_test,
        num_iteration=model.best_iteration
        )

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


if __name__ == "__main__":

    datasets = {
        "打率-被打率(7日間)"      : "fm_dataset_7d.pkl",
        "打率-被打率(30日間)"     : "fm_dataset_30d.pkl",

        "出塁率-被出塁率(7日間)"  : "fm_dataset_7d_OBP.pkl",
        "出塁率-被出塁率(30日間)" : "fm_dataset_30d_OBP.pkl",

        "長打率-被長打率(7日間)"  : "fm_dataset_7d_SLG.pkl",
        "長打率-被長打率(30日間)" : "fm_dataset_30d_SLG.pkl",

        "OPS-被OPS(7日間)"       : "fm_dataset_7d_OPS.pkl",
        "OPS-被OPS(30日間)"      : "fm_dataset_30d_OPS.pkl",

        "ISO-被ISO(7日間)"       : "fm_dataset_7d_ISO.pkl",
        "ISO-被ISO(30日間)"      : "fm_dataset_30d_ISO.pkl",

        "三振率-奪三振率(7日間)"  : "fm_dataset_7d_K.pkl",
        "三振率-奪三振率(30日間)" : "fm_dataset_30d_K.pkl",

        "四球率-与四球率(7日間)"  : "fm_dataset_7d_BB.pkl",
        "四球率-与四球率(30日間)" : "fm_dataset_30d_BB.pkl",

        "本塁打率-被本塁打率(7日間)"  : "fm_dataset_7d_HR.pkl",
        "本塁打率-被本塁打率(30日間)" : "fm_dataset_30d_HR.pkl",
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
            X,
            y,
            feature_cols,
            dataset_name
        )

        results.append(metrics)

        print("\nTop20 Feature Importance")
        print(importance.head(20))

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