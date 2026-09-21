# 次元数確認スクリプト　作業後消す。

import pandas as pd
import numpy as np

# データセットの読み込み
df = pd.read_pickle("fm_dataset_7d_30d.pkl")
df1 = pd.read_pickle("fm_dataset_score_diff_7d_30d.pkl") 
df2 = pd.read_pickle("fm_dataset.pkl")
df3 = pd.read_pickle("fm_dataset_score_diff.pkl")

# 件数確認(カテゴリ特徴量の0の個数を数える)
X = np.stack(df["cat_features"])
X1 = np.stack(df1["cat_features"])
X2 = np.stack(df2["cat_features"])
X3 = np.stack(df3["cat_features"])


print(f"fm_dataset_7d_30d.pkl のカテゴリ特徴量の0の個数: {(X == 0).sum()}")
print(f"fm_dataset_score_diff_7d_30d.pkl のカテゴリ特徴量の0の個数: {(X1 == 0).sum()}")
print(f"fm_dataset.pkl のカテゴリ特徴量の0の個数: {(X2 == 0).sum()}")
print(f"fm_dataset_score_diff.pkl のカテゴリ特徴量の0の個数: {(X3 == 0).sum()}")