# 次元数確認スクリプト　作業後消す。
import pandas as pd
import numpy as np


df = pd.read_pickle("fm_dataset_7d_30d.pkl")
df1=pd.read_pickle("fm_dataset_score_diff_7d_30d.pkl")
df2=pd.read_pickle("fm_dataset.pkl")
df3=pd.read_pickle("fm_dataset_score_diff.pkl")

print("\nfm_dataset_7d_30d.pkl の内容:")
print(df.head())
print(len(df["num_features"].iloc[0]))
print(df["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset_7d_30d.pkl の num_features の内容 (200行目から210行目まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(200, min(211, len(df))):
    print(f"行 {i}: {df['num_features'].iloc[i]}")

print("\nfm_dataset_score_diff_7d_30d.pkl の内容:")
print(df1.head())
print(len(df1["num_features"].iloc[0]))
print(df1["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset_score_diff_7d_30d.pkl の num_features の内容 (200行目から210行目まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(200, min(211, len(df1))):
    print(f"行 {i}: {df1['num_features'].iloc[i]}")

print("\nfm_dataset.pkl の内容:")
print(df2.head())
print(len(df2["num_features"].iloc[0]))
print(df2["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset.pkl の num_features の内容 (5行まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(min(5, len(df2))):
    print(f"行 {i}: {df2['num_features'].iloc[i]}")

print("\nfm_dataset_score_diff.pkl の内容:")
print(df3.head())
print(len(df3["num_features"].iloc[0]))
print(df3["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset_score_diff.pkl の num_features の内容 (5行まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(min(5, len(df3))):
    print(f"行 {i}: {df3['num_features'].iloc[i]}")





# コメントアウト
"""

df = pd.read_pickle("fm_dataset_7d_30d.pkl")
df1=pd.read_pickle("fm_dataset_score_diff_7d_30d.pkl")
df2=pd.read_pickle("fm_dataset.pkl")
df3=pd.read_pickle("fm_dataset_score_diff.pkl")

print("\nfm_dataset_7d_30d.pkl の内容:")
print(df.head())
print(len(df["num_features"].iloc[0]))
print(df["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset_7d_30d.pkl の num_features の内容 (34行目から44行目まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(34, min(45, len(df))):
    print(f"行 {i}: {df['num_features'].iloc[i]}")

print("\nfm_dataset_score_diff_7d_30d.pkl の内容:")
print(df1.head())
print(len(df1["num_features"].iloc[0]))
print(df1["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset_score_diff_7d_30d.pkl の num_features の内容 (34行目から44行目まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(34, min(45, len(df1))):
    print(f"行 {i}: {df1['num_features'].iloc[i]}")

print("\nfm_dataset.pkl の内容:")
print(df2.head())
print(len(df2["num_features"].iloc[0]))
print(df2["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset.pkl の num_features の内容 (5行まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(min(5, len(df2))):
    print(f"行 {i}: {df2['num_features'].iloc[i]}")

print("\nfm_dataset_score_diff.pkl の内容:")
print(df3.head())
print(len(df3["num_features"].iloc[0]))
print(df3["num_features"].head())
# 60行までnum_featuresの内容を表示
print("\nfm_dataset_score_diff.pkl の num_features の内容 (5行まで):")
print("[contrast_30d_mean,contrast_30d_std,contrast_7d_mean,contrast_7d_std]")
for i in range(min(5, len(df3))):
    print(f"行 {i}: {df3['num_features'].iloc[i]}")

"""