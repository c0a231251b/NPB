import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. 推論結果ファイルの読み込み ===
df1 = pd.read_csv("c:/Users/Admin/Desktop/NPB/evaluate/code/output_lgbm_fm_69_features_two_stage_so_out_class_weight/fm_two_stage_predictions.csv")
df2 = pd.read_csv("c:/Users/Admin/Desktop/NPB/evaluate/code/output_lgbm_fm_69_features_two_stage_so_out_class_weight/lightgbm_two_stage_predictions.csv")

# === 2. データフレームを辞書にまとめる ===
models = {
    "FM Two-Stage": df1,
    "LightGBM Two-Stage": df2
}

# === 3. 1×2 の subplot を作成（横並びで見やすく） ===
fig, axes = plt.subplots(1, 2, figsize=(22, 10))
axes = axes.flatten()

# === 4. 各モデルの混同行列を subplot に描画 ===
for ax, (model_name, df) in zip(axes, models.items()):
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    sns.heatmap(
        cm_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        cbar=True,
        annot_kws={"size": 12}
    )

    ax.set_title(f"{model_name}", fontsize=18, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)

plt.tight_layout()
plt.show()

# テキストで混同行列を表示
for model_name, df in models.items():
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n混同行列 - {model_name}:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

