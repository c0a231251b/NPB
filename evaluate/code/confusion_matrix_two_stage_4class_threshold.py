import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. 推論結果ファイルの読み込み ===
df2 = pd.read_csv("c:/Users/Admin/Desktop/NPB/evaluate/code/output_lgbm_fm_69_features_two_stage_so_out_class_weight_threshold/lightgbm_two_stage_threshold_predictions.csv")
df1 = pd.read_csv("c:/Users/Admin/Desktop/NPB/evaluate/code/output_lgbm_fm_69_features_two_stage_so_out_class_weight_threshold/lightgbm_two_stage_argmax_predictions.csv")
df4 = pd.read_csv("c:/Users/Admin/Desktop/NPB/evaluate/code/output_lgbm_fm_69_features_two_stage_so_out_class_weight_threshold/fm_two_stage_threshold_predictions.csv")
df3 = pd.read_csv("c:/Users/Admin/Desktop/NPB/evaluate/code/output_lgbm_fm_69_features_two_stage_so_out_class_weight_threshold/fm_two_stage_argmax_predictions.csv")

# === 2. データフレームを辞書にまとめる ===
models = {
    "LightGBM Argmax": df1,
    "LightGBM Threshold": df2,
    "FM Argmax": df3,
    "FM Threshold": df4
}

# === 3. 1×4 の subplot を作成（横並びで見やすく） ===
fig, axes = plt.subplots(1, 4, figsize=(32, 8))
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
        annot_kws={"size": 14}
    )

    ax.set_title(f"{model_name}", fontsize=20, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=16)
    ax.set_ylabel("True Label", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

plt.tight_layout()
plt.show()

# === 5. テキストで混同行列を表示 ===
for model_name, df in models.items():
    y_true = df["true_label"]
    y_pred = df["pred_label"]

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\n混同行列 - {model_name}:")
    print(pd.DataFrame(cm, index=labels, columns=labels))
