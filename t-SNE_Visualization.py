import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic' # Windowsの標準フォントを指定
# -----------------------

class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49):
        super(FactorizationMachineModel, self).__init__()
        self.player_v = nn.Embedding(num_players, k)

def visualize_tsne():
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}

    model = FactorizationMachineModel(num_players=697)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    embeddings = model.player_v.weight.detach().numpy()

    # ID 0を除外して圧縮
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    low_dim_embs = tsne.fit_transform(embeddings[1:])

    plot_data = []
    for i, vec in enumerate(low_dim_embs):
        p_id = i + 1
        name = id_to_name.get(p_id, "Unknown")
        plot_data.append({'x': vec[0], 'y': vec[1], 'Player': name})
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 10))
    # 散布図の描画
    sns.scatterplot(data=df_plot, x='x', y='y', color='skyblue', alpha=0.6)

    # 注目選手のラベル表示
    #targets = ["坂本勇人", "岡本和真", "戸郷翔征", "菅野智之", "門脇誠", "小林誠司"]
    targets = ["山田哲人", "岡本和真", "佐藤輝明","菊池涼介", "細川成也", "桑原将志","万波中正","山川穂高","頓宮裕真","源田壮亮", "浅村栄斗", "藤原恭大"]
    for i, row in df_plot.iterrows():
        if row['Player'] in targets:
            plt.annotate(row['Player'], (row['x'], row['y']), 
                         fontsize=11, fontweight='bold', family='MS Gothic')

    plt.title("選手エンベディングのクラスタリング (t-SNE)", fontsize=15)
    plt.xlabel("次元 1")
    plt.ylabel("次元 2")
    plt.grid(True, alpha=0.3)
    plt.savefig("player_tsne_clusters.png")
    print("✅ 日本語対応版画像を保存しました。")

if __name__ == "__main__":
    visualize_tsne()