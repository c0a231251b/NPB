import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic'

# 1. モデルの定義
class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49):
        super(FactorizationMachineModel, self).__init__()
        self.player_v = nn.Embedding(num_players, k)

def visualize_tsne_by_role_verify():
    # --- データの準備 ---
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}

    # 投手か野手かを判定するためのリストを作成
    h_df = pd.read_csv("initial_stats_2024.csv")
    p_df = pd.read_csv("pitcher_stats_2024_all.csv")
    
    pitcher_names = set(p_df['name'].unique())

    # --- モデルのロードと重み抽出 ---
    num_players = master["total_players"]
    model = FactorizationMachineModel(num_players=num_players)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    
    embeddings = model.player_v.weight.detach().numpy()
    target_embeddings = embeddings[1:] # ID 0を除外

    # --- t-SNEによる次元圧縮 ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    low_dim_embs = tsne.fit_transform(target_embeddings)

    # --- 可視化用データの整理 ---
    plot_data = []
    for i, vec in enumerate(low_dim_embs):
        p_id = i + 1
        name = id_to_name.get(p_id, "Unknown")
        
        # 役割の判定
        if name in pitcher_names:
            role = "投手"
        else:
            role = "NOT投手"
        
        plot_data.append({
            'x': vec[0], 
            'y': vec[1], 
            'Player': name, # ここが重要！
            'Role': role
        })
    df_plot = pd.DataFrame(plot_data)

    # --- カラー設定 ---
    role_colors = {"投手": "blue", "NOT投手": "red"}

    # --- プロットの実行 ---
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df_plot, 
        x='x', 
        y='y', 
        hue='Role', 
        palette=role_colors, 
        s=60, 
        alpha=0.6
    )

    # 注目選手のラベル表示
    targets = ["戸郷翔征", "岡本和真"]
    for i, row in df_plot.iterrows():
        if row['Player'] in targets:
            # ラベルが見やすいように白い背景（bbox）を付けています
            plt.annotate(row['Player'], (row['x'], row['y']), 
                         fontsize=12, fontweight='bold', family='MS Gothic',
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5))

    plt.title("選手エンベディングの分布（役割確認用）", fontsize=15)
    plt.legend(title="役割", loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig("tsne_role_verify.png")
    plt.show()

if __name__ == "__main__":
    visualize_tsne_by_role_verify()