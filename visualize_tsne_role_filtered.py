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

def visualize_tsne_role_filtered():
    # =====================================================
    # 【ここを編集】表示したい選手名を指定してください
    targets = ["戸郷翔征", "岡本和真"]
    # =====================================================

    # --- データの準備 ---
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}

    # 役割判定用のデータロード
    h_df = pd.read_csv("initial_stats_2024.csv")
    p_df = pd.read_csv("pitcher_stats_2024_all.csv")
    pitcher_names = set(p_df['name'].unique())

    # --- モデルのロードと重み抽出 ---
    num_players = master["total_players"]
    model = FactorizationMachineModel(num_players=num_players)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    
    embeddings = model.player_v.weight.detach().numpy()
    target_embeddings = embeddings[1:] # ID 0を除外

    # --- t-SNEによる次元圧縮 (全選手で計算) ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    low_dim_embs = tsne.fit_transform(target_embeddings)

    # --- 可視化用データの整理 ---
    plot_data = []
    for i, vec in enumerate(low_dim_embs):
        p_id = i + 1
        name = id_to_name.get(p_id, "Unknown")
        
        if name in pitcher_names:
            role = "投手"
        else:
            role = "NOT投手"
        
        plot_data.append({
            'x': vec[0], 
            'y': vec[1], 
            'Player': name,
            'Role': role
        })
    df_plot = pd.DataFrame(plot_data)

    # --- 描画対象のフィルタリング ---
    # ここで指定した選手のみのデータフレームを作成します
    df_filtered = df_plot[df_plot['Player'].isin(targets)].copy()

    # --- カラー設定 ---
    role_colors = {"投手": "blue", "NOT投手": "red"}

    # --- プロットの実行 ---
    plt.figure(figsize=(12, 10))
    
    # 指定した選手のみを描画
    sns.scatterplot(
        data=df_filtered, 
        x='x', 
        y='y', 
        hue='Role', 
        palette=role_colors, 
        s=150, # 選手を絞るので少し大きく
        alpha=1.0, # はっきり表示
        edgecolor='black',
        linewidth=1
    )

    # ラベルの表示
    for i, row in df_filtered.iterrows():
        plt.annotate(row['Player'], (row['x'], row['y']), 
                     fontsize=12, fontweight='bold', family='MS Gothic',
                     xytext=(5, 5), textcoords='offset points', # 点から少しずらす
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

    # --- 軸の固定 (前回の設定を維持) ---
    plt.xlim(-40, 25)
    plt.ylim(-25, 15)
    plt.xticks([-40, -30, -20, -10, 0, 10, 20])
    plt.yticks([-20, -15, -10, -5, 0, 5, 10])

    plt.title(f"選手エンベディング位置確認: {', '.join(targets)}", fontsize=15)
    plt.legend(title="役割", loc='upper right')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    plt.savefig("tsne_role_filtered.png")
    plt.show()

if __name__ == "__main__":
    visualize_tsne_role_filtered()