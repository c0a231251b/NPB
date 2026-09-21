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

def standardize_team_name(name):
    """球団名を指定された日本語表記に統一する関数"""
    name = str(name)
    if any(x in name for x in ["巨人", "読売", "G"]): return "巨人"
    if any(x in name for x in ["阪神", "T"]): return "阪神"
    if any(x in name for x in ["DeNA", "ＤｅＮＡ", "横浜", "DB"]): return "DeNA"
    if any(x in name for x in ["広島", "C"]): return "広島"
    if any(x in name for x in ["ヤクルト", "S"]): return "ヤクルト"
    if any(x in name for x in ["中日", "D"]): return "中日"
    if any(x in name for x in ["ソフトバンク", "H"]): return "ソフトバンク"
    if any(x in name for x in ["日本ハム", "日ハム", "F"]): return "日本ハム"
    if any(x in name for x in ["ロッテ", "M"]): return "ロッテ"
    if any(x in name for x in ["楽天", "E"]): return "楽天"
    if any(x in name for x in ["オリックス", "B"]): return "オリックス"
    if any(x in name for x in ["西武", "L"]): return "西武"
    return "その他"

def visualize_tsne_selective():
    # =====================================================
    # 【ここを編集】比較したいチーム名をリストで指定してください
    # target_teams = ["巨人","阪神","DeNA","広島","ヤクルト","中日"]
    target_teams = ["ソフトバンク","日本ハム","ロッテ","楽天","オリックス","西武"]
    # =====================================================

    # --- データの準備 ---
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}

    h_df = pd.read_csv("initial_stats_2024.csv")
    p_df = pd.read_csv("pitcher_stats_2024_all.csv")
    
    player_to_team = {}
    for _, row in h_df.iterrows():
        player_to_team[row['name']] = standardize_team_name(row['team'])
    for _, row in p_df.iterrows():
        player_to_team[row['name']] = standardize_team_name(row['team'])

    # --- モデルのロードと重み抽出 ---
    num_players = master["total_players"]
    model = FactorizationMachineModel(num_players=num_players)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    
    embeddings = model.player_v.weight.detach().numpy()
    target_embeddings = embeddings[1:]

    # --- t-SNEによる次元圧縮 ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    low_dim_embs = tsne.fit_transform(target_embeddings)

    # --- 可視化用データの整理 ---
    plot_data = []
    for i, vec in enumerate(low_dim_embs):
        p_id = i + 1
        name = id_to_name.get(p_id, "Unknown")
        team = player_to_team.get(name, "不明")
        
        plot_data.append({
            'x': vec[0], 
            'y': vec[1], 
            'Player': name, 
            'Team': team
        })
    df_plot = pd.DataFrame(plot_data)

    # --- データのフィルタリング ---
    df_filtered = df_plot[df_plot['Team'].isin(target_teams)].copy()

    # --- カラー設定 ---
    team_colors = {
        "巨人": "orange", "阪神": "gold", "DeNA": "blue", "広島": "red",
        "ヤクルト": "green", "中日": "dodgerblue", "ソフトバンク": "yellow",
        "日本ハム": "skyblue", "ロッテ": "black", "楽天": "crimson",
        "オリックス": "navy", "西武": "midnightblue"
    }

    # --- プロットの実行 ---
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        data=df_filtered, 
        x='x', 
        y='y', 
        hue='Team', 
        hue_order=target_teams,
        palette=team_colors, 
        s=120, 
        alpha=0.9,
        edgecolor='white', 
        linewidth=0.5
    )

    # --- 軸の範囲と目盛りの固定 ---
    # キャンバスの表示範囲を固定 (xlim, ylim)
    plt.xlim(-40, 25)
    plt.ylim(-25, 15)

    # 軸に表示する数値ラベルを明示的に指定 (xticks, yticks)
    plt.xticks([-40, -30, -20, -10, 0, 10, 20])
    plt.yticks([-20, -15, -10, -5, 0, 5, 10])

    # 名前ラベルの表示
    # 一次的にコメントアウト
    """"
    for i, row in df_filtered.iterrows():
        plt.annotate(row['Player'], (row['x'], row['y']), 
                     fontsize=9, family='MS Gothic', alpha=0.8)
    """
    plt.title(f"選手エンベディング比較: {' vs '.join(target_teams)}", fontsize=15)
    plt.legend(title="球団名", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"tsne_compare_{'_'.join(target_teams)}.png"
    plt.savefig(filename)
    print(f"画像を保存しました: {filename}")
    plt.show()

if __name__ == "__main__":
    visualize_tsne_selective()