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

def standardize_hand_value(hand):
    """CSV内の表記（R/L/B、右/左/両など）を統一する関数"""
    hand = str(hand).strip().upper()
    if hand in ["R", "右", "右打", "右投"]: return "右"
    if hand in ["L", "左", "左打", "左投"]: return "左"
    if hand in ["B", "S", "両", "両打"]: return "両"
    return "不明"

def visualize_tsne_by_hand():
    # --- データの準備 ---
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    id_to_name = {v: k for k, v in master["name_to_id"].items()}

    # 左右の投打情報をCSVから取得して辞書にまとめる
    h_df = pd.read_csv("initial_stats_2024.csv")
    p_df = pd.read_csv("pitcher_stats_2024_all.csv")
    
    # 投手かどうかを100%確実に判定するためのセット（スペース除去済み）
    csv_pitcher_names = set(str(name).replace(" ", "").replace("　", "") for name in p_df['name'].unique())
    
    # 選手ごとの左右（Hand）情報を格納する辞書
    raw_player_hand = {}
    
    # 打者CSVから左右（Hand列）を読み込み（列名が小文字 'hand' の場合にも対応）
    h_hand_col = 'Hand' if 'Hand' in h_df.columns else ('hand' if 'hand' in h_df.columns else None)
    if h_hand_col:
        for _, row in h_df.iterrows():
            name_clean = str(row['name']).replace(" ", "").replace("　", "")
            raw_player_hand[name_clean] = standardize_hand_value(row[h_hand_col])
            
    # 投手CSVから左右（Hand列）を読み込み
    p_hand_col = 'Hand' if 'Hand' in p_df.columns else ('hand' if 'hand' in p_df.columns else None)
    if p_hand_col:
        for _, row in p_df.iterrows():
            name_clean = str(row['name']).replace(" ", "").replace("　", "")
            raw_player_hand[name_clean] = standardize_hand_value(row[p_hand_col])

    # --- モデルのロードと重み抽出 ---
    num_players = master["total_players"]
    model = FactorizationMachineModel(num_players=num_players)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    
    embeddings = model.player_v.weight.detach().numpy()
    target_embeddings = embeddings[1:] # ID 0を除外

    # --- t-SNEによる次元圧縮 ---
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    low_dim_embs = tsne.fit_transform(target_embeddings)

    # --- 可視化用データの整理 (投打×左右のカテゴリ分類) ---
    plot_data = []
    for i, vec in enumerate(low_dim_embs):
        p_id = i + 1
        name = id_to_name.get(p_id, "Unknown")
        name_cleaned = name.replace(" ", "").replace("　", "")
        
        # CSVから左右のタイプを取得（なければ一端"不明"に）
        hand_type = raw_player_hand.get(name_cleaned, "不明")
        
        # 「投手／野手」と「左右」を掛け合わせてカテゴリを決定
        if name_cleaned in csv_pitcher_names:
            # 投手の場合
            if hand_type == "右": category = "右投"
            elif hand_type == "左": category = "左投"
            else: category = "右投" # 投手の左右列がCSVにない場合の安全策（デフォルト右投）
        else:
            # 野手の場合
            if hand_type == "右": category = "右打"
            elif hand_type == "左": category = "左打"
            elif hand_type == "両": category = "両打"
            else: category = "右打" # 打撃の左右情報がない場合の安全策（デフォルト右打）
        
        plot_data.append({
            'x': vec[0], 'y': vec[1], 'Player': name, 'Hand_Category': category
        })
    df_plot = pd.DataFrame(plot_data)

    # --- カラー設定 ---
    hand_colors = {
        "右投": "navy",       # 濃い青
        "左投": "deepskyblue",# 鮮やかな水色
        "右打": "crimson",    # 濃い赤
        "左打": "gold",       # オレンジ・黄色系
        "両打": "green"       # 緑
    }

    # 凡例の並び順を固定
    category_order = ["右投", "左投", "右打", "左打", "両打"]

    # --- プロットの実行 (自動調整スケーリング) ---
    plt.figure(figsize=(14, 10))
    plt.grid(True, alpha=0.3)

    sns.scatterplot(
        data=df_plot, 
        x='x', 
        y='y', 
        hue='Hand_Category', 
        hue_order=category_order,
        palette=hand_colors, 
        s=80,           
        alpha=0.8,
        edgecolor='none'
    )

    plt.title("選手エンベディングの分布（左右・投打別カラー）", fontsize=15)
    plt.legend(title="投打の左右", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig("tsne_player_hand_colors.png")
    print("画像を保存しました: tsne_player_hand_colors.png")
    plt.show()

if __name__ == "__main__":
    visualize_tsne_by_hand()