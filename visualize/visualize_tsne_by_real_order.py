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

# 1. モデルの定義 (元の構造を完全に維持)
class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49):
        super(FactorizationMachineModel, self).__init__()
        self.player_v = nn.Embedding(num_players, k)

def visualize_tsne_by_real_order():
    # --- データの準備 ---
    # 1. マスターデータの読み込み
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}

    # 2. リアルスタメンCSVの読み込みと「最多起用打順」の集計
    lineup_file = "all_games_starting_lineups.csv"
    df_lineups = pd.read_csv(lineup_file)
    
    player_counts = {}
    order_columns = [f"Order_{i}" for i in range(1, 10)]
    
    print("1,716個のリアル打線データから各選手のスタメン起用実績を集計中...")
    for _, row in df_lineups.iterrows():
        for order_idx, col_name in enumerate(order_columns, 1):
            p_name = row[col_name]
            if pd.isna(p_name) or p_name == "Unknown" or not isinstance(p_name, str):
                continue
            # 全角・半角スペースを排除して集計用のキーにする
            name_clean = p_name.replace(" ", "").replace("　", "")
            if name_clean not in player_counts:
                player_counts[name_clean] = {num: 0 for num in range(1, 10)}
            player_counts[name_clean][order_idx] += 1

    # 各選手の「最多起用打順」を確定する辞書を作成
    player_main_order = {}
    for p_name, counts in player_counts.items():
        max_order = max(counts, key=counts.get)
        if counts[max_order] > 0:
            player_main_order[p_name] = f"{max_order}番打者"

    # --- モデルのロードと重み抽出 ---
    num_players = master["total_players"]
    model = FactorizationMachineModel(num_players=num_players)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    
    embeddings = model.player_v.weight.detach().numpy()
    target_embeddings = embeddings[1:] # ID=0 (パディング等) を除外

    # --- t-SNEによる次元圧縮 ---
    print("16次元の打者潜在空間を t-SNE で2次元へ圧縮中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    low_dim_embs = tsne.fit_transform(target_embeddings)

    # --- 表記ブレに対応したマッピング関数の定義 ---
    def get_real_order(master_name):
        if pd.isna(master_name): return "対象外"
        # スペースを完全に排除
        name_clean = master_name.replace(" ", "").replace("　", "")
        
        # 1. 完全一致
        if name_clean in player_main_order:
            return player_main_order[name_clean]
        # 2. 部分・前方一致
        for real_name, order_label in player_main_order.items():
            if name_clean in real_name or real_name in name_clean:
                return order_label
        return "対象外"

    # --- 可視化用データの整理 ---
    plot_data = []
    for i, vec in enumerate(low_dim_embs):
        p_id = i + 1
        name = id_to_name.get(p_id, "Unknown")
        order_label = get_real_order(name)
        
        plot_data.append({
            'x': vec[0], 
            'y': vec[1], 
            'Player': name, 
            'Order': order_label
        })
    df_plot = pd.DataFrame(plot_data)

    # 先発出場実績があり、打順が紐付いた選手のみを抽出
    df_viz = df_plot[df_plot['Order'] != "対象外"].copy()

    # --- 凡例の並び順の固定 ---
    order_order = [f"{i}番打者" for i in range(1, 10)]
    
    # 人間がパッと見て区別しやすい高コントラストなカスタムカラーマップ
    order_colors = {
        "1番打者": "blue",         # 鮮やかな青
        "2番打者": "deepskyblue",  # 水色
        "3番打者": "darkorange",   # オレンジ
        "4番打者": "red",          # 赤（主砲）
        "5番打者": "gold",         # 金・黄（ポイントゲッター）
        "6番打者": "purple",       # 紫
        "7番打者": "magenta",      # ピンク・赤紫
        "8番打者": "green",        # 緑（捕手層など）
        "9番打者": "limegreen"     # 明るい黄緑
    }

    # --- プロットの実行 ---
    plt.figure(figsize=(14, 10))
    
    scatter = sns.scatterplot(
        data=df_viz, 
        x='x', 
        y='y', 
        hue='Order', 
        hue_order=order_order, 
        palette=order_colors, # 9色の高コントラストパレットを適用
        s=100,                # 視認性向上のためドットサイズをやや大きく(90->100)
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    """
    # 主要選手（各打順の代表格や主力選手）のラベル表示（復活させました）
    targets = ["近本光司", "周東佑京", "牧秀悟", "岡本和真", "村上宗隆", "万波中正", "佐藤輝明", "甲斐拓也", "戸郷翔征"]
    for i, row in df_viz.iterrows():
        if row['Player'] in targets:
            plt.annotate(row['Player'], (row['x'], row['y']), 
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=10, fontweight='bold', family='MS Gothic',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", edgecolor="gray", alpha=0.7))
    """
    plt.title("FMモデル打者潜在空間（2次元）の t-SNE 投影と実際の最多起用打順による分布", fontsize=15, fontweight='bold', pad=15)
    plt.legend(title="2025年実際の最多起用打順", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11, title_fontsize=12)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    output_png = "tsne_player_vectors_by_real_order.png"
    plt.savefig(output_png, dpi=300)
    plt.show()
    print(f"成功しました！人間識別重視カラーの t-SNE 散布図を保存しました: {output_png}")

if __name__ == "__main__":
    visualize_tsne_by_real_order()