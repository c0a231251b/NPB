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

def visualize_tsne_by_individual_orders():
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
    
    print(" 1,716個のリアル打線データから各選手のスタメン起用実績を集計中...")
    for _, row in df_lineups.iterrows():
        for order_idx, col_name in enumerate(order_columns, 1):
            p_name = row[col_name]
            if pd.isna(p_name) or p_name == "Unknown" or not isinstance(p_name, str):
                continue
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
    target_embeddings = embeddings[1:] # ID=0を除外

    # --- t-SNEによる次元圧縮 ---
    print(" 16次元の打者潜在空間を t-SNE で2次元へ圧縮中...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    low_dim_embs = tsne.fit_transform(target_embeddings)

    # --- 表記ブレに対応したマッピング関数の定義 ---
    def get_real_order(master_name):
        if pd.isna(master_name): return "対象外"
        name_clean = master_name.replace(" ", "").replace("　", "")
        if name_clean in player_main_order:
            return player_main_order[name_clean]
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

    # --- 9色のカスタムカラーマップの設定 ---
    order_colors = {
        "1番打者": "blue",
        "2番打者": "deepskyblue",
        "3番打者": "darkorange",
        "4番打者": "red",
        "5番打者": "gold",
        "6番打者": "purple",
        "7番打者": "magenta",
        "8番打者": "green",
        "9番打者": "limegreen"
    }
    
    order_list = [f"{i}番打者" for i in range(1, 10)]

    # --- 3行3列のマルチパネルプロットの実行 ---
    print(" 3行3列の打順別比較プロットを生成しています...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 16), sharex=True, sharey=True)
    axes = axes.flatten() # 2次元配列のaxesを1次元にフラット化してループしやすくする

    # 軸の範囲を固定して比較しやすくするために最大・最小値を取得
    x_min, x_max = df_viz['x'].min() * 1.1, df_viz['x'].max() * 1.1
    y_min, y_max = df_viz['y'].min() * 1.1, df_viz['y'].max() * 1.1

    # 各打順を1つずつサブプロットに描画
    for idx, target_order in enumerate(order_list):
        ax = axes[idx]
        
        # 1. 背景として「他のすべての選手」を薄いグレーでプロット
        df_bg = df_viz[df_viz['Order'] != target_order]
        ax.scatter(
            df_bg['x'], df_bg['y'], 
            c='lightgray', s=40, alpha=0.4, label='他の打順の選手', edgecolor='none'
        )
        
        # 2. 前景として「該当する打順の選手」だけを固有カラーで強調プロット
        df_fg = df_viz[df_viz['Order'] == target_order]
        ax.scatter(
            df_fg['x'], df_fg['y'], 
            c=order_colors[target_order], s=90, alpha=0.9, 
            label=target_order, edgecolor='w', linewidth=0.5
        )
        
        """
        # 主要選手のプロット位置へのテキストラベル表示（見やすさ重視でシンプルに配置）
        targets = ["近本光司", "周東佑京", "牧秀悟", "岡本和真", "村上宗隆", "万波中正", "佐藤輝明", "甲斐拓也", "戸郷翔征"]
        for _, row in df_fg.iterrows():
            if row['Player'] in targets:
                ax.annotate(
                    row['Player'], (row['x'], row['y']),
                    xytext=(4, 4), textcoords='offset points',
                    fontsize=9, fontweight='bold', family='MS Gothic',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", edgecolor="gray", alpha=0.8)
                )
        """
        # サブプロットごとの装飾設定
        ax.set_title(f"【{target_order}】の分布", fontsize=12, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

    # 全体タイトルの設定
    plt.suptitle("FMモデル打者潜在空間における打順別（1番〜9番）t-SNE投影マトリクス", fontsize=16, fontweight='bold', y=0.96)
    
    # 共有の軸ラベルを設定
    fig.text(0.5, 0.02, 't-SNE Dimension 1', ha='center', fontsize=12)
    fig.text(0.02, 0.5, 't-SNE Dimension 2', va='center', rotation='vertical', fontsize=12)
    
    plt.tight_layout(rect=[0.03, 0.03, 0.97, 0.94])
    
    output_png = "tsne_player_vectors_matrix_by_order.png"
    plt.savefig(output_png, dpi=300)
    plt.show()
    print(f"成功しました！3行3列マトリクス仕様の t-SNE 散布図を保存しました: {output_png}")

if __name__ == "__main__":
    visualize_tsne_by_individual_orders()