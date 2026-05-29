import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic'

def run_real_correlation_analysis():
    # --- 1. データの読み込み ---
    cluster_file = "extracted_player_clusters_K4.csv"
    lineup_file = "all_games_starting_lineups.csv"
    
    if not os.path.exists(cluster_file):
        print(f"Error: '{cluster_file}' が見つかりません。")
        return
    if not os.path.exists(lineup_file):
        print(f"Error: '{lineup_file}' が見つかりません。")
        return

    # クラスターデータ (Player_ID, Player_Name, Cluster_ID)
    df_clusters = pd.read_csv(cluster_file)
    # ★修正: 半角・全角スペースを確実に両方とも排除
    df_clusters['Player_Clean'] = df_clusters['Player_Name'].str.replace(" ", "", regex=False).str.replace(" ", "", regex=False)
    
    # スターティングラインナップCSV
    df_lineups = pd.read_csv(lineup_file)

    print("1716個のリアル打線データから各選手の打順起用実績を集計中...")

    # --- 2. 各選手が1番〜9番の各打順で先発出場した回数を集計 ---
    player_counts = {}
    order_columns = [f"Order_{i}" for i in range(1, 10)]
    
    for _, row in df_lineups.iterrows():
        for order_idx, col_name in enumerate(order_columns, 1):
            p_name = row[col_name]
            if pd.isna(p_name) or p_name == "Unknown" or not isinstance(p_name, str):
                continue
                
            # ★修正: 集計側も半角・全角スペースを完全に排除
            name_clean = p_name.replace(" ", "").replace(" ", "")
            
            if name_clean not in player_counts:
                player_counts[name_clean] = {num: 0 for num in range(1, 10)}
            player_counts[name_clean][order_idx] += 1

    # --- 3. 各選手の「実際の最頻打順」を確定 ---
    player_main_order = {}
    for p_name, counts in player_counts.items():
        max_order = max(counts, key=counts.get)
        if counts[max_order] > 0:
            player_main_order[p_name] = f"{max_order}番打者"

    # --- 4. 表記ブレ（姓のみ vs フルネーム）に極めて強いマッピング処理 ---
    def match_real_order(cluster_name):
        if pd.isna(cluster_name): return np.nan
        # 1. 完全一致で検索
        if cluster_name in player_main_order:
            return player_main_order[cluster_name]
        # 2. 前方一致・部分一致で検索（「岡本」が「岡本和真」の最頻打順にヒットするようにする安全策）
        for real_name, order_label in player_main_order.items():
            if cluster_name in real_name or real_name in cluster_name:
                return order_label
        return np.nan

    # 判定用関数を適用してマージ
    df_clusters['Real_Order'] = df_clusters['Player_Clean'].apply(match_real_order)

    # 有効なクラスターIDと実際の打順実績が両方紐付いた選手だけを抽出
    df_analysis = df_clusters.dropna(subset=['Real_Order', 'Cluster_ID']).copy()
    df_analysis['Cluster_ID'] = df_analysis['Cluster_ID'].astype(int)

    # --- 5. クロス集計（相関行列）の実行 ---
    order_labels = [f"{i}番打者" for i in range(1, 10)]
    
    # 人数ベースのクロス集計
    cross_tab = pd.crosstab(df_analysis['Cluster_ID'], df_analysis['Real_Order'])
    cross_tab = cross_tab.reindex(columns=order_labels).fillna(0).astype(int)
    
    # クラスターごとの比率（%）ベースのクロス集計
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    # --- 6. コンソールへの集計結果出力 ---
    print("\n" + "="*70)
    print(" 【打者潜在空間×2025スタメン実績】相関行列 (該当選手数)")
    print("="*70)
    print(cross_tab)
    print("\n" + "="*70)
    print(" 各打者クラスター内における実際の起用打順の構成比率 (%)")
    print("="*70)
    print(cross_tab_pct.round(1).to_string())
    print("="*70)
    print(f"今回正しく紐付けに成功した総野手レギュラー数: {len(df_analysis)} 名")

    # --- 7. 純度100%の相関ヒートマップの生成と保存 ---
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        cross_tab_pct, 
        annot=cross_tab, 
        fmt="d", 
        cmap="YlGnBu", 
        linewidths=1.0,
        cbar_kws={'label': 'クラスター内での起用割合 (%)'}
    )
    
    plt.title("FMモデル打者クラスター（K=4）と2025年実際のスタメン起用打順の数理的相関関係", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("2025年シーズン全試合から集計した『最も起用頻度の高かった実際の打順』", fontsize=12, labelpad=10)
    plt.ylabel("FM潜在空間（V）から算出した打者クラスターID", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    output_fig = "cluster_real_batting_order_heatmap_K4.png"
    plt.savefig(output_fig, dpi=300)
    plt.close()
    
    print(f"\n 成功しました！真の打順相関ヒートマップ画像を保存しました: {output_fig}")

if __name__ == "__main__":
    run_real_correlation_analysis()