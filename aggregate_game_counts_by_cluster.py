import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic'

def run_game_count_analysis():
    # ==================================================================
    # 【手動設定エリア】解析したいファイルをここで切り替えてください
    # ==================================================================
    #cluster_file = "extracted_player_clusters.csv"      # K=3 の場合はこちら
    cluster_file = "extracted_player_clusters_K4.csv"  # K=4 の場合はこちら
    
    lineup_file = "all_games_starting_lineups.csv"
    # ==================================================================
    
    if not os.path.exists(cluster_file):
        print(f"Error: '{cluster_file}' が見つかりません。")
        return
    if not os.path.exists(lineup_file):
        print(f"Error: '{lineup_file}' が見つかりません。")
        return

    # クラスターデータ (Player_ID, Player_Name, Cluster_ID)
    df_clusters = pd.read_csv(cluster_file)
    
    # エラー対策: Cluster_ID が欠損(NaN)している行を完全に除外する
    df_clusters = df_clusters.dropna(subset=['Cluster_ID']).copy()
    
    # クラスターIDを確実に整数型に変換
    df_clusters['Cluster_ID'] = df_clusters['Cluster_ID'].astype(int)
    
    # 文字列のスペース排除
    df_clusters['Player_Clean'] = df_clusters['Player_Name'].str.replace(" ", "", regex=False).str.replace("　", "", regex=False)
    
    # 選手名からクラスターIDを引く辞書を作成
    cluster_map = dict(zip(df_clusters['Player_Clean'], df_clusters['Cluster_ID']))
    
    # 表記ブレ（フルネーム vs 姓のみ）に対応するマッピング関数
    def get_cluster_id(p_name):
        if pd.isna(p_name) or p_name == "Unknown" or not isinstance(p_name, str):
            return None
        name_clean = p_name.replace(" ", "").replace("　", "")
        
        # 1. 完全一致で検索
        if name_clean in cluster_map:
            return cluster_map[name_clean]
        # 2. 部分一致・前方一致で検索（「岡本」が「岡本和真」のIDにヒットする安全策）
        for c_name, c_id in cluster_map.items():
            if name_clean in c_name or c_name in name_clean:
                return c_id
        return None

    # スターティングラインナップCSV
    df_lineups = pd.read_csv(lineup_file)

    print(f"【使用ファイル: {cluster_file}】")
    print("選手単位ではなく【出場試合（打席）数ベース】でのダイ計数集計を開始します...")

    # --- 2. 試合数ベースのクロス集計マトリクスの初期化 ---
    order_labels = [f"{i}番打者" for i in range(1, 10)]
    
    # 存在するユニークなクラスターIDのリストを取得
    unique_clusters = sorted(df_clusters['Cluster_ID'].unique())
    
    # カウント用のデータフレームを作成（初期値はすべて0）
    game_cross_tab = pd.DataFrame(0, index=unique_clusters, columns=order_labels)

    # 総集計カウンター
    total_matched_slots = 0
    total_missing_slots = 0

    # --- 3. 1716個のリアル打線を1セルずつ走査してダイレクトに加算 ---
    order_columns = [f"Order_{i}" for i in range(1, 10)]
    
    for _, row in df_lineups.iterrows():
        for order_idx, col_name in enumerate(order_columns, 1):
            player_name = row[col_name]
            
            # クラスターIDの取得
            c_id = get_cluster_id(player_name)
            
            # c_id が正常な数値（NaNでない）の場合のみ加算処理を行う
            if c_id is not None and not pd.isna(c_id):
                order_label = f"{order_idx}番打者"
                game_cross_tab.loc[int(c_id), order_label] += 1
                total_matched_slots += 1
            else:
                total_missing_slots += 1

    # --- 4. 比率（%）ベースのマトリクスを算出 ---
    game_cross_tab_pct = game_cross_tab.div(game_cross_tab.sum(axis=1), axis=0) * 100

    # --- 5. コンソールへの結果出力 ---
    print("\n" + "="*75)
    print(" 【試合数ベース・柔軟版】相関行列 (総スタメン起用回数)")
    print("="*75)
    print(game_cross_tab)
    print("\n" + "="*75)
    print(" 各打者クラスター内における実際の起用試合数の構成比率 (%)")
    print("="*75)
    print(game_cross_tab_pct.round(1).to_string())
    print("="*75)
    print(f"クラスター紐付けに成功した総スタメン枠数: {total_matched_slots} 枠")
    print(f"登録外ノイズ（投手やスタメン頻度の極めて低い選手など）: {total_missing_slots} 枠")

    # --- 6. 新しい相関ヒートマップの生成と保存 ---
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        game_cross_tab_pct, 
        annot=game_cross_tab, 
        fmt="d", 
        cmap="YlGnBu", 
        linewidths=1.0,
        cbar_kws={'label': 'クラスター内での起用試合数割合 (%)'}
    )
    
    k_num = len(unique_clusters)
    plt.title(f"FMモデル打者クラスターと2025年実際の【起用試合数】の数理的相関関係 (K={k_num})", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("2025年シーズン全試合の全打席からカウントした『実際の起用打順』", fontsize=12, labelpad=10)
    plt.ylabel("FM潜在空間（V）から算出した打者クラスターID", fontsize=12, labelpad=10)
    plt.tight_layout()
    
    output_fig = f"cluster_real_game_count_heatmap_K{k_num}.png"
    plt.savefig(output_fig, dpi=300)
    plt.close()
    
    print(f"\n成功しました！試合数カウント版ヒートマップ画像を保存しました: {output_fig}")

if __name__ == "__main__":
    run_game_count_analysis()