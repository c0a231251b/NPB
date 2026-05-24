import os
import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic'

def extract_stamen_from_lines(game_data):
    """
    JSONのtext_live内の'lines'から、1番〜9番の最初に登場するスタメン選手を抽出する関数
    """
    stamen_map = {} # {打順(1~9): 選手名}
    
    if 'text_live' not in game_data:
        return stamen_map
        
    # 打順を表す正規表現（例: "1番 キャベッジ", "4番 岡本" などをキャプチャ）
    order_pattern = re.compile(r'^([1-9])番\s+(.+)$')
    
    for live_item in game_data['text_live']:
        plays = live_item.get('plays', [])
        for play in plays:
            lines = play.get('lines', [])
            for line in lines:
                line_clean = line.strip()
                # 正規表現で「X番 選手名」のパターンにマッチするか確認
                match = order_pattern.match(line_clean)
                if match:
                    order_num = int(match.group(1)) # 1〜9
                    player_name = match.group(2).split()[0] # 後ろに続く「三振」などのテキストを排除し名前だけ取得
                    
                    # 各打順で「最初に登場した選手」のみをスタメンとして記録
                    if order_num not in stamen_map:
                        # 特殊なアクション（代打など）ではなく、試合の極めて初期（1回、2回など）の登場かチェック
                        # 基本的に各試合でその打順の最初に見つかった選手をスタメンとみなす
                        stamen_map[order_num] = player_name
                        
        # 1番から9番まで全員揃ったら、それ以降の代打・交代ノイズを防ぐため走査を終了
        if len(stamen_map) == 9:
            break
            
    return stamen_map

def run_real_aggregation():
    # --- 1. データの読み込み ---
    try:
        df_clusters = pd.read_csv("extracted_player_clusters.csv")
    except FileNotFoundError:
        print("Error: 'extracted_player_clusters.csv' が見つかりません。")
        return

    df_clusters['Player_Clean'] = df_clusters['Player_Name'].str.replace(" ", "").str.replace(" ", "")

    # 各選手の各打順での出場回数カウンター { 選手名: {1: 0, 2: 0, ... 9: 0} }
    player_order_counts = {}

    # --- 2. game_data_2025フォルダ内のJSONファイルをループ ---
    json_files = glob.glob("game_data_2025/*.json")
    if not json_files:
        print("Error: 'game_data_2025' フォルダにJSONファイルがありません。")
        return

    print(f" {len(json_files)} 個のJSONファイルから、linesの記述を元に真のスタメンを抽出中...")

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                game_data = json.load(f)
            except Exception:
                continue

            # ハルタさん提案のロジックでこの試合のスタメン（1〜9番）を抽出
            game_stamen = extract_stamen_from_lines(game_data)
            
            # 抽出したスタメンを全体のカウンターに加算
            for order_num, p_name in game_stamen.items():
                name_clean = p_name.replace(" ", "").replace(" ", "")
                
                if name_clean not in player_order_counts:
                    player_order_counts[name_clean] = {num: 0 for num in range(1, 10)}
                player_order_counts[name_clean][order_num] += 1

    # --- 3. 各選手の「最頻打順」を確定 ---
    player_main_order = {}
    for p_name, counts in player_order_counts.items():
        max_order = max(counts, key=counts.get)
        if counts[max_order] > 0:
            player_main_order[p_name] = f"{max_order}番打者"

    # クラスターデータに実際の正しい打順をマッピング
    df_clusters['Real_Order'] = df_clusters['Player_Clean'].map(player_main_order)

    # データのクリーニング（打者限定で有効、かつJSONのプレイログにスタメン登場した選手）
    df_analysis = df_clusters.dropna(subset=['Real_Order', 'Cluster_ID']).copy()
    df_analysis['Cluster_ID'] = df_analysis['Cluster_ID'].astype(int)

    # --- 4. クロス集計の実行 ---
    order_labels = [f"{i}番打者" for i in range(1, 10)]
    cross_tab = pd.crosstab(df_analysis['Cluster_ID'], df_analysis['Real_Order'])
    cross_tab = cross_tab.reindex(columns=order_labels).fillna(0).astype(int)
    
    cross_tab_pct = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

    print("\n" + "="*60)
    print(" 【打者限定・正しいスタメン抽出版】クラスターと実際の主戦打順の相関行列 (人数)")
    print("="*60)
    print(cross_tab)
    print("\n" + "="*60)
    print(" クラスター内における実際の起用打順の構成比率 (%)")
    print("="*60)
    print(cross_tab_pct.round(1))

    # --- 5. ヒートマップの保存 ---
    plt.figure(figsize=(12, 7))
    sns.heatmap(
        cross_tab_pct, 
        annot=cross_tab, 
        fmt="d", 
        cmap="YlGnBu", 
        linewidths=0.8,
        cbar_kws={'label': 'クラスター内での起用割合 (%)'}
    )
    plt.title("打者限定潜在空間クラスター（K=3）と実際のスタメン打順（lines抽出）の相関関係", fontsize=13, fontweight='bold')
    plt.xlabel("JSONのプレイログ（lines）から集計した最も頻度の高かった実際のスタメン打順", fontsize=11)
    plt.ylabel("モデルが算出した打者クラスターID (0〜2)", fontsize=11)
    plt.tight_layout()
    
    output_fig = "cluster_real_batting_order_heatmap.png"
    plt.savefig(output_fig)
    plt.close()
    print(f"\n修正完了！本当の打順相関ヒートマップ画像を保存しました: {output_fig}")

if __name__ == "__main__":
    run_real_aggregation()