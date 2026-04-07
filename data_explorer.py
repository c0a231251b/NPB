import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from stats_2025_train_model import PlayerStatsDB, build_dataset

# ==========================================
# データ分析エクスプローラー
# ==========================================

def analyze_dataset(json_dir, csv_path):
    print("--- データセット診断開始 ---")
    
    # 1. データの読み込み
    dataset, stats_db = build_dataset(json_dir, csv_path)
    
    # 2. 得点の分布を分析
    scores = [d['score'] for d in dataset]
    df_scores = pd.DataFrame(scores, columns=['Score'])
    
    print(f"\n[1. 得点統計]")
    print(df_scores.describe()) # 平均、最小、最大、中央値などを表示
    
    # 得点分布のヒストグラム表示（もしGUI環境なら）
    # plt.hist(scores, bins=range(min(scores), max(scores) + 2), align='left', rwidth=0.8)
    # plt.title("Score Distribution (306 samples)")
    # plt.xlabel("Runs Scored")
    # plt.ylabel("Frequency")
    # plt.show()

    # 3. 選手の「実績あり/なし」比率を調査
    # build_dataset後のstats_dbの状態を点検
    total_players = len(stats_db.db)
    # 2025年CSVを読み込んで、元からいた人数を確認
    initial_df = pd.read_csv(csv_path)
    initial_names = set(initial_df['name'].unique())
    
    new_players = [name for name in stats_db.db if name not in initial_names]
    
    print(f"\n[2. 選手データのカバー率]")
    print(f"総登録選手数: {total_players}")
    print(f"2025年実績あり: {len(initial_names)} 名")
    print(f"2026年新規登場（リーグ平均スタート）: {len(new_players)} 名")
    if new_players:
        print(f"新規選手例: {', '.join(new_players[:5])}...")

    # 4. 指標のレンジ（正規化の必要性チェック）
    all_vectors = [stats_db.get_vector(name) for name in stats_db.db]
    vectors_np = np.array(all_vectors)
    
    print(f"\n[3. 指標のレンジ (AVG, OBP, SLG, ISO)]")
    print(f"MAX: {vectors_np.max(axis=0)}")
    print(f"MIN: {vectors_np.min(axis=0)}")

if __name__ == "__main__":
    analyze_dataset("game_data", "initial_stats_2025.csv")