import json
import glob
import pandas as pd
import numpy as np
from stats_2025_train_model import PlayerStatsDB, build_dataset

# ==========================================
# データ分析エクスプローラー (真の欠損判定版)
# ==========================================

def analyze_dataset(json_dir, csv_path):
    print("--- データセット診断開始 ---")
    
    # 1. データの読み込み
    dataset, stats_db = build_dataset(json_dir, csv_path)
    
    # 2. 得点の分布を分析
    scores = [d['score'] for d in dataset]
    df_scores = pd.DataFrame(scores, columns=['Score'])
    
    print(f"\n[1. 得点統計]")
    print(df_scores.describe())

    # 3. 選手の「実績あり/なし」比率を調査
    # 比較対象として「リーグ平均のベクトル」を算出しておく
    # stats_2025_train_model.py の get_vector 内部ロジックと同じ計算
    s_avg = stats_db.league_avg
    avg_vec = [
        s_avg["H"] / s_avg["AB"], # AVG
        (s_avg["H"] + s_avg["BB"] + s_avg["HBP"]) / (s_avg["AB"] + s_avg["BB"] + s_avg["HBP"] + s_avg["SF"]), # OBP
        s_avg["TB"] / s_avg["AB"], # SLG
        (s_avg["TB"] / s_avg["AB"]) - (s_avg["H"] / s_avg["AB"]) # ISO
    ]

    missing_players = []
    total_players_info = []

    # 2層構造の stats_db.db から全選手をチェック
    for team_name, players in stats_db.db.items():
        for player_name in players.keys():
            # リーグ平均行自体はスキップ
            if player_name == 'LEAGUE_AVERAGE': continue
            
            total_players_info.append((player_name, team_name))
            
            # get_vector で算出したベクトルを取得
            vec = stats_db.get_vector(player_name, team_name)
            
            # 算出したベクトルが、リーグ平均ベクトルと完全に一致するか判定
            # 一致する場合 = 2025年実績が見つからず、補完されたことを意味する
            if np.allclose(vec, avg_vec):
                missing_players.append(f"{team_name}:{player_name}")

    print(f"\n[2. 選手データのカバー率（2025年実績の紐付け結果）]")
    print(f"2026年データに登場した総選手数: {len(total_players_info)} 名")
    print(f"うち、2025年実績が見つからず「リーグ平均」で補完された選手: {len(missing_players)} 名")
    
    coverage = (1 - len(missing_players) / len(total_players_info)) * 100
    print(f"データカバー率（実績あり選手の割合）: {coverage:.2f}%")

    if missing_players:
        print(f"\n[3. 補完された選手（新外国人・新人・育成上がり等）の例]")
        # チームごとに整理して表示すると見やすいが、ここでは上位15名を表示
        print(", ".join(missing_players[:50]) + " ...")

    # 4. 指標のレンジ（ノイズの確認）
    all_vectors = [stats_db.get_vector(p_name, t_name) for p_name, t_name in total_players_info]
    vectors_np = np.array(all_vectors)
    
    print(f"\n[4. 特徴ベクトルのレンジ (AVG, OBP, SLG, ISO)]")
    if len(vectors_np) > 0:
        print(f"最大値: {vectors_np.max(axis=0)}")
        print(f"最小値: {vectors_np.min(axis=0)}")
        print(f"平均値: {vectors_np.mean(axis=0)}")

if __name__ == "__main__":
    # stats_2025_train_model.py と同じディレクトリで実行してください
    analyze_dataset("game_data", "initial_stats_2025.csv")