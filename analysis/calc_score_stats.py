import os
import json
import glob
import numpy as np

def calculate_stats(json_dir):
    scores = []
    paths = glob.glob(os.path.join(json_dir, "*.json"))
    
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # 各チームのスコアを抽出
                for team_data in data['scoreboard']:
                    scores.append(int(team_data['R']))
            except:
                continue

    if not scores:
        print("スコアデータが見つかりませんでした。")
        return

    avg_score = np.mean(scores)
    std_score = np.std(scores)
    mse_baseline = np.var(scores) # 分散 = 「常に平均値を答えた場合」のMSE

    print(f"--- 2025年度NPB得点統計 (解析試合数: {len(paths)}) ---")
    print(f"平均得点: {avg_score:.3f}")
    print(f"標準偏差 (σ): {std_score:.3f}")
    print(f"分散 (常に平均を予測した場合のMSE): {mse_baseline:.3f}")
    
    return std_score

if __name__ == "__main__":
    calculate_stats("game_data_2025")