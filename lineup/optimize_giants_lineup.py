import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from itertools import permutations
import joblib
import os
import glob
import json

# -----------------------------------------------
# 1. モデルの設計図 (StandardLSTM クラス)
# -----------------------------------------------
class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StandardLSTM, self).__init__()
        # 学習時と同じ構造にする必要があります
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# -----------------------------------------------
# 2. 特徴量管理クラス
# -----------------------------------------------
class BaseballFeatureManager:
    def __init__(self, batter_csv, pitcher_csv):
        b_df = pd.read_csv(batter_csv)
        hand_map = {"右打": 0, "左打": 1, "両打": 2}
        b_df['Hand'] = b_df['Hand'].map(hand_map).fillna(0)
        self.batter_stats = b_df.set_index(['team', 'name']).to_dict('index')

        p_df = pd.read_csv(pitcher_csv).drop_duplicates(subset=['team', 'name'])
        p_hand_map = {"右投": 0, "左投": 1}
        p_df['hand'] = p_df['hand'].map(p_hand_map).fillna(0).astype(int)
        
        # 球種グループ化
        fast = ['pitch_ストレート_share', 'pitch_ツーシーム_share', 'pitch_ワンシーム_share']
        break_b = ['pitch_スライダー_share', 'pitch_カットボール_share', 'pitch_カーブ_share', 'pitch_シュート_share', 'pitch_スローカーブ_share', 'pitch_ナックルカーブ_share', 'pitch_スラーブ_share', 'pitch_スローボール_share', 'pitch_パワーカーブ_share', 'pitch_高速スライダー_share']
        fall = ['pitch_フォーク_share', 'pitch_チェンジアップ_share', 'pitch_シンカー_share', 'pitch_スプリット_share', 'pitch_縦スライダー_share', 'pitch_パーム_share', 'pitch_スクリュー_share']
        
        p_df['fast_g'] = p_df[p_df.columns.intersection(fast)].sum(axis=1)
        p_df['break_g'] = p_df[p_df.columns.intersection(break_b)].sum(axis=1)
        p_df['fall_g'] = p_df[p_df.columns.intersection(fall)].sum(axis=1)

        self.target_p_cols = ['hand', 'ERA', 'K/9', 'HR/9', 'fast_g', 'break_g', 'fall_g']
        self.pitcher_stats = p_df.set_index(['team', 'name']).to_dict('index')
        self.p_default = p_df[self.target_p_cols].mean().tolist()

    def get_batter_vector(self, team, name):
        s = self.batter_stats.get((team, name), {
            'Hand':0, 'type_power':0, 'type_avg':0, 'type_speed':0, 'type_eye':0, 'type_all':0
        })
        return [s['Hand'], s['type_power'], s['type_avg'], s['type_speed'], s['type_eye'], s['type_all']]

    def get_pitcher_vector(self, team, name):
        p = self.pitcher_stats.get((team, name))
        return [p[col] for col in self.target_p_cols] if p else self.p_default

# -----------------------------------------------
# 3. 最適化実行メイン処理
# -----------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ファイルの存在確認
    if not os.path.exists("best_type_lstm.pth") or not os.path.exists("scaler.gz"):
        print("Error: 'best_type_lstm.pth' または 'scaler.gz' が見つかりません。")
        print("先に batter_type_lstm.py を実行して保存してください。")
        return

    # インスタンス生成
    manager = BaseballFeatureManager("classified_batter_stats.csv", "pitcher_stats_2024_all.csv")
    
    # モデルとスケーラーのロード (hidden_size=32)
    model = StandardLSTM(input_size=13, hidden_size=32).to(device)
    model.load_state_dict(torch.load("best_type_lstm.pth", map_location=device))
    model.eval()
    scaler = joblib.load("scaler.gz")

    # シミュレーション対象の選手 (ハルタさんのリスト)
    team = "巨人"
    # position_players = ["吉川尚輝", "ヘルナンデス", "丸佳浩", "岡本和真", "坂本勇人", "若林楽人", "大城卓三", "門脇誠"]
    # 阿部監督がよく使う打順
    position_players = ["若林楽人", "キャベッジ", "吉川尚輝", "岡本和真", "ヘルナンデス", "坂本勇人", "甲斐拓也", "門脇誠"]
    pitcher_batter = "戸郷翔征"
    
    opp_team = "NPB"
    opp_pitcher = "LEAGUE_AVERAGE"
    p_v = manager.get_pitcher_vector(opp_team, opp_pitcher)

    print(f"\n--- {team} 打順最適化シミュレーション (1000試行) ---")
    
    import random
    all_combos = list(permutations(position_players))
    sample_combos = random.sample(all_combos, 1000) # まずは1000回で試走

    results = []
    for combo in sample_combos:
        current_lineup = list(combo) + [pitcher_batter]
        
        seq = []
        for name in current_lineup:
            b_v = manager.get_batter_vector(team, name)
            seq.append(b_v + p_v)
        
        # 入力データをLSTMが受け取れる形 [1, 9, 13] に変換
        input_data = scaler.transform(seq)
        input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device)
        
        with torch.no_grad():
            score = model(input_tensor).item()
        results.append((current_lineup, score))

    # スコア順に並び替え
    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🏆 AIが選んだ『最強の並び』トップ3")
    for i in range(3):
        lineup, score = results[i]
        print(f"\n第{i+1}位 (予測得点: {score:.3f}点)")
        print("  " + " -> ".join([f"{n}" for n in lineup]))

if __name__ == "__main__":
    main()