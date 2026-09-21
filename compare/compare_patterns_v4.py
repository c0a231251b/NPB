import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import glob
import os
import json

# -----------------------------------------------
# 1. 特徴量管理クラス (パターン切り替え対応)
# -----------------------------------------------
class BaseballFeatureManager:
    def __init__(self, batter_csv, pitcher_csv):
        # 分類済み打者データ
        self.b_df = pd.read_csv(batter_csv)
        hand_map = {"右打": 0, "左打": 1, "両打": 2}
        self.b_df['Hand'] = self.b_df['Hand'].map(hand_map).fillna(0)
        self.batter_stats = self.b_df.set_index(['team', 'name']).to_dict('index')

        # 投手データ (球種グループ化済み)
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

    def get_batter_vector(self, team, name, pattern="A"):
        s = self.batter_stats.get((team, name), {
            'Hand':0, 'AVG':0, 'HR':0, 'SLG':0,
            'type_power':0, 'type_avg':0, 'type_speed':0, 'type_eye':0, 'type_all':0
        })
        if pattern == "A":
            # パターンA: Hand, AVG, HR, SLG (4次元)
            return [s['Hand'], s['AVG'], s['HR'], s['SLG']]
        else:
            # パターンB: Hand + 5つのタイプ (6次元)
            return [s['Hand'], s['type_power'], s['type_avg'], s['type_speed'], s['type_eye'], s['type_all']]

    def get_pitcher_vector(self, team, name):
        p = self.pitcher_stats.get((team, name))
        return [p[col] for col in self.target_p_cols] if p else self.p_default

# -----------------------------------------------
# 2. データのロード
# -----------------------------------------------
def load_dataset(pattern, json_dir, manager):
    paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    X, y = [], []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            starters = {}
            for entry in data.get('text_live', []):
                if 'pregame' in entry and 'pitchers' in entry['pregame']:
                    for p in entry['pregame']['pitchers']:
                        t = p['team'].replace("ＤｅＮＡ", "DeNA")
                        starters[t] = p['name']
            for team_data in data['scoreboard']:
                t_name = team_data['team'].replace("ＤｅＮＡ", "DeNA")
                opp_team = [t['team'].replace("ＤｅＮＡ", "DeNA") for t in data['scoreboard'] if t['team'].replace("ＤｅＮＡ", "DeNA") != t_name][0]
                starter_name = starters.get(opp_team, "不明")
                lineup = []
                for entry in data.get('text_live', []):
                    if 'pregame' in entry:
                        for lu in entry['pregame']['lineups']:
                            if lu['team'].replace("ＤｅＮＡ", "DeNA") == t_name: lineup = lu['players']
                if len(lineup) == 9:
                    p_v = manager.get_pitcher_vector(opp_team, starter_name)
                    b_v = []
                    for b_name in lineup:
                        b_v.extend(manager.get_batter_vector(t_name, b_name, pattern))
                    X.append(b_v + p_v)
                    y.append(int(team_data['R']))
    return np.array(X), np.array(y)

# -----------------------------------------------
# 3. 比較実行
# -----------------------------------------------
def main():
    manager = BaseballFeatureManager("classified_batter_stats.csv", "pitcher_stats_2024_all.csv")
    json_dir = "game_data_2025"
    
    for pattern in ["A", "B"]:
        print(f"\n--- Pattern {pattern} 解析中 ---")
        X, y = load_dataset(pattern, json_dir, manager)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)
        
        # Lasso回帰で評価
        model = Lasso(alpha=0.05) # alphaを少し調整
        model.fit(X_train_s, y_train)
        pred = model.predict(X_val_s)
        mse = mean_squared_error(y_val, pred)
        active = np.sum(model.coef_ != 0)
        
        label = "Raw Stats" if pattern == "A" else "Batter Types"
        print(f"[{label}]")
        print(f"  特徴量次元数: {X.shape[1]}")
        print(f"  有効な特徴量: {active} / {X.shape[1]}")
        print(f"  Validation MSE: {mse:.4f}")

if __name__ == "__main__":
    main()