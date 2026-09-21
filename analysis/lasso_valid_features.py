"""
有効な特徴量を選択するためのコード例。Lasso回帰を使用して、特徴量の重要性を評価し、次元削減を行います。
"""
import os
import json
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

# -----------------------------------------------
# 1. 特徴量管理クラス (次元削減実装版)
# -----------------------------------------------
class BaseballFeatureManager:
    def __init__(self, batter_csv, pitcher_csv):
        # 打者データ (OPSを除外)
        b_df = pd.read_csv(batter_csv)
        b_hand_map = {"右打": 0, "左打": 1, "両打": 2}
        b_df['Hand'] = b_df['Hand'].map(b_hand_map).fillna(0)
        self.batter_stats = b_df.set_index(['team', 'name']).to_dict('index')

        # 投手データ
        p_df = pd.read_csv(pitcher_csv)
        p_df = p_df.drop_duplicates(subset=['team', 'name'])
        
        # 利き手の数値化
        p_hand_map = {"右投": 0, "左投": 1}
        if p_df['hand'].dtype == object:
            p_df['hand'] = p_df['hand'].map(p_hand_map)
        p_df['hand'] = p_df['hand'].fillna(0).astype(int)

        # --- 球種のグループ化 ---
        fast_balls = ['pitch_ストレート_share', 'pitch_ツーシーム_share', 'pitch_ワンシーム_share']
        breaking_balls = [
            'pitch_スライダー_share', 'pitch_カットボール_share', 'pitch_カーブ_share',
            'pitch_シュート_share', 'pitch_スローカーブ_share', 'pitch_ナックルカーブ_share',
            'pitch_スラーブ_share', 'pitch_スローボール_share', 'pitch_パワーカーブ_share',
            'pitch_高速スライダー_share'
        ]
        falling_balls = [
            'pitch_フォーク_share', 'pitch_チェンジアップ_share', 'pitch_シンカー_share',
            'pitch_スプリット_share', 'pitch_縦スライダー_share', 'pitch_パーム_share',
            'pitch_スクリュー_share'
        ]

        # 既存の列があるか確認しながら加算
        p_df['fast_group'] = p_df[p_df.columns.intersection(fast_balls)].sum(axis=1)
        p_df['breaking_group'] = p_df[p_df.columns.intersection(breaking_balls)].sum(axis=1)
        p_df['falling_group'] = p_df[p_df.columns.intersection(falling_balls)].sum(axis=1)

        # 使用する投手特徴量を7次元に限定
        self.target_p_cols = ['hand', 'ERA', 'K/9', 'HR/9', 'fast_group', 'breaking_group', 'falling_group']
        self.pitcher_stats = p_df.set_index(['team', 'name']).to_dict('index')
        self.p_default = p_df[self.target_p_cols].mean().tolist()

    def get_batter_vector(self, team, name):
        s = self.batter_stats.get((team, name), {'Hand':0, 'AB':0, 'H':0, 'TB':0, 'BB':0, 'HBP':0, 'SF':0, 'HR':0})
        ab = max(s['AB'], 1)
        avg = s['H'] / ab
        slg = s['TB'] / ab
        # [Hand, 打率, 本塁打, 長打率] の4次元 (OPSを除外)
        return [s['Hand'], avg, s['HR'], slg]

    def get_pitcher_vector(self, team, name):
        p = self.pitcher_stats.get((team, name))
        if p:
            return [p[col] for col in self.target_p_cols]
        return self.p_default

# -----------------------------------------------
# 2. データのロードと平坦化 (43次元)
# -----------------------------------------------
def load_flattened_dataset(json_dir, batter_csv, pitcher_csv):
    manager = BaseballFeatureManager(batter_csv, pitcher_csv)
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
                score = int(team_data['R'])
                teams_in_game = [t['team'].replace("ＤｅＮＡ", "DeNA") for t in data['scoreboard']]
                opp_team = [t for t in teams_in_game if t != t_name][0]
                starter_name = starters.get(opp_team, "不明")

                lineup = []
                for entry in data.get('text_live', []):
                    if 'pregame' in entry:
                        for lu in entry['pregame']['lineups']:
                            if lu['team'].replace("ＤｅＮＡ", "DeNA") == t_name: 
                                lineup = lu['players']
                
                if len(lineup) == 9:
                    p_vector = manager.get_pitcher_vector(opp_team, starter_name)
                    b_vectors = []
                    for b_name in lineup:
                        b_vectors.extend(manager.get_batter_vector(t_name, b_name))
                    
                    # 4 * 9 (打者) + 7 (投手) = 43次元
                    X.append(b_vectors + p_vector)
                    y.append(score)
# 例：打者と投手の「力関係」を直接計算して追加する場合
    for b_name in lineup:
        b_vector = manager.get_batter_vector(t_name, b_name) # [Hand, AVG, HR, SLG]
    
        # 打者の打率(index 1) - 投手の防御率(ERA)を逆算した値を混ぜる
        # ※防御率は低い方が良いため、簡易的に (3 - ERA/3) などの差分を取る
        relative_power = b_vector[1] - (manager.pitcher_stats.get((opp_team, starter_name), {'ERA': 4.0})['ERA'] / 10)
    
        b_vectors.extend(b_vector + [relative_power]) # 相対的な指標を1つ追加
                            
        return np.array(X), np.array(y)

def main():
    json_dir = "game_data_2025"
    batter_csv = "initial_stats_2024.csv"
    pitcher_csv = "pitcher_stats_2024_all.csv"
    
    # データのロード
    X, y = load_flattened_dataset(json_dir, batter_csv, pitcher_csv)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # モデルの更新：Lasso（正則化）を導入
    models = {
        "Baseline (Mean)": None,
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.1), # 正則化を追加
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    print("\n" + "="*45)
    print(f"特徴量次元数: {X.shape[1]} (打者36 + 投手7)")
    print("="*45)

    for name, model in models.items():
        if model is None:
            pred = np.full_like(y_val, np.mean(y_train))
        else:
            m = model.fit(X_train_scaled if "Regression" in name else X_train, y_train)
            pred = m.predict(X_val_scaled if "Regression" in name else X_val)
        
        mse = mean_squared_error(y_val, pred)
        print(f"{name:18} MSE: {mse:.4f}")
    
    # Lassoでどの特徴量が生き残ったか（重みがゼロでないか）を確認
    lasso_m = models["Lasso Regression"].fit(X_train_scaled, y_train)
    active_features = np.sum(lasso_m.coef_ != 0)
    print(f"Lassoにより有効と判定された特徴量数: {active_features} / {X.shape[1]}")
    print("="*45)

if __name__ == "__main__":
    main()