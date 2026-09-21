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

# -----------------------------------------------
# 1. 特徴量管理クラス (投手・打者統合)
# -----------------------------------------------
class BaseballFeatureManager:
    def __init__(self, batter_csv, pitcher_csv):
        # 打者データの読み込み
        b_df = pd.read_csv(batter_csv)
        b_hand_map = {"右打": 0, "左打": 1, "両打": 2}
        b_df['Hand'] = b_df['Hand'].map(b_hand_map).fillna(0)
        self.batter_stats = b_df.set_index(['team', 'name']).to_dict('index')

        # 投手データの読み込み
        p_df = pd.read_csv(pitcher_csv)
        p_df = p_df.drop_duplicates(subset=['team', 'name'])
        
        # 利き手の数値化 (右投:0, 左投:1)
        p_hand_map = {"右投": 0, "左投": 1}
        if p_df['hand'].dtype == object:
            p_df['hand'] = p_df['hand'].map(p_hand_map)
        
        p_df['hand'] = p_df['hand'].fillna(0).astype(int)
        self.pitcher_stats = p_df.set_index(['team', 'name']).to_dict('index')
        
        # 投手用特徴量カラムの抽出 (team, name 以外)
        self.p_cols = [c for c in p_df.columns if c not in ['team', 'name']]
        self.p_default = p_df[self.p_cols].mean().tolist()

    def get_batter_vector(self, team, name):
        s = self.batter_stats.get((team, name), {'Hand':0, 'AB':0, 'H':0, 'TB':0, 'BB':0, 'HBP':0, 'SF':0, 'HR':0})
        ab = max(s['AB'], 1)
        avg = s['H'] / ab
        obp = (s['H'] + s['BB'] + s['HBP']) / max((s['AB'] + s['BB'] + s['HBP'] + s['SF']), 1)
        slg = s['TB'] / ab
        return [s['Hand'], avg, s['HR'], slg, obp + slg]

    def get_pitcher_vector(self, team, name):
        p = self.pitcher_stats.get((team, name))
        if p:
            return [p[col] for col in self.p_cols]
        return self.p_default

# -----------------------------------------------
# 2. データのロードと平坦化 (69次元ベクトル化)
# -----------------------------------------------
def load_flattened_dataset(json_dir, batter_csv, pitcher_csv):
    manager = BaseballFeatureManager(batter_csv, pitcher_csv)
    paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    X, y = [], []
    
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
            # 先発投手の特定
            starters = {}
            for entry in data.get('text_live', []):
                if 'pregame' in entry and 'pitchers' in entry['pregame']:
                    for p in entry['pregame']['pitchers']:
                        t = p['team'].replace("ＤｅＮＡ", "DeNA")
                        starters[t] = p['name']

            for team_data in data['scoreboard']:
                t_name = team_data['team'].replace("ＤｅＮＡ", "DeNA")
                score = int(team_data['R'])
                
                # 相手チームと相手投手の特定
                teams_in_game = [t['team'].replace("ＤｅＮＡ", "DeNA") for t in data['scoreboard']]
                opp_team = [t for t in teams_in_game if t != t_name][0]
                starter_name = starters.get(opp_team, "不明")

                # 打順の取得
                lineup = []
                for entry in data.get('text_live', []):
                    if 'pregame' in entry:
                        for lu in entry['pregame']['lineups']:
                            if lu['team'].replace("ＤｅＮＡ", "DeNA") == t_name: 
                                lineup = lu['players']
                
                if len(lineup) == 9:
                    # 投手24次元
                    p_vector = manager.get_pitcher_vector(opp_team, starter_name)
                    
                    # 打者45次元 (9人 × 5指標)
                    b_vectors = []
                    for b_name in lineup:
                        b_vectors.extend(manager.get_batter_vector(t_name, b_name))
                    
                    # 統合して69次元
                    X.append(b_vectors + p_vector)
                    y.append(score)
                            
    return np.array(X), np.array(y)

# -----------------------------------------------
# 3. メイン処理 (学習と比較)
# -----------------------------------------------
def main():
    json_dir = "game_data_2025"
    batter_csv = "initial_stats_2024.csv"
    pitcher_csv = "pitcher_stats_2024_all.csv"
    
    # データのロード
    print("データセット作成中...")
    X, y = load_flattened_dataset(json_dir, batter_csv, pitcher_csv)
    
    # 分割 (80% 学習, 20% 検証)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 標準化 (重回帰には必須)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # --- 1. 重回帰分析 (Linear Regression) ---
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_val_scaled)
    lr_mse = mean_squared_error(y_val, lr_pred)
    
    # --- 2. ランダムフォレスト (Random Forest) ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train) # RFは標準化なしでも可
    rf_pred = rf_model.predict(X_val)
    rf_mse = mean_squared_error(y_val, rf_pred)
    
    # --- 3. 平均予測 (Baseline) ---
    baseline_pred = np.full_like(y_val, np.mean(y_train))
    baseline_mse = mean_squared_error(y_val, baseline_pred)
    
    print("\n" + "="*45)
    print(f"解析サンプル総数: {len(X)}")
    print(f"特徴量次元数: {X.shape[1]} (打者45 + 投手24)")
    print("="*45)
    print(f"平均予測 (Baseline) MSE: {baseline_mse:.4f}")
    print(f"重回帰分析 (Linear) MSE: {lr_mse:.4f}")
    print(f"ランダムフォレスト (RF) MSE: {rf_mse:.4f}")
    print("="*45)

if __name__ == "__main__":
    main()