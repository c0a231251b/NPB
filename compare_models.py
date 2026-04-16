import os
import json
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# -----------------------------------------------
# 1. 成績更新クラス (動的な特徴量生成)
# -----------------------------------------------
class DynamicStatsTracker:
    def __init__(self, initial_csv):
        df = pd.read_csv(initial_csv)
        self.stats = df.set_index(['team', 'name']).to_dict('index')
        for key in self.stats:
            for col in ['AB', 'H', 'TB', 'BB', 'HBP', 'SF', 'HR']:
                if col not in self.stats[key]: self.stats[key][col] = 0

    def get_features(self, team, name):
        s = self.stats.get((team, name), {'AB':0, 'H':0, 'TB':0, 'BB':0, 'HBP':0, 'SF':0, 'HR':0})
        ab = max(s['AB'], 1)
        avg = s['H'] / ab
        obp = (s['H'] + s['BB'] + s['HBP']) / max((s['AB'] + s['BB'] + s['HBP'] + s['SF']), 1)
        slg = s['TB'] / ab
        return [avg, s['HR'], slg, obp + slg]

    def update(self, team, name, result_text):
        key = (team, name)
        if key not in self.stats: self.stats[key] = {'AB':0, 'H':0, 'TB':0, 'BB':0, 'HBP':0, 'SF':0, 'HR':0}
        if any(x in result_text for x in ["安", "二", "三", "本"]):
            self.stats[key]['H'] += 1; self.stats[key]['AB'] += 1
            if "二" in result_text: self.stats[key]['TB'] += 2
            elif "三" in result_text: self.stats[key]['TB'] += 3
            elif "本" in result_text: self.stats[key]['TB'] += 4; self.stats[key]['HR'] += 1
            else: self.stats[key]['TB'] += 1
        elif any(x in result_text for x in ["四球", "死球", "敬遠"]):
            if "四球" in result_text or "敬遠" in result_text: self.stats[key]['BB'] += 1
            else: self.stats[key]['HBP'] += 1
        elif "犠飛" in result_text: self.stats[key]['SF'] += 1
        elif any(x in result_text for x in ["ゴ", "飛", "振", "直", "斜", "失", "野選"]):
            self.stats[key]['AB'] += 1

# -----------------------------------------------
# 2. データのロードと平坦化 (Flatten)
# -----------------------------------------------
def load_and_flatten_data(json_dir, initial_csv):
    tracker = DynamicStatsTracker(initial_csv)
    paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    X_flattened, y = [], []
    
    print(f"データ解析中... ({len(paths)}試合)")
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for team_data in data['scoreboard']:
                t_name = team_data['team'].replace("ＤｅＮＡ", "DeNA")
                score = int(team_data['R'])
                
                lineup = []
                for entry in data['text_live']:
                    if 'pregame' in entry:
                        for lu in entry['pregame']['lineups']:
                            if lu['team'].replace("ＤｅＮＡ", "DeNA") == t_name:
                                lineup = lu['players']
                
                if len(lineup) == 9:
                    game_features = []
                    for p_name in lineup:
                        game_features.extend(tracker.get_features(t_name, p_name))
                    X_flattened.append(game_features)
                    y.append(score)
            
            for entry in data['text_live']:
                if 'plays' in entry:
                    for play in entry['plays']:
                        info = play['lines'][0]; res = play['lines'][1]
                        p_name = info.split(" ")[1]
                        for team_data in data['scoreboard']:
                            tracker.update(team_data['team'].replace("ＤｅＮＡ", "DeNA"), p_name, res)
                            
    return np.array(X_flattened), np.array(y)

# -----------------------------------------------
# 3. メイン処理 (学習・評価)
# -----------------------------------------------
def main():
    json_dir = "game_data_2025"
    initial_csv = "initial_stats_2024.csv"
    
    # データの読み出し
    X_raw, y = load_and_flatten_data(json_dir, initial_csv)
    
    # --- ホールドアウト法 (Train/Test Split) ---
    # データを 8:2 に分割（シャッフルして未知のデータへの汎化性能を測る）
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=42
    )
    
    # --- 標準化 (Standardization) ---
    # ※学習用データのみで fit させ、テストデータには適用のみ行うのが作法
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # --- 重回帰分析 (Linear Regression) ---
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled) # 未知データで予測
    lr_mse = mean_squared_error(y_test, lr_pred)
    
    # --- ランダムフォレスト (Random Forest) ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_raw, y_train) # RFは標準化なしの生データでOK
    rf_pred = rf_model.predict(X_test_raw)    # 未知データで予測
    rf_mse = mean_squared_error(y_test, rf_pred)
    
    print("\n" + "="*45)
    print(f"解析サンプル総数: {len(X_raw)}")
    print(f"学習データ数: {len(X_train_raw)} / テストデータ数: {len(X_test_raw)}")
    print("="*45)
    print(f"常に平均予測 (Baseline): 6.900")
    print(f"標準LSTM (前回結果): 6.880")
    print(f"重回帰分析 (Linear - Test Score): {lr_mse:.4f}")
    print(f"ランダムフォレスト (RF - Test Score): {rf_mse:.4f}")
    print("="*45)

if __name__ == "__main__":
    main()