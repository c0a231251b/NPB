import os
import json
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# -----------------------------------------------
# 設定（パラメータ）
# -----------------------------------------------
PARAMS = {
    "model_type": "Pitcher-Aware LSTM (Hand-Encoded)",
    "input_size": 29,     # [打者5: Hand, AVG, HR, SLG, OPS] + [投手24: hand, ERA, K/9, ...]
    "hidden_size": 128, 
    "num_epochs": 150,
    "learning_rate": 0.0005,
    "json_dir": "game_data_2025",
    "batter_stats": "initial_stats_2024.csv",
    "pitcher_stats": "pitcher_stats_2024_all.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# -----------------------------------------------
# 特徴量管理クラス
# -----------------------------------------------
class BaseballFeatureManager:
    def __init__(self, batter_csv, pitcher_csv):
        # 1. 打者データ (team, name) のコンビで管理
        b_df = pd.read_csv(batter_csv)
        b_hand_map = {"右打": 0, "左打": 1, "両打": 2}
        b_df['Hand'] = b_df['Hand'].map(b_hand_map).fillna(0)
        self.batter_stats = b_df.set_index(['team', 'name']).to_dict('index')

        # 2. 投手データ (ここを修正：team と name の重複を排除して Index に)
        p_df = pd.read_csv(pitcher_csv)
        
        # 同一チーム・同一名の重複があれば削除（移籍等の重複対策）
        p_df = p_df.drop_duplicates(subset=['team', 'name'])
        
        p_hand_map = {"右投": 0, "左投": 1}
        if p_df['hand'].dtype == object:
            p_df['hand'] = p_df['hand'].map(p_hand_map)
        
        p_df['hand'] = p_df['hand'].fillna(0).astype(int)
        
        # ここを修正：投手も (team, name) をキーにする
        self.pitcher_stats = p_df.set_index(['team', 'name']).to_dict('index')
        
        # デフォルト値の準備
        p_features_only = p_df.drop(columns=['team', 'name'])
        self.p_default = p_features_only.mean().tolist()
        self.p_cols = p_features_only.columns.tolist()

    def get_batter_vector(self, team, name):
        s = self.batter_stats.get((team, name), {'Hand':0, 'AB':0, 'H':0, 'TB':0, 'BB':0, 'HBP':0, 'SF':0, 'HR':0})
        ab = max(s['AB'], 1)
        avg = s['H'] / ab
        obp = (s['H'] + s['BB'] + s['HBP']) / max((s['AB'] + s['BB'] + s['HBP'] + s['SF']), 1)
        slg = s['TB'] / ab
        return [s['Hand'], avg, s['HR'], slg, obp + slg]

    # ここを修正：引数に team を追加
    def get_pitcher_vector(self, team, name):
        p = self.pitcher_stats.get((team, name))
        if p:
            return [p[col] for col in self.p_cols]
        return self.p_default

    def update_batter(self, team, name, result_text):
        key = (team, name)
        if key not in self.batter_stats: 
            self.batter_stats[key] = {'Hand':0, 'AB':0, 'H':0, 'TB':0, 'BB':0, 'HBP':0, 'SF':0, 'HR':0}
        s = self.batter_stats[key]
        if any(x in result_text for x in ["安", "二", "三", "本"]):
            s['H'] += 1; s['AB'] += 1
            if "二" in result_text: s['TB'] += 2
            elif "三" in result_text: s['TB'] += 3
            elif "本" in result_text: s['TB'] += 4; s['HR'] += 1
            else: s['TB'] += 1
        elif any(x in result_text for x in ["四球", "死球", "敬遠"]):
            if "四球" in result_text or "敬遠" in result_text: s['BB'] += 1
            else: s['HBP'] += 1
        elif "犠飛" in result_text: s['SF'] += 1
        elif any(x in result_text for x in ["ゴ", "飛", "振", "直", "斜", "失", "野選"]):
            s['AB'] += 1

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.2) 
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# -----------------------------------------------
# メイン学習処理
# -----------------------------------------------
def main():
    manager = BaseballFeatureManager(PARAMS["batter_stats"], PARAMS["pitcher_stats"])
    paths = sorted(glob.glob(os.path.join(PARAMS["json_dir"], "*.json")))
    
    X, y = [], []
    print(f"投手・打者統合解析中... ({len(paths)}試合)")
    
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
                
                # 1. 相手チームと相手投手を特定
                teams_in_game = [t['team'].replace("ＤｅＮＡ", "DeNA") for t in data['scoreboard']]
                opp_team = [t for t in teams_in_game if t != t_name][0]
                starter_name = starters.get(opp_team, "不明")

                # 2. 打順の取得
                lineup = []
                for entry in data.get('text_live', []):
                    if 'pregame' in entry:
                        for lu in entry['pregame']['lineups']:
                            if lu['team'].replace("ＤｅＮＡ", "DeNA") == t_name: 
                                lineup = lu['players']
                
                # 3. 9人揃っている場合のみ特徴量を生成
                if len(lineup) == 9:
                    # 相手投手のベクトルを取得（team と name の両方を渡す）
                    p_vector = manager.get_pitcher_vector(opp_team, starter_name)
                    
                    combined_sequence = []
                    for b_name in lineup:
                        b_vector = manager.get_batter_vector(t_name, b_name)
                        # 29次元 (打者5 + 投手24) を結合
                        combined_sequence.append(b_vector + p_vector)
                    
                    X.append(combined_sequence)
                    y.append(score)

            # 次回のための更新
            for entry in data['text_live']:
                if 'plays' in entry:
                    for play in entry['plays']:
                        info = play['lines'][0]; res = play['lines'][1]
                        if " " in info:
                            p_name = info.split(" ")[1]
                            manager.update_batter(t_name, p_name, res)

    X_arr = np.array(X)
    y_arr = np.array(y).reshape(-1, 1)

    # 特徴量スケーリング
    s, seq, f = X_arr.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr.reshape(-1, f)).reshape(s, seq, f)
    
    X_train = torch.FloatTensor(X_scaled).to(PARAMS["device"])
    y_train = torch.FloatTensor(y_arr).to(PARAMS["device"])

    model = StandardLSTM(PARAMS["input_size"], PARAMS["hidden_size"]).to(PARAMS["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])

    print(f"学習開始... (Sample: {len(X_train)})")
    for epoch in range(PARAMS["num_epochs"]):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, MSE: {loss.item():.4f}")

    final_filename = f"result_pitcher_aware_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    print(f"\n学習完了。最終MSE: {loss.item():.4f}")
    # 必要に応じてモデルの保存(torch.save)を追加

if __name__ == "__main__":
    main()