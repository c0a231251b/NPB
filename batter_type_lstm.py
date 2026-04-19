import os
import json
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------------------------
# 設定（パラメータ）
# -----------------------------------------------
PARAMS = {
    "model_type": "5-Type Batter LSTM",
    "input_size": 13,      # 打者6 (Hand + 5Types) + 投手7
    "hidden_size": 32,     # 過学習を防ぐため、あえて少し小さめに設定
    "num_epochs": 150,
    "learning_rate": 0.0001,
    "batch_size": 32,
    "json_dir": "game_data_2025",
    "batter_stats": "classified_batter_stats.csv", # 分類済みCSVを使用
    "pitcher_stats": "pitcher_stats_2024_all.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# -----------------------------------------------
# 特徴量管理クラス
# -----------------------------------------------
class BaseballFeatureManager:
    def __init__(self, batter_csv, pitcher_csv):
        # 打者データ (分類済みフラグを使用)
        b_df = pd.read_csv(batter_csv)
        hand_map = {"右打": 0, "左打": 1, "両打": 2}
        b_df['Hand'] = b_df['Hand'].map(hand_map).fillna(0)
        self.batter_stats = b_df.set_index(['team', 'name']).to_dict('index')

        # 投手データ (球種グループ化)
        p_df = pd.read_csv(pitcher_csv).drop_duplicates(subset=['team', 'name'])
        p_hand_map = {"右投": 0, "左投": 1}
        p_df['hand'] = p_df['hand'].map(p_hand_map).fillna(0).astype(int)
        
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

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=2, dropout=0.3)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def main():
    manager = BaseballFeatureManager(PARAMS["batter_stats"], PARAMS["pitcher_stats"])
    paths = sorted(glob.glob(os.path.join(PARAMS["json_dir"], "*.json")))
    X, y = [], []

    # データのロード処理
    print(f"データをロード中... ({len(paths)}ファイル)")
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
                    seq = [manager.get_batter_vector(t_name, name) + p_v for name in lineup]
                    X.append(seq)
                    y.append(int(team_data['R']))

    X_arr = np.array(X)
    y_arr = np.array(y).reshape(-1, 1)

    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_arr, y_arr, test_size=0.2, random_state=42, shuffle=True)

    # 標準化
    s, seq, f = X_train_raw.shape
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw.reshape(-1, f)).reshape(s, seq, f)
    X_val_s = scaler.transform(X_val_raw.reshape(-1, f)).reshape(X_val_raw.shape[0], seq, f)

    train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_s).to(PARAMS["device"]), torch.FloatTensor(y_train_raw).to(PARAMS["device"])), batch_size=PARAMS["batch_size"], shuffle=True)
    X_val_tensor = torch.FloatTensor(X_val_s).to(PARAMS["device"])
    y_val_tensor = torch.FloatTensor(y_val_raw).to(PARAMS["device"])

    model = StandardLSTM(PARAMS["input_size"], PARAMS["hidden_size"]).to(PARAMS["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])

    train_losses, val_losses = [], []
    print("学習開始...")
    for epoch in range(PARAMS["num_epochs"]):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)
        
        model.eval()
        with torch.no_grad():
            avg_val = criterion(model(X_val_tensor), y_val_tensor).item()
            val_losses.append(avg_val)

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{PARAMS['num_epochs']} | Train MSE: {avg_train:.4f} | Val MSE: {avg_val:.4f}")

    # グラフ化
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val', linestyle='--')
    plt.title('Learning Curve (5-Type Classification LSTM)')
    plt.legend(); plt.grid(True)
    plt.savefig(f"learning_curve_type_lstm_{datetime.now().strftime('%H%M%S')}.png")
    print(f"\n最終検証MSE: {val_losses[-1]:.4f}")

if __name__ == "__main__":
    main()