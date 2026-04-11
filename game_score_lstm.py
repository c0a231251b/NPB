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
    "model_type": "Standard LSTM (Dynamic Stats)",
    "input_size": 3,      # [打率, 本塁打, OPS]
    "hidden_size": 64,
    "num_epochs": 100,
    "learning_rate": 0.001,
    "json_dir": "game_data_2025",
    "initial_stats": "initial_stats_2024.csv",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# -----------------------------------------------
# 成績更新クラス & モデル定義 (LSTM)
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
        return [avg, s['HR'], obp + slg]

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

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(StandardLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# -----------------------------------------------
# 学習とログ出力
# -----------------------------------------------
def main():
    tracker = DynamicStatsTracker(PARAMS["initial_stats"])
    paths = sorted(glob.glob(os.path.join(PARAMS["json_dir"], "*.json")))
    
    X, y = [], []
    print(f"時系列解析中... ({len(paths)}試合)")
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
                            if lu['team'].replace("ＤｅＮＡ", "DeNA") == t_name: lineup = lu['players']
                if len(lineup) == 9:
                    X.append([tracker.get_features(t_name, p) for p in lineup])
                    y.append(score)
            for entry in data['text_live']:
                if 'plays' in entry:
                    for play in entry['plays']:
                        info = play['lines'][0]; res = play['lines'][1]
                        p_name = info.split(" ")[1]
                        for team_data in data['scoreboard']:
                            tracker.update(team_data['team'].replace("ＤｅＮＡ", "DeNA"), p_name, res)

    X_arr = np.array(X)
    s, seq, f = X_arr.shape
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr.reshape(-1, f)).reshape(s, seq, f)
    
    X_train = torch.FloatTensor(X_scaled).to(PARAMS["device"])
    y_train = torch.FloatTensor(y).view(-1, 1).to(PARAMS["device"])

    model = StandardLSTM(PARAMS["input_size"], PARAMS["hidden_size"]).to(PARAMS["device"])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])

    # 実行結果を保存するためのリスト
    results_log = []
    results_log.append("========== 学習設定 (Parameters) ==========")
    for k, v in PARAMS.items():
        results_log.append(f"{k}: {v}")
    results_log.append(f"Sample Count: {len(X_train)}")
    results_log.append("==========================================\n")

    print("\n".join(results_log))

    print("学習開始...")
    for epoch in range(PARAMS["num_epochs"]):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            msg = f"Epoch {epoch+1}/{PARAMS['num_epochs']}, MSE: {loss.item():.4f}"
            print(msg)
            results_log.append(msg)

    # 最終結果の保存
    final_filename = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(final_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(results_log))
    print(f"\n実行結果を {final_filename} に保存しました。")

if __name__ == "__main__":
    main()