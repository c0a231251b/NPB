import os
import json
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------
# 1. モデル定義 (双方向LSTM + Attention)
# -----------------------------------------------
class BattingOrderModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BattingOrderModel, self).__init__()
        # 双方向LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        # Attention
        self.attention_weight = nn.Linear(hidden_size * 2, 1)
        # 出力層
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_weight(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output, attn_weights

# -----------------------------------------------
# 2. データの準備と指標の計算
# -----------------------------------------------
def load_data(json_dir, stats_csv):
    df = pd.read_csv(stats_csv)
    
    # 指標の計算
    df['avg'] = df['H'] / df['AB'].replace(0, np.nan)
    df['obp'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF']).replace(0, np.nan)
    df['slg'] = df['TB'] / df['AB'].replace(0, np.nan)
    df['ops'] = df['obp'] + df['slg']
    df = df.fillna(0)
    
    stats_db = df.set_index(['team', 'name']).to_dict('index')
    
    feature_cols = ['avg', 'HR', 'ops']
    X, y = [], []
    paths = glob.glob(os.path.join(json_dir, "*.json"))
    
    print(f"解析中... 対象ファイル数: {len(paths)}")
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for team_data in data['scoreboard']:
                    team_name = team_data['team'].replace("ＤｅＮＡ", "DeNA")
                    score = int(team_data['R'])
                    
                    players = []
                    for entry in data['text_live']:
                        if 'pregame' in entry:
                            for lineup in entry['pregame']['lineups']:
                                if lineup['team'].replace("ＤｅＮＡ", "DeNA") == team_name:
                                    players = lineup['players']
                    
                    if len(players) == 9:
                        lineup_features = []
                        for p in players:
                            s = stats_db.get((team_name, p), {'avg':0.240, 'HR':5, 'ops':0.650})
                            lineup_features.append([s[c] for c in feature_cols])
                        
                        X.append(lineup_features)
                        y.append(score)
            except: continue
    
    return np.array(X), np.array(y)

def prepare_tensors(X, y):
    scaler = StandardScaler()
    s, seq, f = X.shape
    X_reshaped = X.reshape(-1, f)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_final = X_scaled.reshape(s, seq, f)
    return torch.FloatTensor(X_final), torch.FloatTensor(y).view(-1, 1), scaler

# -----------------------------------------------
# 3. 学習
# -----------------------------------------------
def train():
    JSON_DIR = "game_data_2025"
    STATS_CSV = "initial_stats_2024.csv"
    
    X_raw, y_raw = load_data(JSON_DIR, STATS_CSV)
    X_train, y_train, scaler = prepare_tensors(X_raw, y_raw)
    
    # 【修正点】deviceとcriterionの定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BattingOrderModel(input_size=3, hidden_size=128).to(device)
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5)

    print(f"学習開始 (目標 MSE 4.2 | サンプル数: {len(X_train)})")
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        outputs, _ = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # 【修正点】スケジューラをループ内に。item()で警告回避
        scheduler.step(loss.item())
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/300], MSE: {loss.item():.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')

    return model, scaler

# -----------------------------------------------
# 4. 可視化
# -----------------------------------------------
def visualize_results(model):
    device = next(model.parameters()).device
    model.eval()
    # 巨人軍テスト
    dummy_input = torch.randn(1, 9, 3).to(device) 
    with torch.no_grad():
        _, attn = model(dummy_input)
    
    weights = attn.squeeze().cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 10), weights, color='purple', alpha=0.7)
    plt.title("AI Attention Weights - Final Model (Giants Test)")
    plt.xlabel("Batting Order")
    plt.ylabel("Importance")
    plt.xticks(range(1, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig("attention_result.png")
    print("Attentionグラフを 'attention_result.png' に保存しました。")

if __name__ == "__main__":
    trained_model, data_scaler = train()
    visualize_results(trained_model)