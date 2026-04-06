import json
import re
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- 1. データ加工・累積成績算出ロジック ---

def classify_play(line):
    res = {"H": 0, "TB": 0, "AB": 0, "BB": 0, "SF": 0}
    if any(kw in line for kw in ["フォアボール", "四球", "敬遠", "デッドボール", "死球"]):
        res["BB"] = 1
    elif any(kw in line for kw in ["ホームラン", "本塁打"]):
        res["H"], res["TB"], res["AB"] = 1, 4, 1
    elif any(kw in line for kw in ["三塁打", "スリーベース"]):
        res["H"], res["TB"], res["AB"] = 1, 3, 1
    elif any(kw in line for kw in ["二塁打", "ツーベース"]):
        res["H"], res["TB"], res["AB"] = 1, 2, 1
    elif any(kw in line for kw in ["ヒット", "安打"]):
        res["H"], res["TB"], res["AB"] = 1, 1, 1
    elif "犠牲フライ" in line:
        res["SF"] = 1
    elif any(kw in line for kw in ["三振", "ゴロ", "フライ", "ライナー", "ダブルプレー", "併殺"]):
        res["AB"] = 1
    return res

class PlayerStatsDB:
    def __init__(self):
        self.db = {}
    def get_vector(self, name):
        s = self.db.get(name, {"H": 0, "TB": 0, "AB": 0, "BB": 0, "SF": 0})
        avg = s["H"] / s["AB"] if s["AB"] > 0 else 0.0
        obp = (s["H"] + s["BB"]) / (s["AB"] + s["BB"] + s["SF"]) if (s["AB"] + s["BB"] + s["SF"]) > 0 else 0.0
        slg = s["TB"] / s["AB"] if s["AB"] > 0 else 0.0
        iso = slg - avg
        return [avg, obp, slg, iso]
    def update(self, name, play_res):
        if name not in self.db:
            self.db[name] = {"H": 0, "TB": 0, "AB": 0, "BB": 0, "SF": 0}
        for k, v in play_res.items():
            self.db[name][k] += v

def build_dataset(json_dir):
    stats_db = PlayerStatsDB()
    dataset = []
    file_paths = sorted(glob.glob(f"{json_dir}/*.json"))
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            game = json.load(f)
        for team_idx, team_info in enumerate(game["scoreboard"]):
            score = int(team_info["R"])
            try:
                lineup = game["text_live"][0]["pregame"]["lineups"][team_idx]["players"][:9]
                lineup_vectors = [stats_db.get_vector(name) for name in lineup]
                dataset.append({"lineup_vectors": lineup_vectors, "score": score})
            except (IndexError, KeyError):
                continue
        for section in game["text_live"]:
            if "回" not in section["inning"]: continue
            for play in section["plays"]:
                if not play["lines"]: continue
                match = re.match(r"\d番\s*([^\s]+)", play["lines"][0])
                if match:
                    res = classify_play(" ".join(play["lines"][1:]))
                    stats_db.update(match.group(1), res)
    return dataset

# --- 2. モデル定義セクション (論文 [cite: 43, 47, 51]) ---

class BattingOrderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BattingOrderRNN, self).__init__()
        # 論文に基づきLSTMを採用 [cite: 51]
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 全結合層 (DENSE) [cite: 48, 57]
        self.fc = nn.Linear(hidden_size, 1)
        # 活性化関数 (ReLU) [cite: 48, 55, 98]
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x) # 隠れ層ベクトル h を得る [cite: 66]
        out = self.fc(h_n[-1])     # DENSE層を適用 [cite: 70]
        out = self.relu(out)       # ReLUでスコアを回帰 [cite: 70, 71]
        return out

# --- 3. メイン実行プロセス ---

if __name__ == "__main__":
    JSON_DIR = "game_data"
    print("データセット構築中...")
    train_data = build_dataset(JSON_DIR)
    
    if not train_data:
        print("エラー: 学習データが作成されませんでした。")
    else:
        # データのテンソル変換 
        # 入力ベクトル列 X: 1番から9番までのベクトル列 [cite: 74]
        X = np.array([s["lineup_vectors"] for s in train_data], dtype=np.float32)
        # 教師ラベル y: その試合の獲得得点 [cite: 44]
        y = np.array([s["score"] for s in train_data], dtype=np.float32).reshape(-1, 1)

        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # モデル初期化 (入力次元: 4指標, 隠れ層: 32)
        model = BattingOrderRNN(input_size=4, hidden_size=32)
        criterion = nn.MSELoss() # 損失関数: 平均二乗誤差 (RMSEのベース) [cite: 50]
        optimizer = optim.Adam(model.parameters(), lr=0.001) # 実験設定 [cite: 97]

        print(f"学習開始... (サンプル数: {len(train_data)})")
        for epoch in range(100):
            model.train()
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/100], Loss (MSE): {total_loss/len(loader):.4f}")

        print("学習完了！最終的なMSEは論文の実験結果(約4.2)と比較評価可能です [cite: 102]。")