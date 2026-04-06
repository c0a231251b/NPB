import json
import re
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ==========================================
# 1. データ加工・累積成績算出セクション
# ==========================================

def classify_play(line):
    """テキストから打席結果を判定する """
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
    """選手の累積セイバーメトリクス指標を管理 [cite: 24, 44]"""
    def __init__(self):
        self.db = {}

    def get_vector(self, name):
        """1番から9番打者の特徴ベクトル(AVG, OBP, SLG, ISO)を生成 [cite: 14, 43, 74]"""
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
    """全試合を時系列順に読み込み学習データを作成 [cite: 15, 16]"""
    stats_db = PlayerStatsDB()
    dataset = []
    file_paths = sorted(glob.glob(f"{json_dir}/*.json"))
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            game = json.load(f)
        
        # 試合開始時点のデータで学習サンプルを作成 [cite: 44, 74]
        for team_idx, team_info in enumerate(game["scoreboard"]):
            score = int(team_info["R"])
            try:
                lineup = game["text_live"][0]["pregame"]["lineups"][team_idx]["players"][:9]
                lineup_vectors = [stats_db.get_vector(name) for name in lineup]
                dataset.append({"lineup_vectors": lineup_vectors, "score": score})
            except (IndexError, KeyError):
                continue
        
        # 試合終了後、成績を更新 [cite: 15, 24]
        for section in game["text_live"]:
            if "回" not in section["inning"]: continue
            for play in section["plays"]:
                if not play["lines"]: continue
                match = re.match(r"\d番\s*([^\s]+)", play["lines"][0])
                if match:
                    res = classify_play(" ".join(play["lines"][1:]))
                    stats_db.update(match.group(1), res)
    return dataset, stats_db

# ==========================================
# 2. RNNモデル定義セクション (論文準拠)
# ==========================================

class BattingOrderRNN(nn.Module):
    """再帰型ニューラルネットワークによる得点予測モデル [cite: 20, 43]"""
    def __init__(self, input_size, hidden_size):
        super(BattingOrderRNN, self).__init__()
        # 再帰型層としてLSTMを採用 [cite: 51]
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 全結合層 (DENSE) [cite: 48, 57]
        self.fc = nn.Linear(hidden_size, 1)
        # 活性化関数 (ReLU) [cite: 48, 55, 98]
        self.relu = nn.ReLU()

    def forward(self, x):
        # 隠れ層ベクトル h を得る [cite: 66]
        _, (h_n, _) = self.lstm(x) 
        # スコアを回帰 [cite: 70, 71]
        out = self.fc(h_n[-1])
        out = self.relu(out)
        return out

# ==========================================
# 3. シミュレーション・推論セクション
# ==========================================

def simulate_batting_order(model, stats_db, player_names):
    """実際の打順と最適化(出塁率降順)打順を比較評価 """
    # 実際の打順のベクトル
    original_vectors = [stats_db.get_vector(name) for name in player_names]
    # 出塁率(index 1)が高い順にソートした打順 [cite: 17, 130]
    sorted_players = sorted(player_names, key=lambda n: stats_db.get_vector(n)[1], reverse=True)
    sorted_vectors = [stats_db.get_vector(name) for name in sorted_players]

    model.eval()
    with torch.no_grad():
        x_orig = torch.tensor([original_vectors], dtype=torch.float32)
        x_sort = torch.tensor([sorted_vectors], dtype=torch.float32)
        pred_orig = model(x_orig).item()
        pred_sort = model(x_sort).item()

    print(f"\n--- 打順シミュレーション結果 ---")
    print(f"実際の打順: {' -> '.join(player_names[:5])}...")
    print(f"  > 予測得点: {pred_orig:.2f}")
    print(f"出塁率降順: {' -> '.join(sorted_players[:5])}...")
    print(f"  > 予測得点: {pred_sort:.2f}")
    print(f"最適化による期待上昇スコア: {pred_sort - pred_orig:.2f}")

# ==========================================
# 4. メイン実行プロセス
# ==========================================

if __name__ == "__main__":
    JSON_DIR = "game_data"
    print("1. データセット構築中...")
    train_data, stats_db = build_dataset(JSON_DIR)

    if not train_data:
        print("エラー: 学習データが作成されませんでした。")
    else:
        # データの準備 [cite: 44, 74]
        X = np.array([s["lineup_vectors"] for s in train_data], dtype=np.float32)
        y = np.array([s["score"] for s in train_data], dtype=np.float32).reshape(-1, 1)
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        # モデル・損失関数・最適化手法の設定 [cite: 50, 97]
        model = BattingOrderRNN(input_size=4, hidden_size=32)
        criterion = nn.MSELoss() 
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        print(f"2. 学習開始... (サンプル数: {len(train_data)})")
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
            if (epoch + 1) % 20 == 0:
                print(f"   Epoch [{epoch+1}/100], Loss (MSE): {total_loss/len(loader):.4f}")

        print("3. 学習完了！ [cite: 102]")

        # 4. シミュレーション実行 [cite: 17, 130]
        # テスト用の打順 (収集したデータに含まれる選手名で指定してください)
        test_lineup = ["牧", "度会", "佐野", "宮﨑", "ビシエド", "山本", "梶原", "林", "石田裕"]
        simulate_batting_order(model, stats_db, test_lineup)