import json
import re
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

# ==========================================
# 1. 2025年初期値を活用する成績管理クラス
# ==========================================

class PlayerStatsDB:
    def __init__(self, csv_path):
        # 1. CSVを読み込む
        df = pd.read_csv(csv_path)
        
        # 2. リーグ平均行を分離して保持
        if 'LEAGUE_AVERAGE' in df['name'].values:
            avg_row = df[df['name'] == 'LEAGUE_AVERAGE'].iloc[0]
            self.league_avg = avg_row.to_dict()
        else:
            # 万が一平均データがない場合のデフォルト値
            self.league_avg = {"AB": 100, "H": 25, "TB": 35, "BB": 10, "HBP": 1, "SF": 2}
        
        # 3. 同名選手の重複問題を解決する (ここが修正ポイント！)
        # 名前が 'LEAGUE_AVERAGE' 以外の選手データを抽出
        players_df = df[df['name'] != 'LEAGUE_AVERAGE'].copy()
        
        # 名前(name)でグループ化して数値を合計する
        # これにより、重複した名前があっても1行にまとめられます
        players_df = players_df.groupby('name').sum()
        
        # 4. 辞書形式に変換
        self.db = players_df.to_dict('index')

    def get_vector(self, name):
        """
        選手名から特徴ベクトル(AVG, OBP, SLG, ISO)を生成。
        実績がない選手はLEAGUE_AVERAGEで補完する。
        """
        # データベースに名前がない場合はリーグ平均を採用
        s = self.db.get(name, self.league_avg)
        
        # 計算用数値の抽出
        ab = s.get("AB", 0)
        h = s.get("H", 0)
        bb = s.get("BB", 0)
        hbp = s.get("HBP", 0)
        sf = s.get("SF", 0)
        tb = s.get("TB", 0)

        # セイバーメトリクス指標の算出
        # 打率 (AVG) = 安打 / 打数
        avg = h / ab if ab > 0 else 0.0
        
        # 出塁率 (OBP) = (安打 + 四球 + 死球) / (打数 + 四球 + 死球 + 犠飛)
        denom_obp = (ab + bb + hbp + sf)
        obp = (h + bb + hbp) / denom_obp if denom_obp > 0 else 0.0
        
        # 長打率 (SLG) = 塁打 / 打数
        slg = tb / ab if ab > 0 else 0.0
        
        # ISO = 長打率 - 打率
        iso = slg - avg
        
        return [avg, obp, slg, iso]

    def update(self, name, play_res):
        """2026年シーズン中の打席結果を累積データに加算"""
        if name not in self.db:
            # 新登場選手はリーグ平均からスタートさせる
            new_entry = self.league_avg.copy()
            self.db[name] = new_entry
            
        for k, v in play_res.items():
            if k in self.db[name]:
                self.db[name][k] += v

# --- 2026年試合結果のパース用関数 (既存のものを流用) ---
def classify_play(line):
    res = {"H": 0, "TB": 0, "AB": 1, "BB": 0, "SF": 0, "HBP": 0}
    if any(kw in line for kw in ["フォアボール", "四球", "敬遠"]):
        res["BB"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["デッドボール", "死球"]):
        res["HBP"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["ホームラン", "本塁打"]):
        res["H"], res["TB"] = 1, 4
    elif any(kw in line for kw in ["三塁打", "スリーベース"]):
        res["H"], res["TB"] = 1, 3
    elif any(kw in line for kw in ["二塁打", "ツーベース"]):
        res["H"], res["TB"] = 1, 2
    elif any(kw in line for kw in ["ヒット", "安打"]):
        res["H"], res["TB"] = 1, 1
    elif "犠牲フライ" in line:
        res["SF"], res["AB"] = 1, 0
    return res

def build_dataset(json_dir, csv_path):
    # 2025年の実績を初期値としてロード
    stats_db = PlayerStatsDB(csv_path)
    dataset = []
    file_paths = sorted(glob.glob(f"{json_dir}/*.json"))
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            game = json.load(f)
        for team_idx, team_info in enumerate(game["scoreboard"]):
            score = int(team_info["R"])
            try:
                lineup = game["text_live"][0]["pregame"]["lineups"][team_idx]["players"][:9]
                # 2025年実績 + 2026年これまでの成績 でベクトル化
                lineup_vectors = [stats_db.get_vector(name) for name in lineup]
                dataset.append({"lineup_vectors": lineup_vectors, "score": score})
            except (IndexError, KeyError):
                continue
        # 試合結果を反映
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
# 2. RNNモデル定義 & 実行セクション
# ==========================================

class BattingOrderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BattingOrderRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        # return self.relu(out)
        return out  # 生の数値を出力させる

def simulate_batting_order(model, stats_db, player_names):
    orig_v = [stats_db.get_vector(name) for name in player_names]
    # 出塁率(index 1)降順ソート
    sorted_p = sorted(player_names, key=lambda n: stats_db.get_vector(n)[1], reverse=True)
    sorted_v = [stats_db.get_vector(name) for name in sorted_p]

    model.eval()
    with torch.no_grad():
        x_orig = torch.tensor([orig_v], dtype=torch.float32)
        x_sort = torch.tensor([sorted_v], dtype=torch.float32)
        p_orig = model(x_orig).item()
        p_sort = model(x_sort).item()

    print(f"\n--- 打順評価結果 (2025実績反映版) ---")
    print(f"実際の打順予測得点: {p_orig:.2f}")
    print(f"出塁率降順予測得点: {p_sort:.2f}")
    print(f"期待上昇スコア: {p_sort - p_orig:.2f}")

if __name__ == "__main__":
    JSON_DIR = "game_data"
    CSV_FILE = "initial_stats_2025.csv"
    
    print("データセット構築中（2025年実績ロード中）...")
    train_data, stats_db = build_dataset(JSON_DIR, CSV_FILE)

    if train_data:
        X = np.array([s["lineup_vectors"] for s in train_data], dtype=np.float32)
        y = np.array([s["score"] for s in train_data], dtype=np.float32).reshape(-1, 1)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=16, shuffle=True)

        model = BattingOrderRNN(input_size=4, hidden_size=64) # 表現力向上のため隠れ層を64に
        optimizer = optim.Adam(model.parameters(), lr=0.0005) # 学習率を少し下げて安定化
        criterion = nn.MSELoss()

        print(f"学習開始... (サンプル数: {len(train_data)})")
        for epoch in range(100):
            model.train()
            l_sum = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                l_sum += loss.item()
            if (epoch+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/100], MSE: {l_sum/len(loader):.4f}")

        # 巨人のスタメン例（坂本 [cite: 146] や 門脇  も2025実績あり）
        test_lineup = ["佐々木", "門脇", "坂本", "岡本和", "大城卓", "丸", "萩尾", "吉川", "戸郷"]
        # 指定した選手のベクトルが [0, 0, 0, 0] になっていないか確認
        print(f"DEBUG: 岡本和の成績ベクトル -> {stats_db.get_vector('岡本和')}")
        simulate_batting_order(model, stats_db, test_lineup)