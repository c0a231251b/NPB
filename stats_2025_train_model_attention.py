import json
import re
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 2025年初期値を活用する成績管理クラス
# ==========================================

class PlayerStatsDB:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        if 'LEAGUE_AVERAGE' in df['name'].values:
            avg_row = df[df['name'] == 'LEAGUE_AVERAGE'].iloc[0]
            self.league_avg = avg_row.to_dict()
        else:
            self.league_avg = {"AB": 100, "H": 25, "TB": 35, "BB": 10, "HBP": 1, "SF": 2}
        
        players_df = df[df['name'] != 'LEAGUE_AVERAGE'].copy()
        players_df['name'] = players_df['name'].str.replace(r'\s+', '', regex=True)
        
        self.db = {}
        for _, row in players_df.iterrows():
            t = self.normalize_team_name(row['team'])
            n = row['name']
            if t not in self.db: self.db[t] = {}
            if n in self.db[t]:
                for col in ["AB", "H", "2B", "3B", "HR", "TB", "BB", "HBP", "SF"]:
                    self.db[t][n][col] += row[col]
            else:
                self.db[t][n] = row.to_dict()

    def normalize_team_name(self, name):
        return name.replace("ＤｅＮＡ", "DeNA").replace("巨 人", "巨人")

    def clean_name(self, name):
        return name.replace(" ", "").replace("　", "")

    # --- ここが抜けていました！ ---
    def update_after_score(self, name, team_name, play_res):
        """得点が入った際の打者の成績をDBに反映する"""
        target_name = self.clean_name(name)
        # get_vectorを呼ぶことで、未登録選手ならリーグ平均で初期化、移籍なら名寄せを行う
        _ = self.get_vector(target_name, team_name)
        norm_team = self.normalize_team_name(team_name)
        
        if norm_team in self.db and target_name in self.db[norm_team]:
            for k, v in play_res.items():
                if k in self.db[norm_team][target_name]:
                    self.db[norm_team][target_name][k] += v

    def get_vector(self, name, team_name):
        target_name = self.clean_name(name)
        norm_team = self.normalize_team_name(team_name)
        if norm_team not in self.db: self.db[norm_team] = {}
        team_data = self.db[norm_team]
        
        s = None
        if target_name in team_data:
            s = team_data[target_name]
        else:
            matches = [full for full in team_data.keys() if full.startswith(target_name)]
            if matches:
                best = max(matches, key=lambda m: team_data[m].get("AB", 0))
                s = team_data[best]

        if s is None:
            all_matches = []
            for other_team, players in self.db.items():
                for full_name, stats in players.items():
                    if full_name.startswith(target_name) or target_name == full_name:
                        all_matches.append(stats)
            if all_matches:
                s = max(all_matches, key=lambda x: x.get("AB", 0))
                self.db[norm_team][target_name] = s
            else:
                s = self.league_avg.copy()
                self.db[norm_team][target_name] = s
        
        ab, h, bb, hbp, sf, tb = s.get("AB",0), s.get("H",0), s.get("BB",0), s.get("HBP",0), s.get("SF",0), s.get("TB",0)
        avg = h / ab if ab > 0 else 0.0
        denom = (ab + bb + hbp + sf)
        obp = (h + bb + hbp) / denom if denom > 0 else 0.0
        slg = tb / ab if ab > 0 else 0.0
        iso = slg - avg
        return [avg, obp, slg, iso]

# --- 2026年試合結果のパース用関数 ---
def classify_play(line):
    res = {"H": 0, "TB": 0, "AB": 1, "BB": 0, "SF": 0, "HBP": 0}
    if any(kw in line for kw in ["フォアボール", "四球", "敬遠"]):
        res["BB"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["デッドボール", "死球"]):
        res["HBP"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["ホームラン", "本塁打"]):
        res["H"], res["TB"] = 1, 4
    elif any(kw in line for kw in ["ヒット", "安打"]):
        res["H"], res["TB"] = 1, 1
    elif "犠牲フライ" in line:
        res["SF"], res["AB"] = 1, 0
    return res

def build_dataset(json_dir, csv_path):
    stats_db = PlayerStatsDB(csv_path)
    dataset = []
    file_paths = sorted(glob.glob(f"{json_dir}/*.json"))
    # 全言語対応のクリーンな正規表現
    score_re = re.compile(r"[A-Za-z0-9一-龠ぁ-んァ-ヶ]+\s+(\d+)-(\d+)\s+[A-Za-z0-9一-龠ぁ-んァ-ヶ]+")
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            game = json.load(f)
        teams = [t["team"] for t in game["scoreboard"]]
        for team_idx, team_name in enumerate(teams):
            score = int(game["scoreboard"][team_idx]["R"])
            try:
                lineup = game["text_live"][0]["pregame"]["lineups"][team_idx]["players"][:9]
                lineup_vectors = [stats_db.get_vector(name, team_name) for name in lineup]
                dataset.append({"lineup_vectors": lineup_vectors, "score": score})
            except (IndexError, KeyError): continue
        
        # 試合中のアップデート
        for section in game["text_live"]:
            inning = section["inning"]
            if "回" not in inning: continue
            current_team = teams[0] if "表" in inning else teams[1]
            for play in section["plays"]:
                if not play["lines"]: continue
                match = re.match(r"\d番\s+([^\s]+(?:\s+[^\s]+)?)", play["lines"][0])
                if match:
                    res = classify_play(" ".join(play["lines"][1:]))
                    # スコア変動チェック
                    m_scores = score_re.findall(" ".join(play["lines"]))
                    if m_scores:
                        stats_db.update_after_score(match.group(1), current_team, res)
                    else:
                        # 通常の打席更新（簡易化）
                        pass
    return dataset, stats_db

# ==========================================
# 2. RNN + Attention モデル定義
# ==========================================

class BattingOrderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BattingOrderRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention_weights(lstm_out)
        soft_attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(soft_attn_weights * lstm_out, dim=1)
        out = self.fc(self.dropout(context))
        return out 

def visualize_ai_attention(model, stats_db, team_name, player_names):
    os.makedirs("graph_data", exist_ok=True)
    v = [stats_db.get_vector(name, team_name) for name in player_names]
    x = torch.tensor([v], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        lstm_out, _ = model.lstm(x)
        attn_scores = model.attention_weights(lstm_out)
        soft_attn_weights = F.softmax(attn_scores, dim=1).squeeze().numpy()
    
    plt.rcParams['font.family'] = 'MS Gothic' # Windowsの標準日本語フォント
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 10), soft_attn_weights, color='purple', alpha=0.7)
    plt.title(f"AI Attention Weights for {team_name} Lineup")
    plt.xlabel("Batting Order")
    plt.ylabel("Attention Weight")
    plt.xticks(range(1, 10))
    plt.savefig("graph_data/ai_attention_weights.png")
    plt.show()
    for i, w in enumerate(soft_attn_weights):
        print(f"{i+1}番打者の重要度: {w:.4f}")

# ==========================================
# 3. メイン処理
# ==========================================

if __name__ == "__main__":
    JSON_DIR = "game_data"
    CSV_FILE = "initial_stats_2025.csv"
    
    print("データセット構築中...")
    train_data, stats_db = build_dataset(JSON_DIR, CSV_FILE)

    if train_data:
        X = np.array([s["lineup_vectors"] for s in train_data], dtype=np.float32)
        y = np.array([s["score"] for s in train_data], dtype=np.float32).reshape(-1, 1)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=16, shuffle=True)

        model = BattingOrderRNN(input_size=4, hidden_size=64)
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
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

        # 可視化の実行
        test_team = "巨人"
        test_lineup = ["キャベッジ", "松本 剛", "泉口", "ダルベック", "岸田 行倫", "中山", "坂本 勇人", "浦田", "竹丸"]
        visualize_ai_attention(model, stats_db, test_team, test_lineup)