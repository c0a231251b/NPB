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

# 日本語フォントの設定（Windows標準）
plt.rcParams['font.family'] = 'MS Gothic'

# ==========================================
# 1. 2024年実績を活用する成績管理クラス
# ==========================================

class PlayerStatsDB:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # リーグ平均の取得
        if 'LEAGUE_AVERAGE' in df['name'].values:
            avg_row = df[df['name'] == 'LEAGUE_AVERAGE'].iloc[0]
            self.league_avg = avg_row.to_dict()
        else:
            self.league_avg = {"AB": 100, "H": 25, "TB": 35, "BB": 10, "HBP": 1, "SF": 2}
        
        # 選手データの構築
        players_df = df[df['name'] != 'LEAGUE_AVERAGE'].copy()
        self.db = {}
        for _, row in players_df.iterrows():
            t = self.normalize_team_name(row['team'])
            n = row['name']
            if t not in self.db: self.db[t] = {}
            self.db[t][n] = row.to_dict()

    def normalize_team_name(self, name):
        return name.replace("ＤｅＮＡ", "DeNA").replace("巨 人", "巨人").replace("ヤクルト", "ヤクルト")

    def clean_name(self, name):
        return name.replace(" ", "").replace("　", "").replace("*", "").replace("+", "")

    def update_after_score(self, name, team_name, play_res):
        """打席結果を成績DBに反映"""
        target_name = self.clean_name(name)
        norm_team = self.normalize_team_name(team_name)
        
        # 存在確認（なければ初期化）
        _ = self.get_vector(target_name, team_name)
        
        if norm_team in self.db and target_name in self.db[norm_team]:
            for k, v in play_res.items():
                if k in self.db[norm_team][target_name]:
                    self.db[norm_team][target_name][k] += v

    def get_vector(self, name, team_name):
        target_name = self.clean_name(name)
        norm_team = self.normalize_team_name(team_name)
        
        if norm_team not in self.db: self.db[norm_team] = {}
        team_data = self.db[norm_team]
        
        # 直接一致または前方一致で検索
        s = team_data.get(target_name)
        if s is None:
            matches = [full for full in team_data.keys() if full.startswith(target_name)]
            if matches:
                s = team_data[max(matches, key=lambda m: team_data[m].get("AB", 0))]

        # 他球団からの移籍チェック
        if s is None:
            for other_team, players in self.db.items():
                if target_name in players:
                    s = players[target_name].copy()
                    self.db[norm_team][target_name] = s
                    break

        # 新外国人や新人（リーグ平均）
        if s is None:
            s = self.league_avg.copy()
            self.db[norm_team][target_name] = s
        
        ab, h, bb, hbp, sf, tb = s.get("AB",0), s.get("H",0), s.get("BB",0), s.get("HBP",0), s.get("SF",0), s.get("TB",0)
        
        # ベクトルの算出
        avg = h / ab if ab > 0 else 0.0
        denom = (ab + bb + hbp + sf)
        obp = (h + bb + hbp) / denom if denom > 0 else 0.0
        slg = tb / ab if ab > 0 else 0.0
        iso = slg - avg
        return [avg, obp, slg, iso]

# --- 打席結果の分類ロジック ---
def classify_play(line):
    res = {"H": 0, "TB": 0, "AB": 1, "BB": 0, "SF": 0, "HBP": 0}
    if any(kw in line for kw in ["四球", "敬遠", "四選"]): res["BB"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["死球"]): res["HBP"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["本塁打"]): res["H"], res["TB"] = 1, 4
    elif any(kw in line for kw in ["三塁打"]): res["H"], res["TB"] = 1, 3
    elif any(kw in line for kw in ["二塁打"]): res["H"], res["TB"] = 1, 2
    elif any(kw in line for kw in ["安打", "内安"]): res["H"], res["TB"] = 1, 1
    elif "犠飛" in line: res["SF"], res["AB"] = 1, 0
    return res

# ==========================================
# 2. データセット構築ロジック（2行JSON対応）
# ==========================================

def build_dataset(json_dir, csv_path):
    stats_db = PlayerStatsDB(csv_path)
    dataset = []
    file_paths = sorted(glob.glob(f"{json_dir}/*.json"))
    
    print(f"解析中... 対象ファイル数: {len(file_paths)}")
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            try:
                game = json.load(f)
            except json.JSONDecodeError: continue
        
        try:
            teams = [t["team"] for t in game["scoreboard"]]
            scores = [int(re.search(r'\d+', t["R"]).group()) for t in game["scoreboard"]]
        except: continue

        for team_idx, team_name in enumerate(teams):
            try:
                # スタメン取得
                players = game["text_live"][0]["pregame"]["lineups"][team_idx]["players"]
                
                # --- 【ここが修正ポイント】 ---
                # 9人揃っていないデータは学習から除外する
                if len(players) < 9:
                    # print(f"  Skip: {os.path.basename(path)} のスタメンが不足しています")
                    continue
                
                lineup = players[:9]
                lineup_vectors = [stats_db.get_vector(name, team_name) for name in lineup]
                
                # 全員のベクトルが正しく4要素であることも確認
                if all(len(v) == 4 for v in lineup_vectors):
                    dataset.append({"lineup_vectors": lineup_vectors, "score": scores[team_idx]})
                # -----------------------------
                
            except: continue
        
        # 試合中の成績更新処理（ここは変更なし）
        for section in game.get("text_live", []):
            inning = section.get("inning", "")
            if "回" not in inning: continue
            current_team = teams[0] if "表" in inning else teams[1]
            for play in section.get("plays", []):
                lines = play.get("lines", [])
                if len(lines) < 2: continue
                match = re.match(r"(\d+)番\s+(.+)", lines[0])
                if match:
                    p_name = match.group(2)
                    res_stats = classify_play(lines[1])
                    stats_db.update_after_score(p_name, current_team, res_stats)
                    
    return dataset, stats_db

# ==========================================
# 3. RNN + Attention モデル定義
# ==========================================

class BattingOrderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BattingOrderRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # (batch, 9, hidden_size)
        attn_scores = self.attention_weights(lstm_out) # (batch, 9, 1)
        soft_attn_weights = F.softmax(attn_scores, dim=1) # 合計1の重み
        context = torch.sum(soft_attn_weights * lstm_out, dim=1) # 加重平均
        out = self.fc(self.dropout(context))
        return out

def visualize_ai_attention(model, stats_db, team_name, player_names):
    v = [stats_db.get_vector(name, team_name) for name in player_names]
    x = torch.tensor([v], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        lstm_out, _ = model.lstm(x)
        attn_scores = model.attention_weights(lstm_out)
        soft_attn_weights = F.softmax(attn_scores, dim=1).squeeze().numpy()

    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 10), soft_attn_weights, color='purple', alpha=0.7)
    plt.title(f"AI Attention Weights (2024 Initial Stats Base) - {team_name}")
    plt.xlabel("Batting Order")
    plt.ylabel("Importance")
    plt.xticks(range(1, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    os.makedirs("graph_data", exist_ok=True)
    plt.savefig("graph_data/ai_attention_weights_2024base.png")
    plt.show()

# ==========================================
# 4. メイン処理
# ==========================================

if __name__ == "__main__":
    JSON_DIR = "game_data_2025"
    CSV_FILE = "initial_stats_2024.csv"
    
    data_list, stats_db = build_dataset(JSON_DIR, CSV_FILE)

    if data_list:
        X = np.array([d["lineup_vectors"] for d in data_list], dtype=np.float32)
        y = np.array([d["score"] for d in data_list], dtype=np.float32).reshape(-1, 1)
        
        loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=32, shuffle=True)

        model = BattingOrderRNN(input_size=4, hidden_size=64)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        print(f"学習開始 (サンプル数: {len(data_list)})")
        for epoch in range(100):
            model.train()
            total_loss = 0
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 20 == 0:
                print(f"Epoch [{epoch+1}/100], MSE: {total_loss/len(loader):.4f}")

        # 巨人軍の現行オーダー等で可視化テスト
        test_lineup = ["坂本", "丸", "ヘルナンデス", "吉川", "岡本", "オコエ", "小林", "門脇", "菅野"]
        visualize_ai_attention(model, stats_db, "巨人", test_lineup)