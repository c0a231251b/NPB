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
            # 球団名を正規化（全角を半角にするなど）して格納
            t = self.normalize_team_name(row['team'])
            n = row['name']
            if t not in self.db:
                self.db[t] = {}
            
            if n in self.db[t]:
                for col in ["AB", "H", "2B", "3B", "HR", "TB", "BB", "HBP", "SF"]:
                    self.db[t][n][col] += row[col]
            else:
                self.db[t][n] = row.to_dict()

    def normalize_team_name(self, name):
        """球団名の表記揺れ（DeNAの全角/半角など）を統一する"""
        # 全角のＤｅＮＡを半角に、あるいはその逆を考慮
        return name.replace("ＤｅＮＡ", "DeNA").replace("巨 人", "巨人")

    def clean_name(self, name):
        return name.replace(" ", "").replace("　", "")

    def get_vector(self, name, team_name):
        target_name = self.clean_name(name)
        # 検索時も球団名を正規化
        norm_team = self.normalize_team_name(team_name)
        
        # チームが存在しない場合は空の辞書をデフォルトにする（KeyError防止）
        if norm_team not in self.db:
            self.db[norm_team] = {}
        
        team_data = self.db[norm_team]
        
        # 1. 自チーム内で検索(完全一致or名字マッチング)
        s = None
        if target_name in team_data:
            s = team_data[target_name]
        else:
            matches = [full for full in team_data.keys() if full.startswith(target_name)]
            if matches:
                best = max(matches, key=lambda m: team_data[m].get("AB", 0))
                s = team_data[best]
                
        # 2. 【移籍対策】自チームで見つからない場合、他チーム（全DB）を捜索
        if s is None:
            all_matches = []
            for other_team, players in self.db.items():
                for full_name, stats in players.items():
                    # 名字またはフルネームで一致するかチェック
                    if full_name.startswith(target_name) or target_name == full_name:
                        all_matches.append(stats)
            
            if all_matches:
                # リーグ全体で最も実績（打数）がある選手を採用
                s = max(all_matches, key=lambda x: x.get("AB", 0))
                # 発見したデータを、現在のチームのキャッシュに保存（移籍完了として扱う）
                self.db[norm_team][target_name] = s
            else:
                # 3. リーグ全体を探してもいない場合のみ、本当の新規選手（リーグ平均）
                s = self.league_avg.copy()
                self.db[norm_team][target_name] = s
        
        ab, h, bb, hbp, sf, tb = s.get("AB",0), s.get("H",0), s.get("BB",0), s.get("HBP",0), s.get("SF",0), s.get("TB",0)
        avg = h / ab if ab > 0 else 0.0
        denom = (ab + bb + hbp + sf)
        obp = (h + bb + hbp) / denom if denom > 0 else 0.0
        slg = tb / ab if ab > 0 else 0.0
        iso = slg - avg
        return [avg, obp, slg, iso]

    def update(self, name, team_name, play_res):
        """シーズン中の結果を特定の球団の選手に対して加算"""
        target_name = self.clean_name(name)
        # get_vector を通じて球団内の対象を特定/キャッシュ
        _ = self.get_vector(target_name, team_name)
        
        if team_name in self.db and target_name in self.db[team_name]:
            for k, v in play_res.items():
                if k in self.db[team_name][target_name]:
                    self.db[team_name][target_name][k] += v

# --- 2026年試合結果のパース用関数 ---
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
    stats_db = PlayerStatsDB(csv_path)
    dataset = []
    file_paths = sorted(glob.glob(f"{json_dir}/*.json"))
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            game = json.load(f)
        
        # チーム名の取得 (先攻=index0, 後攻=index1)
        teams = [t["team"] for t in game["scoreboard"]]
        
        # 1. スタメン情報のベクトル化
        for team_idx, team_name in enumerate(teams):
            score = int(game["scoreboard"][team_idx]["R"])
            try:
                lineup = game["text_live"][0]["pregame"]["lineups"][team_idx]["players"][:9]
                # チーム名を渡して特定
                lineup_vectors = [stats_db.get_vector(name, team_name) for name in lineup]
                dataset.append({"lineup_vectors": lineup_vectors, "score": score})
            except (IndexError, KeyError):
                continue
        
        # 2. 試合結果を成績DBに反映 (時系列)
        for section in game["text_live"]:
            inning = section["inning"]
            if "回" not in inning: continue
            
            # イニングが「表」なら先攻チーム、「裏」なら後攻チーム
            current_team = teams[0] if "表" in inning else teams[1]
            
            for play in section["plays"]:
                if not play["lines"]: continue
                # スペース込みの名前にも対応する正規表現
                match = re.match(r"\d番\s+([^\s]+(?:\s+[^\s]+)?)", play["lines"][0])
                if match:
                    res = classify_play(" ".join(play["lines"][1:]))
                    stats_db.update(match.group(1), current_team, res)
                    
    return dataset, stats_db

# ==========================================
# 2. RNNモデル定義
# ==========================================

class BattingOrderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BattingOrderRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out 

def simulate_batting_order(model, stats_db, team_name, player_names):
    # シミュレーション対象の球団名を渡してベクトル化
    orig_v = [stats_db.get_vector(name, team_name) for name in player_names]
    sorted_p = sorted(player_names, key=lambda n: stats_db.get_vector(n, team_name)[1], reverse=True)
    sorted_v = [stats_db.get_vector(name, team_name) for name in sorted_p]

    model.eval()
    with torch.no_grad():
        x_orig = torch.tensor([orig_v], dtype=torch.float32)
        x_sort = torch.tensor([sorted_v], dtype=torch.float32)
        p_orig = model(x_orig).item()
        p_sort = model(x_sort).item()

    print(f"\n--- {team_name} 打順評価結果 ---")
    print(f"実際の打順予測得点: {p_orig:.2f}")
    print(f"出塁率降順予測得点: {p_sort:.2f}")
    print(f"期待上昇スコア: {p_sort - p_orig:.2f}")

if __name__ == "__main__":
    JSON_DIR = "game_data"
    CSV_FILE = "initial_stats_2025.csv"
    
    print("データセット構築中（球団別特定ロジック適用）...")
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

        # 巨人（後攻）のスタメン例での評価テスト
        test_team = "巨人"
        test_lineup = ["キャベッジ", "松本 剛", "泉口", "ダルベック", "岸田 行倫", "中山", "坂本 勇人", "浦田", "竹丸"]
        print(f"DEBUG: 岡本和の成績ベクトル -> {stats_db.get_vector('岡本和', '巨人')}")
        simulate_batting_order(model, stats_db, test_team, test_lineup)

        