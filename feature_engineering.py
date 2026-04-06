import json
import glob
import os
import re
import pandas as pd

# --- 設定 ---
JSON_DIR = "game_data"
# 算出する指標のリスト（論文の13指標のうち、テキストから計算可能なもの）
FEATURES = ["AVG", "OBP", "SLG", "ISO"] 

# --- 打席結果の判定ロジック ---
def classify_play(line):
    res = {"H": 0, "TB": 0, "AB": 0, "BB": 0, "SF": 0}
    
    # 1. 四死球 (打数に含めない)
    if any(kw in line for kw in ["フォアボール", "四球", "敬遠", "デッドボール", "死球"]):
        res["BB"] = 1
    # 2. 安打 (打数に含める)
    elif "ホームラン" in line or "本塁打" in line:
        res["H"], res["TB"], res["AB"] = 1, 4, 1
    elif "三塁打" in line or "スリーベース" in line:
        res["H"], res["TB"], res["AB"] = 1, 3, 1
    elif "二塁打" in line or "ツーベース" in line:
        res["H"], res["TB"], res["AB"] = 1, 2, 1
    elif "ヒット" in line or "安打" in line:
        res["H"], res["TB"], res["AB"] = 1, 1, 1
    # 3. 犠飛 (打数に含めないが出塁率計算に必要)
    elif "犠牲フライ" in line:
        res["SF"] = 1
    # 4. 凡退 (三振、ゴロ、フライ等は打数に含める)
    elif any(kw in line for kw in ["三振", "ゴロ", "フライ", "ライナー", "ダブルプレー", "併殺"]):
        res["AB"] = 1
        
    return res

# --- 累積データベースの管理 ---
class PlayerStatsDB:
    def __init__(self):
        self.db = {} # {選手名: {H: 0, AB: 0, ...}}

    def get_vector(self, name):
        s = self.db.get(name, {"H": 0, "TB": 0, "AB": 0, "BB": 0, "SF": 0})
        # 指標の算出 (0除算を避ける)
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

# --- メイン処理 ---
def build_dataset():
    stats_db = PlayerStatsDB()
    dataset = []

    # ファイルを日付順にソート (updated_at または ID を使用)
    file_paths = sorted(glob.glob(f"{JSON_DIR}/*.json"))

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            game = json.load(f)
        
        # 1. この試合の各チームのスタメンと得点を取得
        for team_info in game["scoreboard"]:
            team_name = team_info["team"]
            score = int(team_info["R"])
            
            # pregameデータからそのチームの打順(1-9番)を取得
            side = "先攻" if game["text_live"][0]["pregame"]["lineups"][0]["side"] == "先攻" else "後攻"
            # (実際にはチーム名でマッチングさせる処理が必要)
            
            # ここでは簡単のため、最初に見つけた方のスタメンをそのチームのものと仮定
            lineup = []
            for l in game["text_live"][0]["pregame"]["lineups"]:
                # JSON構造に合わせて適切なチームの選手リストを選択
                lineup = l["players"][:9] 
                
                # 指標ベクトルの生成 (試合開始前の累積データを使用) [cite: 15, 44]
                lineup_vectors = [stats_db.get_vector(name) for name in lineup]
                
                # 学習サンプルとして保存
                dataset.append({
                    "game_id": os.path.basename(path),
                    "team": team_name,
                    "lineup_vectors": lineup_vectors, # 9人分 × 4指標
                    "score": score # 教師ラベル [cite: 44]
                })
                break # 今回は1チーム分のみ例示

        # 2. 試合終了後、この試合の結果を累積DBに反映させる [cite: 24]
        for section in game["text_live"]:
            if "回" not in section["inning"]: continue
            for play in section["plays"]:
                # テキストから「打者名」と「結果」を抽出
                if not play["lines"]: continue
                match = re.match(r"\d番\s*([^\s]+)", play["lines"][0])
                if match:
                    player_name = match.group(1)
                    res = classify_play(" ".join(play["lines"][1:]))
                    stats_db.update(player_name, res)

    return dataset

# 実行
train_data = build_dataset()
print(f"作成されたサンプル数: {len(train_data)}")
# 結果の確認 (最初のサンプルの1番打者のベクトルと得点)
if train_data:
    print(f"Sample 0 Score: {train_data[0]['score']}")
    print(f"Sample 0 Lineup[0] Vector: {train_data[0]['lineup_vectors'][0]}")