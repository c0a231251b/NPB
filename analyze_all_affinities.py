import torch
import torch.nn as nn
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49):
        super(FactorizationMachineModel, self).__init__()
        self.player_v = nn.Embedding(num_players, k)

def get_team_code(name):
    """チーム名をアルファベット略称に変換"""
    name = str(name).strip()
    mapping = {
        # セ・リーグ
        "巨人": "G", "読売": "G", "G": "G",
        "阪神": "T", "T": "T",
        "DeNA": "DB", "ＤｅＮＡ": "DB", "横浜": "DB", "DB": "DB", "YB": "DB",
        "広島": "C", "C": "C",
        "ヤクルト": "S", "S": "S",
        "中日": "D", "D": "D",
        # パ・リーグ
        "ソフトバンク": "H", "H": "H",
        "ロッテ": "M", "M": "M",
        "日本ハム": "F", "F": "F",
        "西武": "L", "L": "L",
        "楽天": "E", "E": "E",
        "オリックス": "B", "B": "B"
    }
    return mapping.get(name, name)

def analyze_all_to_txt():
    # 1. データのロード
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}

    # 2. チーム情報の収集
    h_df = pd.read_csv("initial_stats_2024.csv")
    p_df = pd.read_csv("pitcher_stats_2024_all.csv")
    
    player_to_team = {}
    for _, row in h_df.iterrows():
        player_to_team[row['name']] = get_team_code(row['team'])
    for _, row in p_df.iterrows():
        player_to_team[row['name']] = get_team_code(row['team'])

    # 3. 投手と打者のリストアップ
    all_pitchers = [n for n in p_df['name'].unique() if n in name_to_id]
    all_hitters = [n for n in h_df['name'].unique() if n in name_to_id]

    # 4. モデルのロード
    model = FactorizationMachineModel(num_players=697)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    v = model.player_v.weight.detach().numpy()

    # 5. 相性計算とファイル保存
    output_path = "all_player_affinities_standardized.txt"
    print(f"--- チーム名を統一して相性スコアを計算中 (保存先: {output_path}) ---")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("NPB 2025 相性分析レポート (Standardized Team Codes)\n")
        f.write("====================================================\n\n")

        for hitter_name in tqdm(all_hitters):
            h_id = name_to_id[hitter_name]
            h_vec = v[h_id]
            h_team = player_to_team.get(hitter_name, "??")

            f.write(f"【打者】{hitter_name} ({h_team})\n")
            
            affinities = []
            for pitcher_name in all_pitchers:
                p_team = player_to_team.get(pitcher_name, "??")
                
                # 自チーム同士、または不明な場合はスキップ
                if h_team == p_team or h_team == "??" or p_team == "??":
                    continue
                
                p_id = name_to_id[pitcher_id := pitcher_name] # ID取得
                p_id = name_to_id[pitcher_name]
                p_vec = v[p_id]
                
                score = np.dot(h_vec, p_vec)
                affinities.append((pitcher_name, p_team, score))

            # 相性スコア順にソート
            affinities.sort(key=lambda x: x[2], reverse=True)
            
            for p_name, p_team, score in affinities:
                # チーム名を2文字幅で左寄せにすることで、縦のラインを揃える
                f.write(f"  vs {p_name:12} ({p_team:2}) | Score: {score:10.6f}\n")
            
            f.write("-" * 60 + "\n")

    print(f"\n統一完了！ {output_path} を確認してください。")

if __name__ == "__main__":
    analyze_all_to_txt()