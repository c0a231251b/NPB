import pandas as pd
import json
import glob
import os

def create_player_id_master():
    # 1. データの読み込み
    batter_csv = "initial_stats_2024.csv"
    pitcher_csv = "pitcher_stats_2024_all.csv"
    json_dir = "game_data_2025"

    all_names = set()

    # CSVから名前を抽出
    if os.path.exists(batter_csv):
        b_df = pd.read_csv(batter_csv)
        all_names.update(b_df['name'].unique())
    
    if os.path.exists(pitcher_csv):
        p_df = pd.read_csv(pitcher_csv)
        all_names.update(p_df['name'].unique())

    # 試合データ(JSON)からも名前を抽出（CSVにいない新戦力の漏れ防止）
    json_paths = glob.glob(os.path.join(json_dir, "*.json"))
    # JSON（2025年全試合）から、CSVに載っていない新戦力も拾い上げる処理
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # 1. 投手名の取得 (修正案)
            pregame = data.get('pregame', {})
            p_name = pregame.get('pitcher')
            if p_name:
                all_names.add(p_name)
            # 2. 打者名の取得（出場した全選手を拾う）
            for player in data.get('home_team_players', []) + data.get('away_team_players', []):
                all_names.add(player['name'])

    # 2. ソートしてIDを割り振り
    sorted_names = sorted(list(all_names))
    name_to_id = {name: i for i, name in enumerate(sorted_names)}
    id_to_name = {i: name for i, name in enumerate(sorted_names)}

    # 3. JSONとして保存（他のスクリプトで使いやすくするため）
    master_data = {
        "name_to_id": name_to_id,
        "id_to_name": id_to_name,
        "total_players": len(sorted_names)
    }

    with open("player_id_master.json", "w", encoding="utf-8") as f:
        json.dump(master_data, f, ensure_ascii=False, indent=4)

    print(f"選手IDマスターを作成しました。")
    print(f"総選手数: {len(sorted_names)} 名")
    print(f"例: {sorted_names[0]} -> 0, {sorted_names[-1]} -> {len(sorted_names)-1}")

if __name__ == "__main__":
    create_player_id_master()