#============================================================
# prepare_fm_dataset_30day.pyの派生、prepare_fm_dataset_30day_OPS.pyを作成
# OPS - 被OPS（30日間）を特徴量として使用するバージョン
# 目的変数：その試合の得点
# 出力：fm_dataset_30d_OPS.pkl
#============================================================



import os
import numpy as np
import json
import pandas as pd
import glob
import unicodedata
import re

#============================================================
# 正規化関数
#============================================================
def aggressive_normalize(s):
    if not s:
        return ""
    s = unicodedata.normalize('NFKC', str(s)).replace(" ", "").strip()
    kanji_map = {
        '崎': '崎', '﨑': '崎', '辺': '辺', '邊': '辺', '邉': '辺',
        '斉': '斉', '齊': '斉', '齋': '斉', '斎': '斉',
        '高': '高', '髙': '高', '柳': '柳', '栁': '柳',
        'ケ': 'ケ', 'ヶ': 'ケ', '祥': '祥'
    }
    for k, v in kanji_map.items():
        s = s.replace(k, v)
    return s

def standardize_team(name):
    n = aggressive_normalize(name)
    team_map = {
        "G": "巨人", "読売": "巨人", "巨人": "巨人",
        "S": "ヤクルト", "ヤクルト": "ヤクルト",
        "DB": "DeNA", "YB": "DeNA", "DeNA": "DeNA", "ＤｅＮＡ": "DeNA",
        "T": "阪神", "阪神": "阪神",
        "C": "広島", "広島": "広島",
        "D": "中日", "中日": "中日",
        "H": "ソフトバンク", "ソフトバンク": "ソフトバンク",
        "F": "日本ハム", "日本ハム": "日本ハム",
        "M": "ロッテ", "ロッテ": "ロッテ",
        "B": "オリックス", "オリックス": "オリックス",
        "E": "楽天", "楽天": "楽天",
        "L": "西武", "西武": "西武"
    }
    return team_map.get(n, n)

#============================================================
# メイン処理（30day OPS 専用）
#============================================================
def prepare_fm_dataset_30d_OPS():

    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)

    # ★ OPS 版の JSON ディレクトリ
    JSON_DIR_30D = "game_data_2025_match_results_add_30day_stats_OPS"

    name_to_id = master["name_to_id"]
    norm_id_map = {aggressive_normalize(name): vid for name, vid in name_to_id.items()}

    # 初期データ
    b_df = pd.read_csv("initial_stats_2024.csv")
    p_df = pd.read_csv("pitcher_stats_2024_all.csv")
    p_names_all = set(aggressive_normalize(n) for n in p_df['name'])

    # 名前解決辞書
    team_short_resolver = {}
    global_short_resolver = {}

    for df in [b_df, p_df]:
        for _, row in df.iterrows():
            t = standardize_team(row['team'])
            full = aggressive_normalize(row['name'])
            for i in range(1, len(full) + 1):
                prefix = full[:i]
                team_short_resolver[(t, prefix)] = full
                if prefix not in global_short_resolver or len(full) > len(global_short_resolver[prefix]):
                    global_short_resolver[prefix] = full

    dataset = []
    failed_names = []

    json_paths = glob.glob(os.path.join(JSON_DIR_30D, "*.json"))
    print(f"--- 30day OPS データセット生成: {len(json_paths)}試合を処理中 ---")

    #============================================================
    # 試合ごとの処理
    #============================================================
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # スコア
        scores = {standardize_team(e["team"]): int(aggressive_normalize(e.get("R", "0")))
                  for e in data.get("scoreboard", [])}
        if len(scores) < 2:
            continue

        teams = list(scores.keys())

        # lineup 検出
        lineup_detect = {teams[0]: [None]*10, teams[1]: [None]*10}

        for entry in data.get("text_live", []):
            inning = entry.get("inning", "")
            if not inning:
                continue

            bat_t, pit_t = (teams[0], teams[1]) if "表" in inning else (teams[1], teams[0])

            for play in entry.get("plays", []):
                for line in play.get("lines", []):
                    match = re.search(r"([1-9])番\s*(\S+)", line)
                    if match:
                        order = int(match.group(1))
                        short = aggressive_normalize(match.group(2))
                        if lineup_detect[bat_t][order] is None:
                            lineup_detect[bat_t][order] = short

                        if order == 9:
                            is_p = short in p_names_all or any(short in pn for pn in p_names_all)
                            if is_p and lineup_detect[pit_t][0] is None:
                                lineup_detect[pit_t][0] = short

                    p_match = re.search(r"投[手]?[：\s]+(\S+)", line)
                    if p_match:
                        p_name = aggressive_normalize(p_match.group(1))
                        if lineup_detect[pit_t][0] is None:
                            lineup_detect[pit_t][0] = p_name

        #============================================================
        # 30day OPS 特徴量
        #============================================================
        team_contrast30 = {teams[0]: [], teams[1]: []}
        seen = set()

        for ab in data.get("at_bat_features", []):
            batter = aggressive_normalize(ab.get("batter"))
            team = standardize_team(ab.get("batting_team"))
            key = (team, batter)

            if key in seen:
                continue
            seen.add(key)

            contrast = ab.get("contrast_feature", 0.0)
            team_contrast30[team].append(float(contrast))

        #============================================================
        # ID 解決
        #============================================================
        def resolve_id(t, s, role_label):
            if not s:
                return 0
            full = team_short_resolver.get((t, s))
            if not full:
                full = global_short_resolver.get(s)
            if not full:
                for norm_full in norm_id_map.keys():
                    if s in norm_full:
                        full = norm_full
                        break
            p_id = norm_id_map.get(full, 0)
            if p_id == 0:
                failed_names.append(f"{role_label}: {t} {s}")
            return p_id

        #============================================================
        # データセット構築
        #============================================================
        for t_name in teams:
            opp_t = teams[1] if t_name == teams[0] else teams[0]

            b_ids = [
                resolve_id(t_name, b, "打者")
                for b in lineup_detect[t_name][1:10]
            ]

            p_id = resolve_id(opp_t, lineup_detect[opp_t][0], "投手")

            c_30 = team_contrast30[t_name]
            if not c_30:
                c_30 = [0.0]

            num_features = [
                float(np.mean(c_30)),
                float(np.std(c_30))
            ]

            dataset.append({
                "cat_features": b_ids + [p_id],
                "num_features": num_features,
                "target": float(scores[t_name])
            })

    #============================================================
    # 出力
    #============================================================
    with open("matching_failed_30d_OPS.txt", "w", encoding="utf-8") as f:
        if failed_names:
            f.write("\n".join(sorted(set(failed_names))))
        else:
            f.write("すべての選手が特定されました！")

    pd.DataFrame(dataset).to_pickle("fm_dataset_30d_OPS.pkl")
    print("完了! 'fm_dataset_30d_OPS.pkl' を生成しました。")


if __name__ == "__main__":
    prepare_fm_dataset_30d_OPS()
