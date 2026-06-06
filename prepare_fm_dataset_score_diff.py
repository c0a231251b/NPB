import json
import pandas as pd
import glob
import unicodedata
import re

# ==================================================
# 変更点: 目的変数「得点差」への刷新 & 出力ファイル名変更
# スクリプト名: prepare_fm_dataset_score_diff.py
# ==================================================

def aggressive_normalize(s):
    if not s: return ""
    s = unicodedata.normalize('NFKC', str(s)).replace(" ", "").replace(" ", "").strip()
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

def prepare_fm_dataset_score_diff():
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    norm_id_map = {aggressive_normalize(name): vid for name, vid in name_to_id.items()}

    b_df = pd.read_csv("initial_stats_2024.csv")
    p_df = pd.read_csv("pitcher_stats_2024_all.csv")
    p_names_all = set(aggressive_normalize(n) for n in p_df['name'])

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
    failed_names = [] # ID 0 になった選手を溜めるリスト
    json_paths = glob.glob("game_data_2025/*.json")
    print(f"--- 得点差ベースの前処理: {len(json_paths)}試合をスキャン中 ---")
    
    for path in json_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        scores = {standardize_team(e["team"]): int(aggressive_normalize(e.get("R", "0"))) for e in data.get("scoreboard", [])}
        if len(scores) < 2: continue
        teams = list(scores.keys())
        lineup_detect = {teams[0]: [None]*10, teams[1]: [None]*10}

        for entry in data.get("text_live", []):
            inning = entry.get("inning", "")
            if not inning: continue
            bat_t, pit_t = (teams[0], teams[1]) if "表" in inning else (teams[1], teams[0])
            for play in entry.get("plays", []):
                for line in play.get("lines", []):
                    match = re.search(r"([1-9])番\s*(\S+)", line)
                    if match:
                        order, short = int(match.group(1)), aggressive_normalize(match.group(2))
                        if lineup_detect[bat_t][order] is None: lineup_detect[bat_t][order] = short
                        if order == 9:
                            is_p = short in p_names_all or any(short in pn for pn in p_names_all)
                            if is_p and lineup_detect[pit_t][0] is None: lineup_detect[pit_t][0] = short
                    p_match = re.search(r"投[手]?[：\s]+(\S+)", line)
                    if p_match:
                        p_name = aggressive_normalize(p_match.group(1))
                        if lineup_detect[pit_t][0] is None: lineup_detect[pit_t][0] = p_name

        def resolve_id(t, s, role_label):
            if not s: return 0
            full = team_short_resolver.get((t, s))
            if not full: full = global_short_resolver.get(s)
            if not full:
                for norm_full in norm_id_map.keys():
                    if s in norm_full:
                        full = norm_full
                        break
            
            p_id = norm_id_map.get(full, 0)
            if p_id == 0:
                failed_names.append(f"{role_label}: {t} {s}")
            return p_id

        for t_name in teams:
            # 相手チーム（オポジション）の特定
            opp_t = teams[1] if t_name == teams[0] else teams[0]
            
            # 打者と投手の解決
            b_ids = [resolve_id(t_name, b, "打者") for b in lineup_detect[t_name][1:10]]
            p_id = resolve_id(opp_t, lineup_detect[opp_t][0], "投手")
            
            # 最重要変更】targetの計算を「自チーム得点 － 相手チーム得点」の得点差に刷新
            score_difference = float(scores[t_name] - scores[opp_t])
            
            dataset.append({
                "cat_features": b_ids + [p_id], 
                "num_features": [1.0]*49, 
                "target": score_difference
            })

    # デバッグ出力
    output_failed_txt = "matching_failed_score_diff.txt"
    with open(output_failed_txt, "w", encoding="utf-8") as f:
        if failed_names:
            f.write("\n".join(sorted(list(set(failed_names)))))
        else:
            f.write("すべての選手が特定されました！")

    # 【変更】得点差モデル専用のデータセット名で保存
    output_pkl = "fm_dataset_score_diff.pkl"
    pd.DataFrame(dataset).to_pickle(output_pkl)
    print(f" 完了! 得点差データセット '{output_pkl}' を生成しました。")

if __name__ == "__main__":
    prepare_fm_dataset_score_diff()