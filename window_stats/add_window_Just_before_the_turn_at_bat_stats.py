"""
add_window_Just_before_the_turn_at_bat_stats.py
打席直前までの累積成績を付与するスクリプト

入力:
    game_data_2025_match_results/
出力:
    game_data_2025_match_results_Just_before_the_turn_at_bat_stats/

付与する特徴量:
    batter_avg      : 打席直前までのシーズン累積打率
    batter_ops      : 打席直前までのシーズン累積OPS
    pitcher_era     : 打席直前までのシーズン累積ERA
    pitcher_k9      : 打席直前までのシーズン累積K/9
    vs_ops          : 打席直前までの対戦OPS（打者 vs 投手）


コマンド：
python add_window_Just_before_the_turn_at_bat_stats.py game_data_2025_match_results --output_dir game_data_2025_match_results_Just_before_the_turn_at_bat_stats
"""

import json
import os
import re
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import argparse

# ============================================================
# 日付取得
# ============================================================
def get_game_date(filename: str):
    m = re.search(r'(\d{4})(\d{2})(\d{2})', filename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None

# ============================================================
# PitcherQueue（K%版と同じ）
# ============================================================
class PitcherQueue:
    def __init__(self, queue):
        self.q = queue
        self.idx = 0
        self.used = 0

    def current(self):
        if self.idx < len(self.q):
            return self.q[self.idx][0]
        if self.q:
            return self.q[-1][0]
        return None

    def consume(self):
        if self.idx >= len(self.q):
            return
        self.used += 1
        bf = self.q[self.idx][1]
        if self.used >= bf:
            self.idx += 1
            self.used = 0


# ============================================================
# 継投順復元（K%版と同じ）
# ============================================================
def build_pitcher_queues(data):
    pitcher_stats = data.get("pitcher_stats", {})
    pitcher_order = list(pitcher_stats.keys())
    starting = data.get("starting_pitchers", {})
    omote = starting.get("表")
    ura = starting.get("裏")

    if not omote or not ura:
        return [], []

    try:
        omote_idx = pitcher_order.index(omote)
        ura_idx = pitcher_order.index(ura)
    except ValueError:
        return [], []

    if ura_idx < omote_idx:
        ura_names = pitcher_order[ura_idx:omote_idx]
        omote_names = pitcher_order[omote_idx:]
    else:
        omote_names = pitcher_order[omote_idx:ura_idx]
        ura_names = pitcher_order[ura_idx:]

    omote_queue = [(n, pitcher_stats[n].get("bf", 0)) for n in omote_names if pitcher_stats[n].get("bf", 0) > 0]
    ura_queue = [(n, pitcher_stats[n].get("bf", 0)) for n in ura_names if pitcher_stats[n].get("bf", 0) > 0]

    return omote_queue, ura_queue


# ============================================================
# 打席抽出（結果・三振判定）
# ============================================================
def extract_at_bats(data):
    scoreboard = data.get("scoreboard", [])
    if len(scoreboard) < 2:
        return []

    visitor = scoreboard[0]["team"]
    home = scoreboard[1]["team"]

    # ★ 投手キューを構築
    omote_q, ura_q = build_pitcher_queues(data)
    omote_it = PitcherQueue(omote_q)
    ura_it = PitcherQueue(ura_q)

    at_bats = []

    for item in data.get("text_live", []):
        inning = item.get("inning", "")

        if "表" in inning:
            batting_team = visitor
            pitcher_it = omote_it
        elif "裏" in inning:
            batting_team = home
            pitcher_it = ura_it
        else:
            continue

        for play in item.get("plays", []):
            lines = play.get("lines", [])
            if len(lines) < 2:
                continue

            first = lines[0]
            if first.startswith("投手："):
                continue

            if not (("番" in first) or first.startswith("打")):
                continue

            if "番" in first:
                batter = first.split("番", 1)[1].strip()
            else:
                batter = first[1:].strip()

            result = lines[1].strip()

            # ★ 投手名を PitcherQueue から取得
            pitcher = pitcher_it.current()
            if pitcher is None and pitcher_it.q:
                pitcher = pitcher_it.q[-1][0]

            # 三振判定
            is_k = ("三振" in result)

            # 安打判定
            is_hit = ("安" in result or "本" in result or "２" in result or "３" in result)

            # 四死球判定
            is_bb = ("四球" in result or "敬遠" in result)
            is_hbp = ("死球" in result)

            # 打席扱い
            is_ab = not (is_bb or is_hbp or "犠" in result)

            # 塁打数
            bases = 0
            if "本" in result:
                bases = 4
            elif "３" in result:
                bases = 3
            elif "２" in result:
                bases = 2
            elif "安" in result:
                bases = 1

            at_bats.append({
                "inning": inning,
                "batting_team": batting_team,
                "batter": batter,
                "pitcher": pitcher,
                "result": result,
                "is_ab": is_ab,
                "is_hit": is_hit,
                "is_bb": is_bb,
                "is_hbp": is_hbp,
                "is_k": is_k,
                "bases": bases,
            })

            # ★ 投手の打者対戦数を消費
            pitcher_it.consume()

    return at_bats


# ============================================================
# RollingStats（累積成績）
# ============================================================
class RollingStats:

    def __init__(self):
        self.batter_log = defaultdict(list)
        self.pitcher_log = defaultdict(list)
        self.vs_log = defaultdict(lambda: defaultdict(list))

    def add_ab(self, ab):
        batter = ab["batter"]
        pitcher = ab["pitcher"]

        # 打者ログ
        self.batter_log[batter].append(ab)

        # 投手ログ
        self.pitcher_log[pitcher].append(ab)

        # 対戦ログ
        self.vs_log[batter][pitcher].append(ab)

    # -------------------------
    # 打者成績
    # -------------------------
    def batter_avg(self, name):
        log = self.batter_log.get(name, [])
        AB = H = 0
        for ab in log:
            if ab["is_ab"]:
                AB += 1
                if ab["is_hit"]:
                    H += 1
        return round(H / AB, 4) if AB > 0 else None

    def batter_ops(self, name):
        log = self.batter_log.get(name, [])
        AB = H = BB = HBP = TB = 0
        for ab in log:
            if ab["is_ab"]:
                AB += 1
                TB += ab["bases"]
                if ab["is_hit"]:
                    H += 1
            if ab["is_bb"]:
                BB += 1
            if ab["is_hbp"]:
                HBP += 1

        PA = AB + BB + HBP
        if PA == 0 or AB == 0:
            return None

        obp = (H + BB + HBP) / PA
        slg = TB / AB
        return round(obp + slg, 4)

    # -------------------------
    # 投手成績
    # -------------------------
    def pitcher_era(self, name):
        log = self.pitcher_log.get(name, [])
        BF = H = BB = HBP = 0
        for ab in log:
            BF += 1
            if ab["is_hit"]:
                H += 1
            if ab["is_bb"]:
                BB += 1
            if ab["is_hbp"]:
                HBP += 1

        if BF == 0:
            return None

        # 簡易的に失点 = H + BB + HBP とする
        R = H + BB + HBP
        IP = BF / 3
        if IP == 0:
            return None
        return round(9 * R / IP, 4)

    def pitcher_k9(self, name):
        log = self.pitcher_log.get(name, [])
        BF = K = 0
        for ab in log:
            BF += 1
            if ab["is_k"]:
                K += 1
        IP = BF / 3
        if IP == 0:
            return None
        return round(9 * K / IP, 4)

    # -------------------------
    # 対戦 OPS
    # -------------------------
    def vs_ops(self, batter, pitcher):
        log = self.vs_log[batter].get(pitcher, [])
        AB = H = BB = HBP = TB = 0
        for ab in log:
            if ab["is_ab"]:
                AB += 1
                TB += ab["bases"]
                if ab["is_hit"]:
                    H += 1
            if ab["is_bb"]:
                BB += 1
            if ab["is_hbp"]:
                HBP += 1

        PA = AB + BB + HBP
        if PA == 0 or AB == 0:
            return None

        obp = (H + BB + HBP) / PA
        slg = TB / AB
        return round(obp + slg, 4)
    
        # -------------------------
    # 打者 OBP
    # -------------------------
    def batter_obp(self, name):
        log = self.batter_log.get(name, [])
        AB = H = BB = HBP = 0
        for ab in log:
            if ab["is_ab"]:
                AB += 1
                if ab["is_hit"]:
                    H += 1
            if ab["is_bb"]:
                BB += 1
            if ab["is_hbp"]:
                HBP += 1

        PA = AB + BB + HBP
        if PA == 0:
            return None

        return round((H + BB + HBP) / PA, 4)

    # -------------------------
    # 打者 SLG
    # -------------------------
    def batter_slg(self, name):
        log = self.batter_log.get(name, [])
        AB = TB = 0
        for ab in log:
            if ab["is_ab"]:
                AB += 1
                TB += ab["bases"]

        if AB == 0:
            return None

        return round(TB / AB, 4)

    # -------------------------
    # 打者 ISO = SLG - AVG
    # -------------------------
    def batter_iso(self, name):
        slg = self.batter_slg(name)
        avg = self.batter_avg(name)
        if slg is None or avg is None:
            return None
        return round(slg - avg, 4)

    # -------------------------
    # 投手 WHIP
    # -------------------------
    def pitcher_whip(self, name):
        log = self.pitcher_log.get(name, [])
        H = BB = 0
        BF = 0
        for ab in log:
            BF += 1
            if ab["is_hit"]:
                H += 1
            if ab["is_bb"]:
                BB += 1

        IP = BF / 3
        if IP == 0:
            return None

        return round((H + BB) / IP, 4)

    # -------------------------
    # 投手 被OPS
    # -------------------------
    def pitcher_ops(self, name):
        log = self.pitcher_log.get(name, [])
        AB = H = BB = HBP = TB = 0

        for ab in log:
            if ab["is_ab"]:
                AB += 1
                TB += ab["bases"]
                if ab["is_hit"]:
                    H += 1
            if ab["is_bb"]:
                BB += 1
            if ab["is_hbp"]:
                HBP += 1

        PA = AB + BB + HBP
        if PA == 0 or AB == 0:
            return None

        obp = (H + BB + HBP) / PA
        slg = TB / AB
        return round(obp + slg, 4)


# ============================================================
# メイン処理
# ============================================================
def process(json_dir, output_dir):

    files = sorted(Path(json_dir).glob("*.json"))
    stats = RollingStats()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        at_bats = extract_at_bats(data)
        features = []

        for ab in at_bats:

            b_avg = stats.batter_avg(ab["batter"])
            b_ops = stats.batter_ops(ab["batter"])
            p_era = stats.pitcher_era(ab["pitcher"])
            p_k9 = stats.pitcher_k9(ab["pitcher"])
            v_ops = stats.vs_ops(ab["batter"], ab["pitcher"])
            b_obp = stats.batter_obp(ab["batter"])
            b_slg = stats.batter_slg(ab["batter"])
            b_iso = stats.batter_iso(ab["batter"])
            p_whip = stats.pitcher_whip(ab["pitcher"])
            p_ops = stats.pitcher_ops(ab["pitcher"])


            features.append({
                "inning": ab["inning"],
                "batter": ab["batter"],
                "pitcher": ab["pitcher"],
                "batting_team": ab["batting_team"],
                "batter_avg": b_avg,
                "batter_ops": b_ops,
                "pitcher_era": p_era,
                "pitcher_k9": p_k9,
                "vs_ops": v_ops,
                "batter_obp": b_obp,
                "batter_slg": b_slg,
                "batter_iso": b_iso,
                "pitcher_whip": p_whip,
                "pitcher_ops": p_ops,
            })

            # 打席をログに追加（次の打席のため）
            stats.add_ab(ab)

        data["at_bat_features"] = features

        out_path = Path(output_dir) / f.name
        with open(out_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

        print(f"[{i+1}/{len(files)}] {f.name} | 打席数: {len(features)}")

    print("\n[INFO] 完了")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    process(args.input_dir, args.output_dir)
