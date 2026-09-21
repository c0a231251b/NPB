"""
add_window_7day_stats.pyの派生スクリプト
打率-被打率の差分を特徴量として付与するバージョンから出塁率-被出塁率の差分を特徴量として付与するバージョンに変更
その他実装はadd_window_7day_stats.pyと同じ
- game_data_2025_match_results/  (入力)
- game_data_2025_match_results_add_30day_stats_HR/  (出力)

このスクリプトを実行するコマンドは以下の通り
python add_window_30day_stats_HR.py --output_dir game_data_2025_match_results_add_30day_stats_HR game_data_2025_match_results
"""


"""
add_window_7day_stats.py の HR% 版
本塁打率 - 被本塁打率 を特徴量として付与する
"""

import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import argparse

WINDOW_DAYS = 30

# 本塁打判定
def is_home_run(result: str) -> bool:
    return "本" in result

# 打席判定（投手行以外はすべて PA として扱う）
def is_plate_appearance(result: str) -> bool:
    return True

# ============================================================
# 日付取得
# ============================================================
def get_game_date(filename: str):
    basename = Path(filename).stem
    m = re.search(r'(\d{4})(\d{2})(\d{2})\d{2,}$', basename)
    if m:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None

# ============================================================
# PitcherQueue（変更なし）
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
# 継投順復元（変更なし）
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
# 打席抽出（HR% 用イベント追加）
# ============================================================
def extract_at_bats(data):
    scoreboard = data.get('scoreboard', [])
    if len(scoreboard) < 2:
        return []

    visitor = scoreboard[0]['team']
    home = scoreboard[1]['team']

    omote_q, ura_q = build_pitcher_queues(data)
    omote_it = PitcherQueue(omote_q)
    ura_it = PitcherQueue(ura_q)

    at_bats = []

    for item in data.get('text_live', []):
        inning = item.get('inning', '')

        if '表' in inning:
            batting_team = visitor
            pitcher_it = omote_it
        elif '裏' in inning:
            batting_team = home
            pitcher_it = ura_it
        else:
            continue

        for play in item.get('plays', []):
            lines = play.get('lines', [])
            if len(lines) < 2:
                continue

            first = lines[0]
            if first.startswith("投手："):
                continue

            if not (('番' in first) or first.startswith('打')):
                continue

            if '番' in first:
                batter = first.split('番', 1)[1].strip()
            else:
                batter = first[1:].strip()

            result = lines[1].strip()
            pitcher = pitcher_it.current()
            if pitcher is None and pitcher_it.q:
                pitcher = pitcher_it.q[-1][0]

            at_bats.append({
                'inning': inning,
                'batting_team': batting_team,
                'pitcher': pitcher,
                'batter': batter,
                'result': result,
                'is_pa': is_plate_appearance(result),
                'is_hr': is_home_run(result),
            })

            pitcher_it.consume()

    return at_bats

# ============================================================
# 全試合ログ
# ============================================================
def build_log(json_dir: str):
    files = sorted(Path(json_dir).glob('*.json'))
    game_logs = []

    for f in files:
        date = get_game_date(f.name)
        if date is None:
            print(f"[WARN] 日付取得失敗: {f.name}")
            continue

        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            game_logs.append((date, str(f), data))
        except Exception as e:
            print(f"[WARN] 読み込み失敗: {f.name}: {e}")

    game_logs.sort(key=lambda x: x[0])
    season_start = game_logs[0][0] if game_logs else None
    return game_logs, season_start

# ============================================================
# RollingStats（HR% 版）
# ============================================================
class RollingStats:

    def __init__(self, window_days=30):
        self.window = window_days
        self.batter_log = defaultdict(list)
        self.pitcher_log = defaultdict(list)

    def add_game(self, game_date, at_bats):
        for ab in at_bats:
            entry = (
                game_date,
                ab['is_pa'],
                ab['is_hr']
            )
            if ab['batter']:
                self.batter_log[ab['batter']].append(entry)
            if ab['pitcher']:
                self.pitcher_log[ab['pitcher']].append(entry)

    def _calc_hr_rate(self, log, before_date):
        cutoff = before_date - timedelta(days=self.window)

        PA = 0
        HR = 0

        for (d, is_pa, is_hr) in log:
            if cutoff <= d < before_date:
                if is_pa:
                    PA += 1
                if is_hr:
                    HR += 1

        if PA == 0:
            return None

        return round(HR / PA, 4)

    def batter_hr_rate(self, name, game_date):
        return self._calc_hr_rate(self.batter_log.get(name, []), game_date)

    def pitcher_hr_rate(self, name, game_date):
        return self._calc_hr_rate(self.pitcher_log.get(name, []), game_date)

# ============================================================
# メイン処理（HR% 版）
# ============================================================
def process(json_dir, output_dir=None, inplace=False):

    print(f"[INFO] 入力ディレクトリ: {json_dir}")

    game_logs, season_start = build_log(json_dir)
    if not game_logs:
        print("[ERROR] JSONファイルなし")
        return

    stats_unlock_date = season_start + timedelta(days=WINDOW_DAYS)
    rolling = RollingStats(WINDOW_DAYS)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, (game_date, filepath, data) in enumerate(game_logs):

        filename = Path(filepath).name
        at_bats = extract_at_bats(data)

        use_stats = (game_date >= stats_unlock_date)

        features = []

        for ab in at_bats:

            if use_stats:
                b_hr = rolling.batter_hr_rate(ab['batter'], game_date)
                p_hr = rolling.pitcher_hr_rate(ab['pitcher'], game_date) if ab['pitcher'] else None

                if b_hr is not None and p_hr is not None:
                    contrast = round(b_hr - p_hr, 4)
                else:
                    contrast = 0.0
            else:
                b_hr = None
                p_hr = None
                contrast = 0.0

            features.append({
                'inning': ab['inning'],
                'batter': ab['batter'],
                'pitcher': ab['pitcher'],
                'batting_team': ab['batting_team'],
                'batter_hr_30d': b_hr,
                'pitcher_hr_30d': p_hr,
                'contrast_feature': contrast,
            })

        data['at_bat_features'] = features
        rolling.add_game(game_date, at_bats)

        out_path = filepath if inplace else str(Path(output_dir) / filename)

        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

        print(f"[{i+1}/{len(game_logs)}] {filename} | 打席数: {len(features)}")

    print("\n[INFO] 完了")

# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--inplace', action='store_true')
    args = parser.parse_args()

    if not args.output_dir and not args.inplace:
        parser.error('--output_dir か --inplace を指定してください')

    process(args.input_dir, args.output_dir, args.inplace)
