"""
パターン2: 直近1か月（30日）スタッツによる対比コンテキスト特徴量付与スクリプト
修正版（投手継投対応版）

修正内容:
- pitcher_stats をチーム名判定で分けるのを廃止
- starting_pitchers + pitcher_stats順で継投復元
- pitcher:null 大量発生問題を修正
- 継投キュー検証追加
- 打席数 vs BF総和 検証追加
"""

import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path
import argparse
from collections import Counter


# ============================================================
# 定数
# ============================================================
WINDOW_DAYS = 30

HIT_PATTERNS = ['安', '２', '３', '本']
NON_AB_PATTERNS = ['四球', '死球', '犠', '敬遠', 'ギ']


def is_hit(result: str) -> bool:
    return any(p in result for p in HIT_PATTERNS)


def is_at_bat(result: str) -> bool:
    return not any(p in result for p in NON_AB_PATTERNS)


# ============================================================
# 日付取得
# ============================================================
def get_game_date(filename: str) -> datetime | None:

    basename = Path(filename).stem

    m = re.search(r'(\d{4})(\d{2})(\d{2})\d{2,}$', basename)

    if m:
        return datetime(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3))
        )

    return None


# ============================================================
# 投手キュー
# ============================================================
class PitcherQueue:

    def __init__(self, queue: list[tuple[str, int]]):

        self.q = queue
        self.idx = 0
        self.used = 0

    def current(self) -> str | None:

        # 通常
        if self.idx < len(self.q):
            return self.q[self.idx][0]

        # bf超過時は最後の投手を返す
        if self.q:
            return self.q[-1][0]

        return None

    def consume(self):

        if self.idx >= len(self.q):
            return

        self.used += 1

        bf = self.q[self.idx][1]

        if self.used >= bf:

            old_pitcher = self.q[self.idx][0]

            if self.idx + 1 < len(self.q):
                new_pitcher = self.q[self.idx + 1][0]

                print(
                    f"[PITCHER_CHANGE] "
                    f"{old_pitcher} -> {new_pitcher}"
                )

            self.idx += 1
            self.used = 0



# ============================================================
# 継投順復元
# ============================================================
def build_pitcher_queues(data: dict):

    pitcher_stats = data.get("pitcher_stats", {})
    pitcher_order = list(pitcher_stats.keys())

    starting_pitchers = data.get("starting_pitchers", {})

    omote_starter = starting_pitchers.get("表")
    ura_starter = starting_pitchers.get("裏")

    if not omote_starter or not ura_starter:
        return [], []

    try:
        omote_idx = pitcher_order.index(omote_starter)
        ura_idx = pitcher_order.index(ura_starter)

    except ValueError:
        return [], []

    # ========================================================
    # pitcher_stats順:
    # [ジャクソン, 宮城, ウィック, 颯, ランバート, 山本...]
    #
    # のような構造を想定
    # ========================================================

    if ura_idx < omote_idx:

        # 裏投手群
        ura_names = pitcher_order[ura_idx:omote_idx]

        # 表投手群
        omote_names = pitcher_order[omote_idx:]

    else:

        omote_names = pitcher_order[omote_idx:ura_idx]
        ura_names = pitcher_order[ura_idx:]

    omote_queue = []
    ura_queue = []

    for name in omote_names:

        bf = pitcher_stats[name].get("bf", 0)

        if bf > 0:
            omote_queue.append((name, bf))

    for name in ura_names:

        bf = pitcher_stats[name].get("bf", 0)

        if bf > 0:
            ura_queue.append((name, bf))
    
    return omote_queue, ura_queue


# ============================================================
# 打席抽出
# ============================================================
def extract_at_bats(data: dict) -> list[dict]:

    scoreboard = data.get('scoreboard', [])

    if len(scoreboard) < 2:
        return []

    visitor_team = scoreboard[0]['team']
    home_team = scoreboard[1]['team']

    # ========================================================
    # 継投順復元
    # ========================================================
    omote_queue, ura_queue = build_pitcher_queues(data)

    omote_it = PitcherQueue(omote_queue)
    ura_it = PitcherQueue(ura_queue)

    at_bats = []

    for item in data.get('text_live', []):

        inning = item.get('inning', '')

        if '表' in inning:

            batting_team = visitor_team
            pitcher_it = omote_it

        elif '裏' in inning:

            batting_team = home_team
            pitcher_it = ura_it

        else:
            continue

        for play in item.get('plays', []):

            lines = play.get('lines', [])

            if not lines or len(lines) < 2:
                continue

            first = lines[0]

            # ====================================================
            # 投手行スキップ
            # ====================================================
            if first.startswith("投手："):
                continue

            # ====================================================
            # 打席判定
            # ====================================================
            is_batter_line = (
                ('番' in first)
                or first.startswith('打')
            )

            if not is_batter_line:
                continue

            # ====================================================
            # 打者名
            # ====================================================
            if '番' in first:
                batter_name = first.split('番', 1)[1].strip()

            else:
                batter_name = first[1:].strip()

            result = lines[1].strip()


            pitcher = pitcher_it.current()




            

            

            # キュー超過時は最後の投手を維持
            if pitcher is None and pitcher_it.q:
                pitcher = pitcher_it.q[-1][0]


            
            
            

            at_bats.append({
                'inning': inning,
                'batting_team': batting_team,
                'pitcher': pitcher,
                'batter': batter_name,
                'result': result,
                'is_hit': is_hit(result),
                'is_ab': is_at_bat(result),
            })

            
            

            # ====================================================
            # bf消費
            # ====================================================
            pitcher_it.consume()
            
            

            

    return at_bats


def debug_problem_game(data, at_bats, filename):

    print("\n" + "=" * 120)
    print("DEBUG:", filename)
    print("=" * 120)

    official_bf = sum(
        p.get("bf", 0)
        for p in data["pitcher_stats"].values()
    )

    print(f"公式BF合計 : {official_bf}")
    print(f"抽出打席数 : {len(at_bats)}")
    print(f"差分       : {official_bf - len(at_bats)}")

    print("\n【投手別BF確認】")

    actual = Counter()

    for ab in at_bats:
        actual[ab["pitcher"]] += 1

    for pitcher, stats in data["pitcher_stats"].items():

        official = stats.get("bf", 0)
        extracted = actual[pitcher]

        mark = ""

        if official != extracted:
            mark = " ← MISMATCH"

        print(
            f"{pitcher:10s} "
            f"official={official:3d} "
            f"extracted={extracted:3d}"
            f"{mark}"
        )

    print("\n【抽出された全打席】")

    for i, ab in enumerate(at_bats, start=1):

        print(
            f"{i:03d}",
            f"{ab['inning']:6s}",
            f"{ab['batter']:10s}",
            f"{ab['pitcher']:10s}",
            ab['result']
        )

    print("\n【extract_at_bats() が無視した行】")

    for item in data.get("text_live", []):

        inning = item.get("inning", "")

        for play in item.get("plays", []):

            lines = play.get("lines", [])

            if not lines:
                continue

            first = lines[0]

            if "番" not in first:

                print(
                    "[SKIPPED]",
                    inning,
                    lines
                )

    print("=" * 120)

# ============================================================
# 継投キュー消費検証
# ============================================================
def validate_queue_consumption(
    data: dict,
    at_bats: list[dict],
    filename: str,
    suspicious_games: set
) -> int:

    pitcher_stats = data.get("pitcher_stats", {})
    actual_counts = defaultdict(int)

    for ab in at_bats:
        if ab["pitcher"]:
            actual_counts[ab["pitcher"]] += 1

    mismatch_count = 0

    for pitcher, stats in pitcher_stats.items():
        official_bf = stats.get("bf", 0)
        extracted_bf = actual_counts.get(pitcher, 0)

        if official_bf != extracted_bf:
            mismatch_count += 1

            print(
                f'[QUEUE_MISMATCH] '
                f'{filename} '
                f'{pitcher} '
                f'official_bf={official_bf} '
                f'extracted_bf={extracted_bf}'
            )

            # validate_queue_consumption
            if abs(official_bf - extracted_bf) == 1:
                suspicious_games.add(filename)

    return mismatch_count 


# ============================================================
# BF総和検証
# ============================================================
def validate_bf(
    data: dict,
    at_bats: list[dict],
    filename: str
):

    pitcher_stats = data.get("pitcher_stats", {})

    bf_sum = 0

    for p in pitcher_stats.values():

        bf = p.get("bf", 0)

        if isinstance(bf, int):
            bf_sum += bf

    ab_count = len(at_bats)

    print(
        filename,
        "bf_sum=",
        bf_sum,
        "at_bats=",
        len(at_bats)
    )

    if bf_sum != ab_count:

        print(
            f'[BF_MISMATCH] '
            f'{filename} '
            f'bf_sum={bf_sum} '
            f'at_bats={ab_count}'
        )

        return False

    return True


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
# RollingStats
# ============================================================
class RollingStats:

    def __init__(self, window_days: int = 30):

        self.window = window_days

        self.batter_log = defaultdict(list)
        self.pitcher_log = defaultdict(list)

    def add_game(self, game_date: datetime, at_bats: list[dict]):

        for ab in at_bats:

            if ab['batter']:

                self.batter_log[ab['batter']].append(
                    (game_date, ab['is_ab'], ab['is_hit'])
                )

            if ab['pitcher']:

                self.pitcher_log[ab['pitcher']].append(
                    (game_date, ab['is_ab'], ab['is_hit'])
                )

    def _calc_avg(self, log: list, before_date: datetime):

        cutoff = before_date - timedelta(days=self.window)

        ab_cnt = 0
        hit_cnt = 0

        for (d, is_ab, hit) in log:

            if cutoff <= d < before_date and is_ab:

                ab_cnt += 1

                if hit:
                    hit_cnt += 1

        if ab_cnt == 0:
            return None

        return round(hit_cnt / ab_cnt, 4)

    def batter_avg(self, name: str, game_date: datetime):

        return self._calc_avg(
            self.batter_log.get(name, []),
            game_date
        )

    def pitcher_bavg(self, name: str, game_date: datetime):

        return self._calc_avg(
            self.pitcher_log.get(name, []),
            game_date
        )


# ============================================================
# メイン処理
# ============================================================
def process(
    json_dir: str,
    output_dir: str | None = None,
    inplace: bool = False
):

    print(f"[INFO] 入力ディレクトリ: {json_dir}")

    game_logs, season_start = build_log(json_dir)

    if not game_logs:
        print("[ERROR] JSONファイルなし")
        return

    stats_unlock_date = season_start + timedelta(days=WINDOW_DAYS)

    print(f"[INFO] 開幕日: {season_start.date()}")
    print(f"[INFO] 試合数: {len(game_logs)}")
    print(f"[INFO] 解禁日: {stats_unlock_date.date()}")

    rolling = RollingStats(window_days=WINDOW_DAYS)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 検証統計
    # ========================================================
    total_queue_mismatch = 0
    total_bf_mismatch = 0
    suspicious_games = set()

    for i, (game_date, filepath, data) in enumerate(game_logs):

        filename = Path(filepath).name
        
        at_bats = extract_at_bats(data)
        





        # ====================================================
        # 継投キュー検証
        # ====================================================
        queue_bad = validate_queue_consumption(
            data,
            at_bats,
            filename,
            suspicious_games
        )
        total_queue_mismatch += queue_bad

        # ====================================================
        # BF検証
        # ====================================================
        bf_ok = validate_bf(
            data,
            at_bats,
            filename
        )

        if not bf_ok:
            total_bf_mismatch += 1

        use_stats = (game_date >= stats_unlock_date)

        features = []

        for ab in at_bats:

            if use_stats:

                b_avg = rolling.batter_avg(
                    ab['batter'],
                    game_date
                )

                p_avg = None

                if ab['pitcher']:
                    p_avg = rolling.pitcher_bavg(
                        ab['pitcher'],
                        game_date
                    )

                if b_avg is not None and p_avg is not None:
                    contrast = round(b_avg - p_avg, 4)
                else:
                    contrast = 0.0

            else:

                b_avg = None
                p_avg = None
                contrast = 0.0

            features.append({
                'inning': ab['inning'],
                'batter': ab['batter'],
                'pitcher': ab['pitcher'],
                'batting_team': ab['batting_team'],
                'batter_avg_30d': b_avg,
                'pitcher_bavg_30d': p_avg,
                'contrast_feature': contrast,
            })

        data['at_bat_features'] = features

        rolling.add_game(game_date, at_bats)

        out_path = filepath if inplace else str(
            Path(output_dir) / filename
        )

        with open(out_path, 'w', encoding='utf-8') as fp:
            json.dump(
                data,
                fp,
                ensure_ascii=False,
                indent=2
            )

        unlock_str = ""

        if not use_stats:
            unlock_str = " [30日未満: 0.0]"

        print(
            f"[{i+1:3d}/{len(game_logs)}] "
            f"{filename} "
            f"| 打席数: {len(features)}"
            f"{unlock_str}"
        )

    print("\n[INFO] 完了")

    print("\n===== 検証結果 =====")
# 期待値に合わせるなら
    print("BF_MISMATCH =", total_bf_mismatch)
    print("QUEUE_MISMATCH =", total_queue_mismatch)
    print(f"差分1のファイル: {sorted(suspicious_games)}")



# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('input_dir')

    parser.add_argument('--output_dir', default=None)

    parser.add_argument(
        '--inplace',
        action='store_true'
    )

    args = parser.parse_args()

    if not args.output_dir and not args.inplace:

        parser.error(
            '--output_dir か --inplace を指定してください'
        )

    process(
        args.input_dir,
        args.output_dir,
        args.inplace
    )