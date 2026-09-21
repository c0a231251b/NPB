# -*- coding: utf-8 -*-
"""
game_data_2025_match_results/*.json に npb_roster_2025.csv の投打情報を付与するスクリプト。

追加する主な項目:
- batter_stats[*].batting_hand / throwing_hand
- pitcher_stats[*].pitching_hand / batting_hand
- starting_pitchers_hands
- text_live[*].pregame.lineups[*].players_with_hands
- text_live[*].plays[*] の打者・投手イベントに batter_hand / pitcher_hand など
- at_bat_stats[*].batter_hand / pitcher_hand があるJSONにも対応

実行例:
python add_handedness_to_game_json.py `
--input_dir "C:/Users/Admin/Desktop/NPB/game_data_2025_match_results" `
--roster_csv "C:/Users/Admin/Desktop/NPB/npb_roster_2025.csv" `
--output_dir "C:/Users/Admin/Desktop/NPB/game_data_2025_match_results_with_hands"


上書きしたい場合:
    python add_handedness_to_game_json.py --input_dir game_data_2025_match_results --roster_csv npb_roster_2025.csv --overwrite
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# 正規化・投打パース
# -----------------------------

TEAM_ALIASES = {
    "ＤｅＮＡ": "DeNA",
    "DeNA": "DeNA",
    "横浜": "DeNA",
    "横浜DeNA": "DeNA",
    "ソフトバンク": "ソフトバンク",
    "福岡ソフトバンク": "ソフトバンク",
    "日本ハム": "日本ハム",
    "北海道日本ハム": "日本ハム",
    "ロッテ": "ロッテ",
    "千葉ロッテ": "ロッテ",
    "オリックス": "オリックス",
    "楽天": "楽天",
    "東北楽天": "楽天",
    "西武": "西武",
    "埼玉西武": "西武",
    "巨人": "巨人",
    "読売": "巨人",
    "阪神": "阪神",
    "広島": "広島",
    "東洋広島": "広島",
    "中日": "中日",
    "ヤクルト": "ヤクルト",
    "東京ヤクルト": "ヤクルト",
}


def norm_text(s: Any) -> str:
    """選手名・チーム名照合用の正規化。"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = re.sub(r"\s+", "", s)
    return s.strip()


def norm_team(team: Any) -> str:
    t = norm_text(team)
    return TEAM_ALIASES.get(t, t)


def remove_parentheses(s: str) -> str:
    # 例: (翁田)大勢 -> 大勢
    return re.sub(r"[\(\（][^\)\）]*[\)\）]", "", s)


def strip_latin_initial(s: str) -> str:
    # 例: F.グリフィン -> グリフィン
    return re.sub(r"^[A-Za-zＡ-Ｚａ-ｚ]\.?", "", s)


def parse_throw_bat(tb: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    npb_roster_2025.csv の「投打」列を、投 / 打に分ける。
    例: 右右 -> ("右", "右"), 左左 -> ("左", "左"), 右両 -> ("右", "両")
    """
    s = norm_text(tb)
    chars = [c for c in s if c in {"右", "左", "両"}]
    if len(chars) >= 2:
        return chars[0], chars[1]
    if len(chars) == 1:
        return chars[0], None
    return None, None


def name_variants(name: Any) -> List[str]:
    """
    JSON側は「戸郷」「中村悠」のような短縮名、
    roster側は「戸郷 翔征」「中村 悠平」のようなフルネームが多いため、
    照合候補を複数作る。
    """
    raw = "" if name is None else unicodedata.normalize("NFKC", str(name))
    no_spaces = norm_text(raw)
    no_paren = norm_text(remove_parentheses(raw))
    no_initial = norm_text(strip_latin_initial(raw))
    no_paren_initial = norm_text(strip_latin_initial(remove_parentheses(raw)))

    variants = [no_spaces, no_paren, no_initial, no_paren_initial]

    # 空白区切りがある場合は姓だけも候補にする
    tokens = [norm_text(x) for x in re.split(r"[\s\u3000]+", raw.strip()) if norm_text(x)]
    if tokens:
        variants.append(tokens[0])

    # 重複除去
    out: List[str] = []
    for v in variants:
        if v and v not in out:
            out.append(v)
    return out


# -----------------------------
# roster検索インデックス
# -----------------------------

class RosterIndex:
    def __init__(self, roster_csv: str | Path):
        df = pd.read_csv(roster_csv)
        required = {"チーム名", "選手名", "投打"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"roster CSVに必要な列がありません: {sorted(missing)}")

        self.by_team_exact: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
        self.global_exact: Dict[str, List[dict]] = defaultdict(list)
        self.by_team_records: Dict[str, List[dict]] = defaultdict(list)
        self.all_records: List[dict] = []

        for _, row in df.iterrows():
            team = norm_team(row["チーム名"])
            player_name = str(row["選手名"])
            throw_hand, bat_hand = parse_throw_bat(row["投打"])

            rec = {
                "team": team,
                "player_name": player_name,
                "player_name_norm": norm_text(player_name),
                "throwing_hand": throw_hand,
                "batting_hand": bat_hand,
                "raw_throw_bat": row["投打"],
            }
            self.all_records.append(rec)
            self.by_team_records[team].append(rec)

            for v in name_variants(player_name):
                self.by_team_exact[(team, v)].append(rec)
                self.global_exact[v].append(rec)

    def lookup(self, name: Any, team: Any = None) -> Optional[dict]:
        """
        できるだけ team + name で照合し、無理ならグローバルで一意に照合する。
        短縮名は「JSON名がrosterフルネームの先頭に一致」で拾う。
        """
        if name is None:
            return None

        teams: List[Optional[str]] = []
        if team is not None:
            teams.append(norm_team(team))
        teams.append(None)

        variants = name_variants(name)

        # 1) team内の完全一致
        for t in teams:
            if t is None:
                break
            for v in variants:
                hits = self.by_team_exact.get((t, v), [])
                if len(hits) == 1:
                    return hits[0]

        # 2) team内の前方一致: 戸郷 -> 戸郷翔征 / 中村悠 -> 中村悠平
        for t in teams:
            if t is None:
                break
            for v in variants:
                hits = [
                    r for r in self.by_team_records.get(t, [])
                    if r["player_name_norm"].startswith(v) or v.startswith(r["player_name_norm"])
                ]
                if len(hits) == 1:
                    return hits[0]

        # 3) グローバル完全一致。一意なら採用
        for v in variants:
            hits = self.global_exact.get(v, [])
            if len(hits) == 1:
                return hits[0]

        # 4) グローバル前方一致。一意なら採用
        for v in variants:
            hits = [
                r for r in self.all_records
                if r["player_name_norm"].startswith(v) or v.startswith(r["player_name_norm"])
            ]
            # 同じ選手が別variantで重複する可能性を除去
            unique = {}
            for r in hits:
                unique[(r["team"], r["player_name_norm"])] = r
            hits = list(unique.values())
            if len(hits) == 1:
                return hits[0]

        return None


# -----------------------------
# JSON構造に左右情報を付与
# -----------------------------

BATTER_LINE_RE = re.compile(r"^(?:(\d+)番|打|走|打走|守|投打|代打)\s+(.+)$")
PITCHER_LINE_RE = re.compile(r"^投手[:：]\s*(.+)$")


def get_teams_from_game(game: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    scoreboard[0] = 表の攻撃チーム、scoreboard[1] = 裏の攻撃チームとして扱う。
    """
    scoreboard = game.get("scoreboard") or []
    if len(scoreboard) >= 2:
        return scoreboard[0].get("team"), scoreboard[1].get("team")
    return None, None


def batting_team_from_inning(inning: Any, top_team: Any, bottom_team: Any) -> Optional[str]:
    s = str(inning or "")
    if "表" in s:
        return top_team
    if "裏" in s:
        return bottom_team
    return None


def fielding_team_from_inning(inning: Any, top_team: Any, bottom_team: Any) -> Optional[str]:
    s = str(inning or "")
    if "表" in s:
        return bottom_team
    if "裏" in s:
        return top_team
    return None


def add_hand_fields_to_player_dict(
    obj: dict,
    roster: RosterIndex,
    player_name: Any,
    team: Any,
    role: str,
    unresolved: Counter,
) -> None:
    """
    role:
      - "batter": batting_handを主に使う
      - "pitcher": throwing_handを主に使う
      - "both": 両方足す
    """
    rec = roster.lookup(player_name, team)
    if rec is None:
        unresolved[(norm_team(team), str(player_name), role)] += 1
        if role in {"batter", "both"}:
            obj.setdefault("batter_hand", None)
            obj.setdefault("batting_hand", None)
        if role in {"pitcher", "both"}:
            obj.setdefault("pitcher_hand", None)
            obj.setdefault("throwing_hand", None)
        return

    if role in {"batter", "both"}:
        # 学習CSVで使いやすい名前
        obj["batter_hand"] = rec["batting_hand"]
        obj["batting_hand"] = rec["batting_hand"]
        obj.setdefault("throwing_hand", rec["throwing_hand"])

    if role in {"pitcher", "both"}:
        # 学習CSVで使いやすい名前
        obj["pitcher_hand"] = rec["throwing_hand"]
        obj["throwing_hand"] = rec["throwing_hand"]
        obj.setdefault("batting_hand", rec["batting_hand"])


def enrich_pregame_lineups(game: dict, roster: RosterIndex, unresolved: Counter) -> None:
    for block in game.get("text_live", []) or []:
        pregame = block.get("pregame")
        if not isinstance(pregame, dict):
            continue

        for lineup in pregame.get("lineups", []) or []:
            team = lineup.get("team")
            players = lineup.get("players", []) or []
            players_with_hands = []

            for i, name in enumerate(players, start=1):
                item = {"order": i, "player_name": name}
                add_hand_fields_to_player_dict(item, roster, name, team, "batter", unresolved)
                players_with_hands.append(item)

            # 元の players は壊さず、追加情報だけ別キーにする
            lineup["players_with_hands"] = players_with_hands


def enrich_batter_stats(game: dict, roster: RosterIndex, unresolved: Counter) -> None:
    batter_stats = game.get("batter_stats")
    if not isinstance(batter_stats, dict):
        return

    for team, rows in batter_stats.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = row.get("player_name")
            add_hand_fields_to_player_dict(row, roster, name, team, "batter", unresolved)


def enrich_pitcher_stats(game: dict, roster: RosterIndex, unresolved: Counter) -> None:
    pitcher_stats = game.get("pitcher_stats")
    if not isinstance(pitcher_stats, dict):
        return

    for team, rows in pitcher_stats.items():
        # 形式1: {"巨人": [{"player_name": "..."}]}
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                name = row.get("player_name")
                add_hand_fields_to_player_dict(row, roster, name, team, "pitcher", unresolved)

        # 形式2: {"巨人": {"戸郷": {"win": ...}}}
        elif isinstance(rows, dict):
            for name, row in rows.items():
                if not isinstance(row, dict):
                    continue
                row.setdefault("player_name", name)
                add_hand_fields_to_player_dict(row, roster, name, team, "pitcher", unresolved)


def enrich_starting_pitchers(game: dict, roster: RosterIndex, unresolved: Counter) -> None:
    starters = game.get("starting_pitchers")
    if not isinstance(starters, dict):
        return

    top_team, bottom_team = get_teams_from_game(game)
    out = {}

    for side, name in starters.items():
        # 「表」の投手は裏チーム、「裏」の投手は表チームとして扱う
        pitcher_team = bottom_team if side == "表" else top_team if side == "裏" else None
        item = {"player_name": name, "team": pitcher_team}
        add_hand_fields_to_player_dict(item, roster, name, pitcher_team, "pitcher", unresolved)
        out[side] = item

    game["starting_pitchers_hands"] = out


def enrich_at_bat_stats(game: dict, roster: RosterIndex, unresolved: Counter) -> None:
    """
    merged_game_data_2025_Just_before_the_turn_at_bat_stats.json のような
    at_bat_stats形式にも対応。
    """
    rows = game.get("at_bat_stats")
    if not isinstance(rows, list):
        return

    top_team, bottom_team = get_teams_from_game(game)

    for row in rows:
        if not isinstance(row, dict):
            continue

        inning = row.get("inning")
        batting_team = row.get("batting_team") or batting_team_from_inning(inning, top_team, bottom_team)
        fielding_team = fielding_team_from_inning(inning, top_team, bottom_team)

        batter = row.get("batter")
        pitcher = row.get("pitcher")

        batter_rec = roster.lookup(batter, batting_team)
        pitcher_rec = roster.lookup(pitcher, fielding_team)

        if batter_rec is None:
            unresolved[(norm_team(batting_team), str(batter), "at_bat_batter")] += 1
            row["batter_hand"] = None
        else:
            row["batter_hand"] = batter_rec["batting_hand"]

        if pitcher_rec is None:
            unresolved[(norm_team(fielding_team), str(pitcher), "at_bat_pitcher")] += 1
            row["pitcher_hand"] = None
        else:
            row["pitcher_hand"] = pitcher_rec["throwing_hand"]


def enrich_text_live_plays(game: dict, roster: RosterIndex, unresolved: Counter) -> None:
    """
    text_liveの各playにも、可能な範囲で現在投手・打者の左右を付与。
    元の lines は変更しない。
    """
    top_team, bottom_team = get_teams_from_game(game)
    current_pitcher_by_half: Dict[str, Optional[str]] = {"表": None, "裏": None}

    for block in game.get("text_live", []) or []:
        inning = block.get("inning")
        half = "表" if "表" in str(inning) else "裏" if "裏" in str(inning) else None
        if half is None:
            continue

        batting_team = batting_team_from_inning(inning, top_team, bottom_team)
        fielding_team = fielding_team_from_inning(inning, top_team, bottom_team)

        for play in block.get("plays", []) or []:
            if not isinstance(play, dict):
                continue
            lines = play.get("lines") or []
            if not lines:
                continue

            first = str(lines[0]).strip()

            m_pitcher = PITCHER_LINE_RE.match(first)
            if m_pitcher:
                pitcher_name = m_pitcher.group(1).strip()
                current_pitcher_by_half[half] = pitcher_name
                play["event_type"] = "pitcher"
                play["pitcher"] = pitcher_name
                play["pitching_team"] = fielding_team
                add_hand_fields_to_player_dict(play, roster, pitcher_name, fielding_team, "pitcher", unresolved)
                continue

            m_batter = BATTER_LINE_RE.match(first)
            if m_batter:
                batter_name = m_batter.group(2).strip()
                pitcher_name = current_pitcher_by_half.get(half)

                play["event_type"] = "plate_appearance"
                play["batter"] = batter_name
                play["batting_team"] = batting_team
                add_hand_fields_to_player_dict(play, roster, batter_name, batting_team, "batter", unresolved)

                play["pitcher"] = pitcher_name
                play["pitching_team"] = fielding_team
                if pitcher_name:
                    rec = roster.lookup(pitcher_name, fielding_team)
                    if rec is None:
                        unresolved[(norm_team(fielding_team), str(pitcher_name), "current_pitcher")] += 1
                        play["pitcher_hand"] = None
                    else:
                        play["pitcher_hand"] = rec["throwing_hand"]
                else:
                    play["pitcher_hand"] = None


def enrich_game(game: dict, roster: RosterIndex, unresolved: Counter) -> dict:
    game = copy.deepcopy(game)

    enrich_pregame_lineups(game, roster, unresolved)
    enrich_batter_stats(game, roster, unresolved)
    enrich_pitcher_stats(game, roster, unresolved)
    enrich_starting_pitchers(game, roster, unresolved)
    enrich_at_bat_stats(game, roster, unresolved)
    enrich_text_live_plays(game, roster, unresolved)

    return game


# -----------------------------
# 入出力
# -----------------------------

def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_file(src: Path, dst: Path, roster: RosterIndex, unresolved: Counter) -> None:
    data = load_json_file(src)

    if isinstance(data, list):
        enriched = [
            enrich_game(x, roster, unresolved) if isinstance(x, dict) else x
            for x in data
        ]
    elif isinstance(data, dict):
        enriched = enrich_game(data, roster, unresolved)
    else:
        enriched = data

    dst.parent.mkdir(parents=True, exist_ok=True)
    save_json_file(dst, enriched)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="game_data_2025_match_results")
    parser.add_argument("--roster_csv", default="npb_roster_2025.csv")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--report_csv", default="unresolved_handedness_report.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    roster = RosterIndex(args.roster_csv)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"JSONファイルが見つかりません: {input_dir / args.pattern}")

    if args.overwrite:
        output_dir = input_dir
    else:
        output_dir = Path(args.output_dir) if args.output_dir else input_dir.with_name(input_dir.name + "_with_hands")

    unresolved: Counter = Counter()

    for src in files:
        dst = src if args.overwrite else output_dir / src.name
        process_file(src, dst, roster, unresolved)
        print(f"OK: {src} -> {dst}")

    # 未照合レポート
    report_rows = [
        {
            "team": k[0],
            "player_name": k[1],
            "role": k[2],
            "count": v,
        }
        for k, v in unresolved.most_common()
    ]

    report_path = (input_dir if args.overwrite else output_dir) / args.report_csv
    pd.DataFrame(report_rows).to_csv(report_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"processed_files: {len(files)}")
    print(f"output_dir: {output_dir}")
    print(f"unresolved_unique: {len(report_rows)}")
    print(f"unresolved_report: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
