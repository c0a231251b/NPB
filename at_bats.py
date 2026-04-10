import glob
import json
import os
import re

"""
JSONファイルから打席数をカウントする
"""
JSON_DIR = "game_data_2025" # 必要に応じて変更

def classify_play(line):
    """
    打席結果を分類する（必要に応じてカウントに利用）
    """
    res = {"H": 0, "TB": 0, "AB": 1, "BB": 0, "SF": 0, "HBP": 0}
    if any(kw in line for kw in ["フォアボール", "四球", "敬遠"]):
        res["BB"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["デッドボール", "死球"]):
        res["HBP"], res["AB"] = 1, 0
    elif any(kw in line for kw in ["ホームラン", "本塁打"]):
        res["H"], res["TB"] = 1, 4
    elif any(kw in line for kw in ["ヒット", "安打"]):
        res["H"], res["TB"] = 1, 1
    elif "犠牲フライ" in line:
        res["SF"], res["AB"] = 1, 0
    return res

def count_at_bats_in_game(game: dict) -> int:
    total = 0
    for section in game.get("text_live", []):
        inning = section.get("inning", "")
        if "回" not in inning:
            continue

        for play in section.get("plays", []):
            lines = play.get("lines", [])
            if len(lines) < 2: 
                continue # 2行ない場合はスキップ
    
            # 1行目から「1番 西川」の形式を抽出
            match = re.match(r"(\d+)番\s+(.+)", lines[0])
            if match:
                pos = int(match.group(1))
                name = match.group(2)
                result = lines[1] # 2行目が「三振」などの結果
                
                # 打席としてカウント
                total += 1
                
                # 必要に応じて成績を更新するロジックをここに挟む
                res_stats = classify_play(result)
    
    return total

def main() -> None:
    paths = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
    if not paths:
        print(f"JSONが見つかりません: {JSON_DIR}")
        return

    grand_total = 0

    # 追加：出力ファイルを開く
    with open("result.txt", "w", encoding="utf-8") as out:
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                game = json.load(f)

            at_bats = count_at_bats_in_game(game)
            grand_total += at_bats

            # printの代わりに file=out を使う
            print(f"{os.path.basename(path)}: {at_bats}打席", file=out)

        print("-" * 40, file=out)
        print(f"対象ファイル数: {len(paths)}", file=out)
        print(f"総打席数: {grand_total}", file=out)

if __name__ == "__main__":
    main()