import json
from pathlib import Path

# 入力ファイル
INPUT_FILE = "merged_game_data_2025.json"

# 出力ファイル
OUTPUT_FILE = "text_live_result_patterns.txt"

def extract_play_results(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result_set = set()

    # text_live が複数試合分ある前提
    for game in data:
        text_live = game.get("text_live", [])
        for item in text_live:
            plays = item.get("plays", [])
            for play in plays:
                lines = play.get("lines", [])

                # lines が 1 行だけで「投手：◯◯」の場合はスキップ
                if len(lines) == 1 and lines[0].startswith("投手："):
                    continue

                # 打者行（例： "1番 西川" ）はスキップ
                if len(lines) >= 1 and ("番 " in lines[0] or lines[0].startswith("打 ")):
                    # 2 行目が打席結果
                    if len(lines) >= 2:
                        result_set.add(lines[1].strip())
                    continue

                # それ以外のケース（稀に1行だけの結果がある場合）
                for line in lines:
                    # 投手行は除外
                    if line.startswith("投手："):
                        continue
                    result_set.add(line.strip())

    return sorted(result_set)


def main():
    results = extract_play_results(INPUT_FILE)

    # txt に保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")

    print(f"抽出完了：{OUTPUT_FILE} に {len(results)} 個の打席結果パターンを保存しました。")


if __name__ == "__main__":
    main()
