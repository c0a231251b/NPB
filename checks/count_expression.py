import json
from pathlib import Path
from collections import Counter

# ==========================================
# JSONディレクトリ
# ==========================================
JSON_DIR = "game_data_2025_updated_hoge/"   # ←変更してください

# ==========================================
# 打席結果を集計
# ==========================================
result_counter = Counter()

# ==========================================
# JSON走査
# ==========================================
files = sorted(Path(JSON_DIR).glob("*.json"))

for filepath in files:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # text_live を取得
        for inning_data in data.get("text_live", []):

            for play in inning_data.get("plays", []):

                lines = play.get("lines", [])

                # 最低2行必要
                if len(lines) < 2:
                    continue

                first_line = lines[0]
                result     = lines[1].strip()

                # 打席行のみ対象
                is_batter_line = (
                    ('番' in first_line)
                    or first_line.startswith('打')
                )

                if not is_batter_line:
                    continue

                result_counter[result] += 1

    except Exception as e:
        print(f"[ERROR] {filepath.name}: {e}")

# ==========================================
# 出現回数順に表示
# ==========================================
print("\n===== 打席結果一覧 =====\n")

for result, count in result_counter.most_common():
    print(f"{result:<20} : {count}")