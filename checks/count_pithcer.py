import json
from pathlib import Path

# ==========================================
# JSONディレクトリ
# ==========================================
JSON_DIR = "game_data_2025_updated_hoge/"

# ==========================================
# 出力ファイル
# ==========================================
OUTPUT_FILE = "a.txt"

# ==========================================
# 集計用
# ==========================================
total_pitcher_count = 0

# 「投手：」が0回
no_pitcher_files = []

# 「投手：」が1回
one_pitcher_files = []

# 出力内容
output_lines = []

# ==========================================
# JSON走査
# ==========================================
files = sorted(Path(JSON_DIR).glob("*.json"))

for filepath in files:

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        pitcher_lines = []

        for inning_data in data.get("text_live", []):

            for play in inning_data.get("plays", []):

                lines = play.get("lines", [])

                for line in lines:

                    # 「投手：」行
                    if line.startswith("投手："):

                        total_pitcher_count += 1
                        pitcher_lines.append(line)

                        output_lines.append(
                            f"{filepath.name} -> {line}"
                        )

        # ==========================================
        # ファイルごとの判定
        # ==========================================
        pitcher_line_count = len(pitcher_lines)

        if pitcher_line_count == 0:
            no_pitcher_files.append(filepath.name)

        elif pitcher_line_count == 1:
            one_pitcher_files.append(
                f"{filepath.name} -> {pitcher_lines[0]}"
            )

    except Exception as e:
        error_msg = f"[ERROR] {filepath.name}: {e}"
        print(error_msg)
        output_lines.append(error_msg)

# ==========================================
# 集計結果
# ==========================================
output_lines.append("")
output_lines.append("===== 集計結果 =====")
output_lines.append(f'「投手：」を含む総行数: {total_pitcher_count}')

# ==========================================
# 「投手：」が存在しないJSON
# ==========================================
output_lines.append("")
output_lines.append("===== 「投手：」が0回のJSON =====")

for name in no_pitcher_files:
    output_lines.append(name)

# ==========================================
# 「投手：」が1回だけのJSON
# ==========================================
output_lines.append("")
output_lines.append("===== 「投手：」が1回だけのJSON =====")

for line in one_pitcher_files:
    output_lines.append(line)

# ==========================================
# a.txt に保存
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print(f"\n保存完了: {OUTPUT_FILE}")