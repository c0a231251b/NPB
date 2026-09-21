from pathlib import Path

# ==========================================
# JSONディレクトリ
# ==========================================
JSON_DIR = "aiueo/"

# ==========================================
# カウント
# ==========================================
null_pitcher_count = 0

# ファイルごとの件数
file_counts = []

# ==========================================
# JSON走査
# ==========================================
files = sorted(Path(JSON_DIR).glob("*.json"))

for filepath in files:

    try:
        # ファイル全体を文字列として読む
        text = filepath.read_text(encoding="utf-8")

        # "pitcher": null の数を数える
        count = text.count('"pitcher": null')




        null_pitcher_count += count

        # 1件以上ある場合のみ保存
        if count > 0:
            file_counts.append(
                f"{filepath.name} -> {count}"
            )

    except Exception as e:
        print(f"[ERROR] {filepath.name}: {e}")

# ==========================================
# 結果表示
# ==========================================
print("\n===== 集計結果 =====")
print(f'"pitcher": null の総数: {null_pitcher_count}')

print("\n===== ファイル別件数 =====")

for line in file_counts:
    print(line)