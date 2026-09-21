import json

input_file = "merged_game_data_2025.json"
output_file = "a.txt"

results = []

def search_keys(obj):
    """JSON のどんな階層からでも '表' と '裏' を探す再帰関数"""
    if isinstance(obj, dict):
        # 表・裏があれば記録
        if "表" in obj and "裏" in obj:
            results.append((obj["表"], obj["裏"]))

        # 子要素を探索
        for v in obj.values():
            search_keys(v)

    elif isinstance(obj, list):
        for item in obj:
            search_keys(item)

def main():
    # JSON 読み込み
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 再帰的に探索
    search_keys(data)

    # 書き出し
    with open(output_file, "w", encoding="utf-8") as f:
        for front, back in results:
            f.write(f"表: {front}\n")
            # 一行改行を出さないでほしい
            f.write(f"裏: {back}\n")

    print(f"抽出完了！ {len(results)} 件の '表' '裏' を a.txt に保存しました。")

if __name__ == "__main__":
    main()
