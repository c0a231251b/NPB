import json
from pathlib import Path

def merge_json_files(
        

    input_dir: str = "game_data_2025_match_results_Just_before_the_turn_at_bat_stats",
    output_path: str = "merged_game_data_2025_Just_before_the_turn_at_bat_stats.json"
    #input_dir: str = "game_data_2025_match_results",
    #output_path: str = "merged_game_data_2025.json"
) -> None:
    """
    指定フォルダ内のすべての .json ファイルを読み込み、
    1つのJSONファイル（配列）として結合して書き出す。
    """
    input_path = Path(input_dir)
    all_data = []

    # フォルダ内の .json ファイルをすべて走査
    for json_file in sorted(input_path.glob("*.json")):
        print(f"読み込み中: {json_file}")
        with json_file.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"JSONの読み込みに失敗しました: {json_file} -> {e}")
                continue

        # ファイルの中身が配列かオブジェクトかで処理を分ける
        if isinstance(data, list):
            all_data.extend(data)
        else:
            all_data.append(data)

    # 結合結果を書き出し
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(all_data, out_f, ensure_ascii=False, indent=2)

    print(f"結合完了: {output_path}")

if __name__ == "__main__":
    merge_json_files()
