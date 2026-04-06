import json
import glob
from collections import Counter

def analyze_play_expressions(json_dir):
    all_lines = []
    # 保存した全JSONファイルを読み込む
    file_paths = glob.glob(f"{json_dir}/*.json")
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for section in data.get("text_live", []):
                # 「試合前情報」などは除外
                if "回" not in section.get("inning", ""):
                    continue
                
                for play in section.get("plays", []):
                    # 各プレイの2行目以降に結果が書かれていることが多い
                    if len(play["lines"]) >= 2:
                        # 1行目は「○番 選手名...」なので、2行目以降を抽出
                        result_line = " ".join(play["lines"][1:])
                        all_lines.append(result_line)

    # 頻出する表現をカウント
    expression_counts = Counter(all_lines)
    
    print(f"総打席数: {len(all_lines)}")
    print("--- 頻出する表現（上位100件） ---")
    for expr, count in expression_counts.most_common(100):
        print(f"{count}回: {expr}")

# 実行（フォルダ名を指定）
analyze_play_expressions("game_data")