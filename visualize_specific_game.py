import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

def analyze_specific_game(file_path):
    if not os.path.exists(file_path):
        print(f"エラー: ファイル {file_path} が見つかりません。")
        return

    # 保存用フォルダの作成
    save_dir = "graph_data"
    os.makedirs(save_dir, exist_ok=True)

    records = []
    score_re = re.compile(r"(\d+)-(\d+)")
    batter_re = re.compile(r"(\d+)番")
    
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"試合解析中: {data.get('title', file_path)}")
    
    current_total = 0
    for section in data.get("text_live", []):
        if "回" not in section.get("inning", ""): continue
        for play in section.get("plays", []):
            lines = play.get("lines", [])
            if not lines: continue
            
            m_bat = batter_re.search(lines[0])
            if m_bat:
                pos = int(m_bat.group(1))
                text = " ".join(lines)
                m_scores = score_re.findall(text)
                if m_scores:
                    s1, s2 = map(int, m_scores[-1])
                    new_total = s1 + s2
                    if new_total > current_total:
                        runs = new_total - current_total
                        records.append({"pos": pos, "runs": runs})
                        current_total = new_total
    
    df = pd.DataFrame(records)
    summary = df.groupby("pos")["runs"].sum().reindex(range(1,10), fill_value=0)
    
    print("\n--- 特定試合 打順別得点貢献 ---")
    print(summary)
    
    plt.figure(figsize=(10, 6))
    plt.bar(summary.index, summary.values, color="salmon", edgecolor="black")
    plt.title(f"Runs Produced - {os.path.basename(file_path)}")
    plt.xlabel("Batting Position")
    plt.ylabel("Runs")
    plt.xticks(range(1, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # グラフの保存処理（ファイル名を流用）
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(save_dir, f"contribution_{file_name}.png")
    plt.savefig(save_path)
    print(f"\nグラフを保存しました: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    target = os.path.join("game_data", "game_2021038622_text.json")
    analyze_specific_game(target)