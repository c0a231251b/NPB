import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import re

def analyze_total_contribution(json_dir):
    # 保存用フォルダの作成
    save_dir = "graph_data"
    os.makedirs(save_dir, exist_ok=True)

    records = []
    # チーム名に挟まれたスコアのみを抽出する正規表現に強化
    score_re = re.compile(r"[A-Za-z0-9一-龠ぁ-んァ-ヶ]+\s+(\d+)-(\d+)\s+[A-Za-z0-9一-龠ぁ-んァ-ヶ]+")
    batter_re = re.compile(r"(\d+)番")
    
    file_paths = glob.glob(os.path.join(json_dir, "*.json"))
    print(f"解析中... 対象ファイル数: {len(file_paths)}")

    for filename in file_paths:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        current_total = 0
        for section in data.get("text_live", []):
            if "回" not in section.get("inning", ""): continue
            for play in section.get("plays", []):
                lines = play.get("lines", [])
                if not lines: continue
                
                m_bat = batter_re.search(lines[0])
                if not m_bat: continue
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
    if df.empty:
        print("得点データが見つかりませんでした。")
        return
        
    summary = df.groupby("pos")["runs"].sum().reindex(range(1,10), fill_value=0)
    
    print("\n--- 打順別 合計得点貢献数 ---")
    print(summary)
    
    plt.figure(figsize=(10, 6))
    plt.bar(summary.index, summary.values, color="skyblue", edgecolor="black")
    plt.title("Total Runs Produced by Batting Order (All Games)")
    plt.xlabel("Batting Position")
    plt.ylabel("Total Runs")
    plt.xticks(range(1, 10))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # グラフの保存処理
    save_path = os.path.join(save_dir, "total_runs_contribution.png")
    plt.savefig(save_path)
    print(f"\nグラフを保存しました: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    analyze_total_contribution("game_data")