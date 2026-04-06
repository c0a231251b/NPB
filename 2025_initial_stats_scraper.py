import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def scrape_initial_stats():
    # 2025年度のチーム別個人打撃成績URLリスト
    urls = [
        "https://npb.jp/bis/2025/stats/idb1_g.html", "https://npb.jp/bis/2025/stats/idb1_t.html",
        "https://npb.jp/bis/2025/stats/idb1_db.html", "https://npb.jp/bis/2025/stats/idb1_c.html",
        "https://npb.jp/bis/2025/stats/idb1_s.html", "https://npb.jp/bis/2025/stats/idb1_d.html",
        "https://npb.jp/bis/2025/stats/idb1_h.html", "https://npb.jp/bis/2025/stats/idb1_f.html",
        "https://npb.jp/bis/2025/stats/idb1_m.html", "https://npb.jp/bis/2025/stats/idb1_e.html",
        "https://npb.jp/bis/2025/stats/idb1_b.html", "https://npb.jp/bis/2025/stats/idb1_l.html"
    ]

    all_players = []

    for url in urls:
        print(f"Scraping: {url}")
        res = requests.get(url)
        res.encoding = "utf-8"
        soup = BeautifulSoup(res.text, "html.parser")

        # テーブルの取得 
        table = soup.find("table", class_="tablefix2")
        if not table:
            continue

        # データ行の抽出 [cite: 38]
        rows = table.find("tbody").find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if not cols:
                continue
            
            # 選手名のクレンジング（スペースや注釈記号の除去） [cite: 34, 39]
            name = cols[0].get_text(strip=True).replace("\u3000", "").replace("*", "").replace("+", "")
            
            try:
                # 必要なカラムのインデックスに合わせて抽出 [cite: 35, 36, 37]
                # 指標: 名前, 打数, 安打, 二塁打, 三塁打, 本塁打, 塁打, 四球, 死球, 犠飛
                data = {
                    "name": name,
                    "AB": int(cols[3].get_text()),  # 打数
                    "H": int(cols[5].get_text()),   # 安打
                    "2B": int(cols[6].get_text()),  # 二塁打
                    "3B": int(cols[7].get_text()),  # 三塁打
                    "HR": int(cols[8].get_text()),  # 本塁打
                    "TB": int(cols[9].get_text()),  # 塁打
                    "BB": int(cols[15].get_text()), # 四球
                    "HBP": int(cols[17].get_text()),# 死球
                    "SF": int(cols[14].get_text())  # 犠飛
                }
                all_players.append(data)
            except (ValueError, IndexError):
                continue
        
        time.sleep(1) # サーバー負荷軽減

    df = pd.DataFrame(all_players)

    # --- 実績なし選手用の「リーグ平均」を算出 ---
    # 野手として意味のある平均を出すため、一定打数以上の平均を計算
    summary = df.sum(numeric_only=True)
    player_count = len(df)
    
    # 1人あたりの平均値を算出
    avg_stats = {
        "name": "LEAGUE_AVERAGE",
        "AB": summary["AB"] / player_count,
        "H": summary["H"] / player_count,
        "2B": summary["2B"] / player_count,
        "3B": summary["3B"] / player_count,
        "HR": summary["HR"] / player_count,
        "TB": summary["TB"] / player_count,
        "BB": summary["BB"] / player_count,
        "HBP": summary["HBP"] / player_count,
        "SF": summary["SF"] / player_count
    }
    
    # 平均行を追加
    df = pd.concat([df, pd.DataFrame([avg_stats])], ignore_index=True)

    # CSV保存
    df.to_csv("initial_stats_2025.csv", index=False, encoding="utf-8-sig")
    print(f"\n完了！ {len(df)-1}名の選手データとリーグ平均を 'initial_stats_2025.csv' に保存しました。")

if __name__ == "__main__":
    scrape_initial_stats()