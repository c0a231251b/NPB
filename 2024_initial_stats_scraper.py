import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import numpy as np

def scrape_2024_stats():
    # 2024年度のチーム別個人打撃成績URL
    team_urls = {
        "巨人": "https://npb.jp/bis/2024/stats/idb1_g.html",
        "阪神": "https://npb.jp/bis/2024/stats/idb1_t.html",
        "DeNA": "https://npb.jp/bis/2024/stats/idb1_db.html",
        "広島": "https://npb.jp/bis/2024/stats/idb1_c.html",
        "ヤクルト": "https://npb.jp/bis/2024/stats/idb1_s.html",
        "中日": "https://npb.jp/bis/2024/stats/idb1_d.html",
        "ソフトバンク": "https://npb.jp/bis/2024/stats/idb1_h.html",
        "日本ハム": "https://npb.jp/bis/2024/stats/idb1_f.html",
        "ロッテ": "https://npb.jp/bis/2024/stats/idb1_m.html",
        "楽天": "https://npb.jp/bis/2024/stats/idb1_e.html",
        "オリックス": "https://npb.jp/bis/2024/stats/idb1_b.html",
        "西武": "https://npb.jp/bis/2024/stats/idb1_l.html"
    }

    all_players = []

    def to_int(element):
        val = element.get_text(strip=True).replace(',', '')
        if not val or val == '-':
            return 0
        return int(val)

    for team_name, url in team_urls.items():
        print(f"Scraping {team_name}: {url}")
        try:
            res = requests.get(url)
            res.encoding = res.apparent_encoding 
            soup = BeautifulSoup(res.text, "html.parser")
            
            rows = soup.find_all("tr", class_="ststats")
            
            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 20:
                    continue
                
                # --- 打ち方(Hand)の判定を文字列に変更 ---
                hand_mark = cols[0].get_text(strip=True)
                if "*" in hand_mark:
                    hand = "左打"
                elif "+" in hand_mark:
                    hand = "両打"
                else:
                    hand = "右打"
                
                raw_name = cols[1].get_text(strip=True)
                name = raw_name.replace("\u3000", "").replace(" ", "").replace("*", "").replace("+", "")
                
                if name == "計":
                    continue

                player_data = {
                    "team": team_name,
                    "name": name,
                    "Hand": hand,
                    "AB": to_int(cols[4]),
                    "H": to_int(cols[6]),
                    "2B": to_int(cols[7]),
                    "3B": to_int(cols[8]),
                    "HR": to_int(cols[9]),
                    "TB": to_int(cols[10]),
                    "SB": to_int(cols[12]),
                    "BB": to_int(cols[16]),
                    "HBP": to_int(cols[18]),
                    "SF": to_int(cols[15])
                }
                all_players.append(player_data)
            
            time.sleep(0.5)
        except Exception as e:
            print(f"  Error scraping {team_name}: {e}")

    if not all_players:
        print("\nError: 選手データを取得できませんでした。")
        return

    df = pd.DataFrame(all_players)

    # リーグ平均の算出
    # Handは文字列になったため、平均計算からは自動的に除外されます
    summary = df.mean(numeric_only=True)
    avg_stats = {
        "team": "NPB",
        "name": "LEAGUE_AVERAGE",
        "Hand": "右打", # 平均行は便宜上「右打」固定、または「-」とする
        "AB": summary["AB"], "H": summary["H"], "2B": summary["2B"],
        "3B": summary["3B"], "HR": summary["HR"], "TB": summary["TB"],
        "SB": summary["SB"],
        "BB": summary["BB"], "HBP": summary["HBP"], "SF": summary["SF"]
    }
    
    df = pd.concat([df, pd.DataFrame([avg_stats])], ignore_index=True)

    output_file = "initial_stats_2024.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n完了！ {len(df)-1} 名の選手データを '{output_file}' に保存しました。")

if __name__ == "__main__":
    scrape_2024_stats()