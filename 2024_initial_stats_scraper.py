import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

def scrape_2024_stats():
    # 2024年度のチーム別個人打撃成績URL
    # ソースコードに基づき、正しいURLとチーム名を設定
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
        """テキストを数値に変換。空文字やハイフンは0にする"""
        val = element.get_text(strip=True).replace(',', '')
        if not val or val == '-':
            return 0
        return int(val)

    for team_name, url in team_urls.items():
        print(f"Scraping {team_name}: {url}")
        try:
            res = requests.get(url)
            # ソースコードの meta charset="utf-8" に合わせつつ、自動判定も行う
            res.encoding = res.apparent_encoding 
            
            soup = BeautifulSoup(res.text, "html.parser")
            
            # ソースコードにある tr class="ststats" を抽出
            rows = soup.find_all("tr", class_="ststats")
            
            if not rows:
                print(f"  Warning: No data rows found for {team_name}")
                continue

            for row in rows:
                cols = row.find_all("td")
                # 選手データ行は td が20列以上ある
                if len(cols) < 20:
                    continue
                
                # 名前 (index 1) のクレンジング
                raw_name = cols[1].get_text(strip=True)
                name = raw_name.replace("\u3000", "").replace(" ", "").replace("*", "").replace("+", "")
                
                # 投手などで打撃成績が「計」となっている行は除外
                if name == "計":
                    continue

                # 各指標の抽出 (ソースコードの列順に基づく)
                # index 4:打数, 6:安打, 7:二塁打, 8:三塁打, 9:本塁打, 10:塁打, 15:犠飛, 16:四球, 18:死球
                player_data = {
                    "team": team_name,
                    "name": name,
                    "AB": to_int(cols[4]),
                    "H": to_int(cols[6]),
                    "2B": to_int(cols[7]),
                    "3B": to_int(cols[8]),
                    "HR": to_int(cols[9]),
                    "TB": to_int(cols[10]),
                    "BB": to_int(cols[16]),
                    "HBP": to_int(cols[18]),
                    "SF": to_int(cols[15])
                }
                all_players.append(player_data)
            
            time.sleep(0.5) # マナーとして待機
        except Exception as e:
            print(f"  Error scraping {team_name}: {e}")

    if not all_players:
        print("\nError: 選手データを取得できませんでした。構造を確認してください。")
        return

    df = pd.DataFrame(all_players)

    # リーグ平均の算出 (コールドスタート対策用)
    summary = df.sum(numeric_only=True)
    player_count = len(df)
    
    avg_stats = {
        "team": "NPB",
        "name": "LEAGUE_AVERAGE",
        "AB": summary["AB"] / player_count,
        "H": summary["H"] / player_count,
        "2B": summary["2B"] / player_count,
        "3B": summary["3B"] / player_count,
        "HR": summary["HR"] / player_count,
        "TB": summary["TB"] / player_count,
        "BB": summary["BB"] / player_count,
        "HBP": summary["HBP"] / player_count,
        "SF": summary["SF"] / player_count,
    }
    
    # 平均行を末尾に追加
    df = pd.concat([df, pd.DataFrame([avg_stats])], ignore_index=True)

    # CSVとして保存
    output_file = "initial_stats_2024.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n完了！ {len(df)-1} 名の選手データを '{output_file}' に保存しました。")

if __name__ == "__main__":
    scrape_2024_stats()