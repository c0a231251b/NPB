import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_initial_stats():
    # 2025年度のチーム名とURLの対応マップ
    # チーム名は後のマッチング精度を高めるため、JSON側の表記に合わせています
    team_urls = {
        "巨人": "https://npb.jp/bis/2025/stats/idb1_g.html",
        "阪神": "https://npb.jp/bis/2025/stats/idb1_t.html",
        "ＤｅＮＡ": "https://npb.jp/bis/2025/stats/idb1_db.html",
        "広島": "https://npb.jp/bis/2025/stats/idb1_c.html",
        "ヤクルト": "https://npb.jp/bis/2025/stats/idb1_s.html",
        "中日": "https://npb.jp/bis/2025/stats/idb1_d.html",
        "ソフトバンク": "https://npb.jp/bis/2025/stats/idb1_h.html",
        "日本ハム": "https://npb.jp/bis/2025/stats/idb1_f.html",
        "ロッテ": "https://npb.jp/bis/2025/stats/idb1_m.html",
        "楽天": "https://npb.jp/bis/2025/stats/idb1_e.html",
        "オリックス": "https://npb.jp/bis/2025/stats/idb1_b.html",
        "西武": "https://npb.jp/bis/2025/stats/idb1_l.html"
    }

    all_players = []

    for team_name, url in team_urls.items():
        print(f"Scraping {team_name}: {url}")
        try:
            res = requests.get(url)
            res.encoding = "utf-8"
            soup = BeautifulSoup(res.text, "html.parser")

            # テーブルの取得 
            table = soup.find("table", class_="tablefix2")
            if not table:
                continue

            # データ行の抽出
            rows = table.find("tbody").find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if not cols:
                    continue
                
                # 選手名のクレンジング（スペースや注釈記号の除去）
                name = cols[0].get_text(strip=True).replace("\u3000", "").replace("*", "").replace("+", "")
                
                try:
                    # 指標: チーム, 名前, 打数, 安打, 二塁打, 三塁打, 本塁打, 塁打, 四球, 死球, 犠飛
                    data = {
                        "team": team_name,              # 追加したチーム情報
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
        except Exception as e:
            print(f"Error scraping {team_name}: {e}")
            continue

    df = pd.DataFrame(all_players)

    # --- 実績なし選手用の「リーグ平均」を算出 ---
    summary = df.sum(numeric_only=True)
    player_count = len(df)
    
    avg_stats = {
        "team": "NPB", # 平均行のチーム名はNPBとする
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
    print(f"\n完了！ {len(df)-1}名の選手データを 'initial_stats_2025.csv' に保存しました。")

if __name__ == "__main__":
    scrape_initial_stats()