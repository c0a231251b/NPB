import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_pitcher_stats():
    # 12球団のURLリスト
    teams = [
        ("巨人", "https://npb.jp/bis/2024/stats/idp1_g.html"),
        ("阪神", "https://npb.jp/bis/2024/stats/idp1_t.html"),
        ("DeNA", "https://npb.jp/bis/2024/stats/idp1_db.html"),
        ("広島", "https://npb.jp/bis/2024/stats/idp1_c.html"),
        ("ヤクルト", "https://npb.jp/bis/2024/stats/idp1_s.html"),
        ("中日", "https://npb.jp/bis/2024/stats/idp1_d.html"),
        ("ソフトバンク", "https://npb.jp/bis/2024/stats/idp1_h.html"),
        ("日本ハム", "https://npb.jp/bis/2024/stats/idp1_f.html"),
        ("ロッテ", "https://npb.jp/bis/2024/stats/idp1_m.html"),
        ("楽天", "https://npb.jp/bis/2024/stats/idp1_e.html"),
        ("オリックス", "https://npb.jp/bis/2024/stats/idp1_b.html"),
        ("西武", "https://npb.jp/bis/2024/stats/idp1_l.html"),
    ]

    all_data = []

    for team_name, url in teams:
        print(f"{team_name}のデータを取得中...")
        try:
            res = requests.get(url)
            res.encoding = 'utf-8'
            soup = BeautifulSoup(res.text, 'html.parser')

            # 投手データ行（ststatsクラス）を全て取得
            rows = soup.find_all('tr', class_='ststats')

            for row in rows:
                cols = row.find_all('td')
                # 少なくとも防御率(25番目)までの列があることを確認
                if len(cols) < 26: continue

                # 名前（2番目）
                name = cols[1].get_text(strip=True).replace('　', '')
                # 登板（3番目）
                games = cols[2].get_text(strip=True)
                # 投球回（14番目：整数 + 15番目：端数）
                ip_main = cols[13].get_text(strip=True)
                ip_frac = cols[14].get_text(strip=True)
                innings = f"{ip_main}{ip_frac}"
                
                # 被安打（16番目）
                hits = cols[15].get_text(strip=True)
                # 被本塁打（17番目）
                hr = cols[16].get_text(strip=True)
                # 奪三振（21番目）
                so = cols[20].get_text(strip=True)
                # 防御率（26番目）
                era = cols[25].get_text(strip=True)

                all_data.append({
                    "team": team_name,
                    "name": name,
                    "G": games,
                    "IP": innings,
                    "H": hits,
                    "HR": hr,
                    "SO": so,
                    "ERA": era
                })
            
            time.sleep(1) # サーバーへの負荷軽減
        except Exception as e:
            print(f"{team_name}でエラーが発生しました: {e}")

    # DataFrame作成とCSV保存
    df = pd.DataFrame(all_data)
    
    # 指標のクリーニング（防御率が '-' の場合などを考慮）
    df['ERA'] = df['ERA'].replace('-', '9.99')
    
    # CSV出力
    output_file = "pitcher_stats_2024.csv"
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n完了！ {output_file} に {len(df)} 名のデータを保存しました。")

if __name__ == "__main__":
    scrape_pitcher_stats()