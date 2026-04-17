import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time

def scrape_pitcher_data():
    # リーグとチームの設定
    teams_config = {
        "Central": ["G", "T", "DB", "C", "S", "D"],
        "Pacific": ["B", "M", "H", "E", "L", "F"]
    }
    
    # ループする背番号リスト (00, 0, 1-99)
    numbers = ["00", "0"] + [str(i) for i in range(1, 100)]
    
    all_pitchers_list = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("スクレイピングを開始します。完了まで時間がかかります...")

    for league, teams in teams_config.items():
        for team in teams:
            print(f"現在解析中: {league} - {team}")
            for num in numbers:
                url = f"https://nf3.sakura.ne.jp/2024/{league}/{team}/p/{num}_stat.htm"
                
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    # ページが存在しない(404)場合はスキップ
                    if response.status_code != 200:
                        continue
                    
                    response.encoding = 'utf-8'
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # 選手名がタイトルに含まれているか確認（背番号が空き番号の場合の対策）
                    title_text = soup.title.string
                    if "投手成績" not in title_text:
                        continue

                    # --- データ抽出ロジック ---
                    # 1. 選手名・利き手
                    name_match = re.search(r'投手成績 \d+ (.*?) -', title_text)
                    name = name_match.group(1).strip() if name_match else "不明"
                    page_text = soup.get_text()
                    hand = "右投" if "右投" in page_text else "左投" if "左投" in page_text else "不明"

                    # 2. 指標テーブル (K/9, HR/9)
                    stats_table_div = soup.find('div', string=re.compile('通算成績\(各種指標\)')) or \
                                      soup.find('caption', string=re.compile('通算成績\(各種指標\)'))
                    
                    if not stats_table_div:
                        continue # 指標テーブルがない選手（登板なし等）はスキップ

                    target_table = stats_table_div.find_parent('table')
                    ths = [th.text.strip() for th in target_table.find_all('th')]
                    tds = [td.text.strip() for td in target_table.find_all('td')]

                    k9 = float(tds[ths.index('K/9')]) if 'K/9' in ths else 0.0
                    hr9 = float(tds[ths.index('HR/9')]) if 'HR/9' in ths else 0.0

                    # 3. 防御率 (ERA)
                    era_table = soup.find('th', string=re.compile('防御率')).find_parent('table')
                    era_text = era_table.find_all('td')[0].text.strip()
                    era = float(era_text) if era_text != "-" else 9.99

                    # 4. 球種と割合
                    pitch_data = {}
                    pitch_table_th = soup.find('th', string=re.compile('球種'))
                    if pitch_table_th:
                        pitch_table = pitch_table_th.find_parent('table')
                        for tr in pitch_table.find_all('tr'):
                            tds_p = tr.find_all('td')
                            if len(tds_p) >= 3:
                                p_type = tds_p[0].text.strip()
                                p_share = tds_p[2].text.strip().replace('%', '')
                                if p_type != '合計' and p_share != "-":
                                    pitch_data[f"pitch_{p_type}_share"] = float(p_share)

                    # データの結合
                    pitcher_info = {
                        "team": team,
                        "name": name,
                        "hand": hand,
                        "ERA": era,
                        "K/9": k9,
                        "HR/9": hr9,
                        **pitch_data
                    }
                    all_pitchers_list.append(pitcher_info)
                    print(f"  取得成功: {name} (背番号:{num})")
                    
                    # サーバー負荷軽減のための待機
                    time.sleep(1)

                except Exception as e:
                    print(f"  エラー (URL: {num}): {e}")
                    continue

    # CSVとして保存
    if all_pitchers_list:
        df = pd.DataFrame(all_pitchers_list)
        # NaN（その球種を持っていない選手）を0で埋める
        df = df.fillna(0)
        
        filename = "pitcher_stats_2024_all.csv"
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n全データの取得が完了しました！")
        print(f"保存先: {filename}")
        print(f"合計選手数: {len(df)}")
    else:
        print("データが取得できませんでした。")

if __name__ == "__main__":
    scrape_pitcher_data()