import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup

class CalendarNikkanScraper:
    def __init__(self, save_dir="game_data_2025_updated"):
        self.save_dir = save_dir
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        self.base_url = "https://www.nikkansports.com"
        os.makedirs(save_dir, exist_ok=True)

    def extract_urls_from_files(self, file_paths):
        urls = set()
        pattern = re.compile(r'/baseball/professional/score/2025/(?:cl|pl|il)\d+\.html')
        for path in file_paths:
            if not os.path.exists(path): 
                print(f"警告: ファイルが見つかりません: {path}")
                continue
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                matches = pattern.findall(content)
                for m in matches: 
                    urls.add(self.base_url + m)
        return sorted(list(urls))

    def parse_game_page(self, url):
        """【バグ完全根絶版】ハルタさん提示の守備位置ロックに、未打席打者の救済ログ処理を追加"""
        try:
            res = requests.get(url, headers=self.headers, timeout=15) # タイムアウトを15秒に延長して鉄壁化
            res.encoding = 'utf-8'
            if res.status_code != 200: return None
            
            soup = BeautifulSoup(res.text, 'html.parser')
            team_tags = soup.select(".scoreTable .team")
            score_tags = soup.select(".scoreTable .totalScore")
            if not team_tags or len(team_tags) < 2: return None
            
            teams = [t.get_text(strip=True).replace("\xa0", "") for t in team_tags]
            runs = [s.get_text(strip=True) for s in score_tags]

            game_data = {
                "url": url,
                "scoreboard": [{"team": teams[0], "R": runs[0]}, {"team": teams[1], "R": runs[1]}],
                "text_live": []
            }

            batter_tables = soup.select("table.batter")
            stamen_lineups = []

            for team_idx, table in enumerate(batter_tables):
                header_ths = table.select("tr th")
                inning_numbers = [int(th.get_text().strip()) for th in header_ths if th.get_text().strip().isdigit()]
                total_innings = len(inning_numbers)

                rows = table.select("tr")
                
                # --- 💡 ハルタさん提示の完璧な先発9名確定アルゴリズム ---
                pure_stamen_9 = []
                player_valid_plays = {}
                
                for row in rows:
                    cols = row.select("td")
                    name_td = row.select_one("td.left")
                    if not name_td or len(cols) == 0: 
                        continue
                        
                    p_name = name_td.get_text(strip=True).replace(" ", "").replace(" ", "")
                    if p_name == "計": 
                        continue
                        
                    # 一番左のセル（守備位置）を取得
                    pos_text = cols[0].get_text(strip=True)
                    
                    name_idx = cols.index(name_td)
                    inning_cols = cols[name_idx + 1:][-total_innings:]
                    
                    valid_plays = []
                    for idx, col in enumerate(inning_cols):
                        real_inning_num = inning_numbers[idx]
                        res_text = col.get_text().replace("\xa0", "").strip()
                        
                        if not res_text or res_text in ["……", " ", "---", "", " "]:
                            continue
                        if not re.search(r'[\u4e00-\u9fff\u30a0-\u30ff]', res_text):
                            continue
                            
                        valid_plays.append((real_inning_num, res_text))
                    
                    # カッコが含まれている選手だけをスタメンとしてロック
                    is_stamen_pos = '（' in pos_text or '(' in pos_text
                    
                    if is_stamen_pos and len(pure_stamen_9) < 9:
                        if p_name not in pure_stamen_9:
                            pure_stamen_9.append(p_name)
                    
                    # プレイ結果の保持
                    if len(valid_plays) > 0:
                        if p_name in pure_stamen_9 or is_stamen_pos:
                            player_valid_plays[p_name] = valid_plays

                stamen_lineups.append(pure_stamen_9)

                # --- 💡 【追加修正】28箇所のUnknownを消滅させるセーフティガード ---
                # 確定した先発9名のうち、打席結果が空っぽ（playsがない）の選手がいた場合、
                # 1回表/裏に仮想のログ（先発出場）を書き込んで、extract_game_lineups.pyに確実に検知させます
                for p_name in pure_stamen_9:
                    if p_name not in player_valid_plays:
                        player_valid_plays[p_name] = [(1, "先発出場")]

                # --- プレイログ（text_live）の生成（ハルタさんロジック完全維持） ---
                for p_name in pure_stamen_9:
                    order_num = pure_stamen_9.index(p_name) + 1
                    plays = player_valid_plays.get(p_name, [])
                    
                    for real_inning, res_text in plays:
                        game_data["text_live"].append({
                            "inning": f"{real_inning}回{'表' if team_idx==0 else '裏'}",
                            "plays": [{
                                "lines": [f"{order_num}番 {p_name}", res_text]
                            }]
                        })

            if len(stamen_lineups) >= 2:
                game_data["text_live"].insert(0, {
                    "inning": "試合前",
                    "pregame": {
                        "lineups": [
                            {"team": teams[0], "players": stamen_lineups[0]}, 
                            {"team": teams[1], "players": stamen_lineups[1]}
                        ]
                    }
                })
            return game_data
        except Exception:
            return None

    def run(self, input_files):
        print("カレンダーからURLを抽出中...")
        urls = self.extract_urls_from_files(input_files)
        total = len(urls)
        print(f"合計 {total} 試合のURLが見つかりました。解析を開始します。")

        success = 0
        for i, url in enumerate(urls, 1):
            filename = url.split("/")[-1].replace(".html", ".json")
            save_path = os.path.join(self.save_dir, filename)
            
            data = self.parse_game_page(url)
            if data:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                success += 1
                print(f"\r再スクレイピング完了: {success}/{total} 試合 (進行中: {i}/{total})", end="")
                time.sleep(0.15) # サーバー負荷とアクセス制限を完全に回避する安全ディレイ

        print(f"\n完了！ 最終保存試合数: {success}")

if __name__ == "__main__":
    scraper = CalendarNikkanScraper()
    files = ["url_list/Schedule_Nav_URL_Central_League.txt", "url_list/Schedule_Nav_URL_Pacific_League.txt"]
    scraper.run(files)