import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup

class CalendarNikkanScraper:
    def __init__(self, save_dir="game_data_2025"):
        self.save_dir = save_dir
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.base_url = "https://www.nikkansports.com"
        os.makedirs(save_dir, exist_ok=True)

    def extract_urls_from_files(self, file_paths):
        """テキストファイルから試合ページのURLを全て抽出する"""
        urls = set()
        # 2025年の試合スコアURLのパターン (cl, pl, il対応)
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
        """指定されたURLから試合情報を抽出する (以前の改良版ロジック)"""
        try:
            res = requests.get(url, headers=self.headers, timeout=10)
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
            lineups = []
            for team_idx, table in enumerate(batter_tables):
                rows = table.select("tr")
                player_names = []
                for row in rows:
                    name_td = row.select_one("td.left")
                    cols = row.select("td")
                    if not name_td or len(cols) < 10: continue
                    p_name = name_td.get_text(strip=True)
                    if p_name == "計": continue
                    
                    if len(player_names) < 9 and p_name not in player_names:
                        player_names.append(p_name)

                    for inning_idx, col in enumerate(cols[9:], 1):
                        res_text = col.get_text(strip=True)
                        if res_text and res_text not in ["……", " ", "---", "\xa0"]:
                            game_data["text_live"].append({
                                "inning": f"{inning_idx}回{'表' if team_idx==0 else '裏'}",
                                "plays": [{"lines": [f"{len(player_names)}番 {p_name}", res_text]}]
                            })
                lineups.append(player_names)

            if len(lineups) >= 2:
                game_data["text_live"].insert(0, {
                    "inning": "試合前",
                    "pregame": {"lineups": [{"team": teams[0], "players": lineups[0]}, {"team": teams[1], "players": lineups[1]}]}
                })
            return game_data
        except:
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
            
            # 既存の有効なファイル(打席あり)があればスキップ
            if os.path.exists(save_path):
                with open(save_path, "r", encoding="utf-8") as f:
                    try:
                        temp = json.load(f)
                        if any("plays" in inn for inn in temp.get("text_live", [])):
                            success += 1
                            continue
                    except: pass

            data = self.parse_game_page(url)
            if data:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                success += 1
                print(f"\r取得完了: {success}/{total} 試合 (進行中: {i}/{total})", end="")
                time.sleep(0.5) # サーバー負荷軽減

        print(f"\n完了！ 最終保存試合数: {success}")

if __name__ == "__main__":
    scraper = CalendarNikkanScraper()
    # アップロードされた2つのファイルを指定
    files = ["日程ナビ URL セリーグ.txt", "日程ナビ URL パリーグ.txt"]
    scraper.run(files)