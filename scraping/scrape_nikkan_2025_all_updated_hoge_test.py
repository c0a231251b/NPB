import os
import re
import json
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from bs4 import BeautifulSoup
import unicodedata
import traceback
# 🌟【最重要修正】NameErrorの原因となっていたインポートを追加
from collections import defaultdict

class CalendarNikkanScraper:
    def __init__(self, save_dir="game_data_2025_updated_hoge_test"):
        self.save_dir = save_dir
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        self.base_url = "https://www.nikkansports.com"
        os.makedirs(save_dir, exist_ok=True)
        
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

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
        """【提供された全HTML構造を完全解読・統合した最終決定版】"""
        try:
            res = self.session.get(url, headers=self.headers, timeout=(5, 10))
            res.encoding = 'utf-8'
            if res.status_code != 200: return None
            
            raw_html_text = res.text
            soup = BeautifulSoup(raw_html_text, 'html.parser')
            
            # 1. スコアボード抽出
            score_table = soup.select_one("table.scoreTable")
            scoreboard_data = []
            teams = []
            
            if score_table:
                rows = score_table.select("tr")
                for row in rows:
                    td_team = row.select_one("td.team")
                    td_total = row.select_one("td.totalScore")
                    if td_team and td_total:
                        t_name = td_team.get_text(strip=True).replace(" ", "").replace("\xa0", "")
                        total_r = td_total.get_text(strip=True)
                        teams.append(t_name)
                        
                        tds = row.select("td")
                        name_idx = tds.index(td_team)
                        total_idx = tds.index(td_total)
                        
                        inning_scores = []
                        for td_score in tds[name_idx + 1 : total_idx]:
                            score_text = td_score.get_text(strip=True)
                            if score_text != "":
                                inning_scores.append(score_text)
                                
                        scoreboard_data.append({
                            "team": t_name,
                            "R": total_r,
                            "detail": inning_scores
                        })

            if len(teams) < 2: return None
            away_team, home_team = teams[0], teams[1]

            game_data = {
                "url": url,
                "stadium": "不明",
                "scoreboard": scoreboard_data,
                "text_live": [],
                "batter_stats": {away_team: [], home_team: []},
                "pitcher_stats": {}, 
                "starting_pitchers": {"表": "不明", "裏": "不明"}
            }

            info_tag = soup.select_one("p.data")
            if info_tag:
                info_text = info_tag.get_text()
                stadium_match = re.search(r'◇(?:公式戦|オープン戦)◇開始\d+時\d+分◇([^◇\n]+)◇', info_text)
                if stadium_match:
                    game_data["stadium"] = stadium_match.group(1).strip()

            # 2. 🌟投手成績パース：空行を絶対の切れ目として両チームの投手を「[球団名]投手名」で隔離保存
            pitcher_tables = soup.select("table.pitcher")
            
            first_block_pitchers = []  # 前半チーム（アウェイ）の全投手
            second_block_pitchers = [] # 後半チーム（ホーム）の全投手
            is_past_blank_tr = False   # 空行（境界線）を通過したかどうかのフラグ

            for p_table in pitcher_tables:
                caption_tag = p_table.select_one("caption")
                caption_text = caption_tag.get_text() if caption_tag else ""
                
                # キャプションから現在の所属を初期特定
                belong_team = home_team if home_team in caption_text else away_team
                
                # table.contents でHTML内のすべての物理子要素を順番通りに走査
                for child in p_table.contents:
                    if child == "\n" or not child.name:
                        continue
                    if child.name == "tr":
                        if child.find("th"): continue # 見出しヘッダーはスルー
                        
                        # 行内の全テキストをトリムし、中身が完全に空、またはtdがない行（空行）を検知！
                        row_text_clean = child.get_text().replace("\xa0", "").replace(" ", "").strip()
                        if row_text_clean == "" or not child.find("td"):
                            is_past_blank_tr = True
                            continue
                            
                        p_tds = child.select("td")
                        if not p_tds or len(p_tds) < 17: continue
                        p_name = p_tds[1].get_text(strip=True).replace(" ", "").replace(" ", "")
                        if not p_name: continue
                        
                        # 🌟山本重複を完全に防ぐ：現在のフラグ状態から正しい球団を決定
                        actual_team = home_team if is_past_blank_tr and len(pitcher_tables) == 1 else belong_team
                        unique_pitcher_key = f"[{actual_team}]{p_name}"
                        
                        try:
                            game_data["pitcher_stats"][unique_pitcher_key] = {
                                "player_name_raw": p_name,
                                "team": actual_team,
                                "win": int(p_tds[3].get_text().strip()) if p_tds[3].get_text().strip().isdigit() else 0,
                                "lose": int(p_tds[4].get_text().strip()) if p_tds[4].get_text().strip().isdigit() else 0,
                                "save": int(p_tds[5].get_text().strip()) if p_tds[5].get_text().strip().isdigit() else 0,
                                "games": int(p_tds[6].get_text().strip()) if p_tds[6].get_text().strip().isdigit() else 0,
                                "innings": p_tds[7].get_text().strip(),
                                "bf": int(p_tds[8].get_text().strip()) if p_tds[8].get_text().strip().isdigit() else 0,
                                "np": int(p_tds[9].get_text().strip()) if p_tds[9].get_text().strip().isdigit() else 0,
                                "h": int(p_tds[10].get_text().strip()) if p_tds[10].get_text().strip().isdigit() else 0,
                                "so": int(p_tds[11].get_text().strip()) if p_tds[11].get_text().strip().isdigit() else 0,
                                "bb": int(p_tds[12].get_text().strip()) if p_tds[12].get_text().strip().isdigit() else 0,
                                "hbp": int(p_tds[13].get_text().strip()) if p_tds[13].get_text().strip().isdigit() else 0,
                                "r": int(p_tds[14].get_text().strip()) if p_tds[14].get_text().strip().isdigit() else 0,
                                "er": int(p_tds[15].get_text().strip()) if p_tds[15].get_text().strip().isdigit() else 0,
                                "era": p_tds[16].get_text().strip()
                            }
                            
                            if not is_past_blank_tr:
                                first_block_pitchers.append(unique_pitcher_key)
                            else:
                                second_block_pitchers.append(unique_pitcher_key)
                        except Exception:
                            pass
                is_past_blank_tr = True

            # 🌟【ハルタさん指定の絶対原則】空行で分かれた各ブロックの「先頭(0番目)」を先発投手とする
            if first_block_pitchers:
                game_data["starting_pitchers"]["裏"] = game_data["pitcher_stats"][first_block_pitchers[0]]["player_name_raw"]
            if second_block_pitchers:
                game_data["starting_pitchers"]["表"] = game_data["pitcher_stats"][second_block_pitchers[0]]["player_name_raw"]

            # 3. 野手成績パース（hr対応、計・残塁等の除外を維持）
            batter_tables = soup.select("table.batter")
            stamen_lineups = {away_team: [], home_team: []}
            timeline_list = []


            for table in batter_tables:
                cap_tag = table.select_one("caption")
                cap_text = cap_tag.get_text() if cap_tag else ""
                t_name_clean = away_team if away_team in cap_text else home_team
                side_label = "表" if t_name_clean == away_team else "裏"
                
                header_ths = table.select("tr th")
                inning_headers = [unicodedata.normalize('NFKC', th.get_text().strip()) for th in header_ths]
                
                inning_col_indices = []
                inning_numbers = []
                for idx, h in enumerate(inning_headers):
                    if h.isdigit():
                        inning_col_indices.append(idx)
                        inning_numbers.append(int(h))

                rows = table.select("tr")
                pure_stamen_9 = []
                for row in rows:
                    cols = row.select("td")
                    name_td = row.select_one("td.left")
                    if not name_td or len(cols) == 0: continue
                    p_name = name_td.get_text(strip=True).replace(" ", "").replace(" ", "")
                    if p_name in ["計", "残塁", "併殺"] or "残塁" in p_name or "併殺" in p_name: continue
                    
                    pos_text = cols[0].get_text(strip=True)
                    if ('（' in pos_text or '(' in pos_text) and len(pure_stamen_9) < 9:
                        if p_name not in pure_stamen_9: pure_stamen_9.append(p_name)
                
                stamen_lineups[t_name_clean] = pure_stamen_9

                player_order_labels = {}
                for row in rows:
                    cols = row.select("td")
                    name_td = row.select_one("td.left")
                    if not name_td or len(cols) == 0: continue
                    p_name = name_td.get_text(strip=True).replace(" ", "").replace(" ", "")
                    if p_name in ["計", "残塁", "併殺"] or "残塁" in p_name or "併殺" in p_name: continue
                    
                    pos_text = cols[0].get_text(strip=True)
                    if p_name in pure_stamen_9:
                        player_order_labels[p_name] = f"{pure_stamen_9.index(p_name) + 1}番"
                    elif "投" in pos_text:
                        player_order_labels[p_name] = "投"
                    else:
                        player_order_labels[p_name] = "打"

                for row in rows:
                    cols = row.select("td")
                    name_td = row.select_one("td.left")
                    if not name_td or len(cols) < 9: continue
                    p_name = name_td.get_text(strip=True).replace(" ", "").replace(" ", "")
                    if p_name in ["計", "残塁", "併殺"] or "残塁" in p_name or "併殺" in p_name: continue
                    
                    try:
                        game_data["batter_stats"][t_name_clean].append({
                            "player_name": p_name,
                            "ab": int(cols[3].get_text().strip()) if cols[3].get_text().strip().isdigit() else 0,
                            "h": int(cols[5].get_text().strip()) if cols[5].get_text().strip().isdigit() else 0,
                            "rbi": int(cols[6].get_text().strip()) if cols[6].get_text().strip().isdigit() else 0,
                            "avg": cols[7].get_text().strip(),
                            "hr": int(cols[8].get_text().strip()) if cols[8].get_text().strip().isdigit() else 0
                        })
                    except Exception:
                        pass

                for loop_idx, real_inning_num in enumerate(inning_numbers):
                    th_absolute_idx = inning_col_indices[loop_idx]
                    for row in rows:
                        all_cells = row.find_all(['th', 'td'])
                        name_td = row.select_one("td.left")
                        if not name_td or len(all_cells) <= th_absolute_idx: continue
                        p_name = name_td.get_text(strip=True).replace(" ", "").replace(" ", "")
                        if p_name in ["計", "残塁", "併殺"] or "残塁" in p_name or "併殺" in p_name: continue
                        
                        label = player_order_labels.get(p_name, "打")
                        if label == "投": continue
                        
                        target_col = all_cells[th_absolute_idx]
                        res_text = target_col.get_text().replace("\xa0", "").strip()
                        
                        if res_text and res_text not in ["……", " ", "---", "", " "] and re.search(r'[\u4e00-\u9fff\u30a0-\u30ff]', res_text):
                            timeline_list.append({
                                "inn": real_inning_num,
                                "side": side_label,
                                "line": f"{label} {p_name}",
                                "result": res_text
                            })
                if "cl202532901" in url:
                    print("\n====3回裏確認====")
                    for e in timeline_list:
                        if e["inn"] == 3 and e["side"] == "裏":
                            print(e)

            # ==================== 4. 🌟【時系列ワープ完全修正】テキスト速報に基づく時系列マージ ====================
            live_text_pool = soup.get_text()
            
            pregame_lineups = []
            for t in [away_team, home_team]:
                p_list = [b["player_name"] for b in game_data["batter_stats"][t][:9]]
                pregame_lineups.append({"team": t, "players": p_list})
            game_data["text_live"].append({"inning": "試合前", "pregame": {"lineups": pregame_lineups}})

            # テキストライブの有効行から、現実のバッターボックス進行順のテキスト行を一列に回収
            raw_lines = []
            for tag in soup.select(".liveText, tr, td.left, p, div"):
                txt = tag.get_text().strip()
                if not txt: continue
                if any(k in txt for k in ["回表", "回裏", "投手：", "番", "打 "]) and len(txt) < 100:
                    if txt not in raw_lines: raw_lines.append(txt)

            current_inning = "試合前"
            inning_plays_map = defaultdict(list)

            for line in raw_lines:
                inn_match = re.search(r'(\d+回[表裏])', line)
                if inn_match:
                    current_inning = inn_match.group(1)
                    continue
                
                if "投手：" in line:
                    continue

                batter_match = re.search(r'(\d+番\s*\S+|打\s*\S+)', line)
                if batter_match:
                    parts = line.split(None, 1)
                    if len(parts) < 2: parts = [line, "結果不明"]
                    b_line = parts[0].strip()
                    res_line = parts[1].strip()
                    
                    inning_plays_map[current_inning].append({
                        "lines": [b_line, res_line]
                    })

            # ハルタさんが提示してくれた 3回表「茂木→中村」で終わる巡回ルールを、
            # 最初からテキストライブの物理配列順にマージすることで時系列の歪みを完全治療！
            for inn_title in sorted(inning_plays_map.keys(), key=lambda x: (int(re.search(r'\d+', x).group()), '裏' in x)):
                plays_list = []
                for p in inning_plays_map[inn_title]:
                    plays_list.append({"lines": p["lines"]})
                game_data["text_live"].append({
                    "inning": inn_title,
                    "plays": plays_list
                })

            return game_data
        except Exception as e:
            print(f"\n❌ パース中に想定外のエラーが発生しました URL: {url}")
            print(traceback.format_exc())
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
            
            if os.path.exists(save_path):
                success += 1
                continue
            
            data = self.parse_game_page(url)
            if data:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                success += 1
                print(f"\r再スクレイピング完了: {success}/{total} 試合 (進行中: {i}/{total})", end="")
                time.sleep(0.2)

        print(f"\n完了！ 最終保存試合数: {success}")

if __name__ == "__main__":
    scraper = CalendarNikkanScraper()
    files = ["url_list/Schedule_Nav_URL_Central_League.txt", "url_list/Schedule_Nav_URL_Pacific_League.txt"]
    scraper.run(files)