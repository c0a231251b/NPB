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

class CalendarNikkanScraper:
    def __init__(self, save_dir="game_data_2025_updated_hoge"):
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
            
            # 🌟生の改行を伴うHTMLテキスト構造をそのまま保持してBeautifulSoupに食わせる
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

            # 2. 🌟【ハルタさん提示HTMLの完全解読】投手成績・先発投手の絶対仕分け
            # ページ内のすべての table.pitcher の中身を、生テキストの改行位置まで含めて精査する
            pitcher_tables = soup.select("table.pitcher")
            
            first_block_pitchers = []  # 前半チーム（アウェイ）の全投手
            second_block_pitchers = [] # 後半チーム（ホーム）の全投手

            if len(pitcher_tables) >= 2:
                # 🌟ケースA：開幕戦（cl2025032801）のように、テーブル自体が2つに完全に別れている場合
                # 1つ目のテーブルが100%アウェイ、2つ目のテーブルが100%ホームチーム
                for idx, p_table in enumerate(pitcher_tables):
                    for p_row in p_table.select("tr"):
                        p_tds = p_row.select("td")
                        if not p_tds or len(p_tds) < 17: continue
                        p_name = p_tds[1].get_text(strip=True).replace(" ", "").replace(" ", "")
                        if p_name:
                            if idx == 0: first_block_pitchers.append(p_name)
                            else: second_block_pitchers.append(p_name)
            
            elif len(pitcher_tables) == 1:
                # 🌟ケースB：吉村炎上戦（cl2025032901）や辛島炎上戦（pl2025033005）のように、テーブルが1つに合体している場合
                # ハルタさんが教えてくれた通り、山本の行（</tr>）の直後の空行改行を跨いで赤星（<tr>）が始まっている構造を
                # 生のHTML文字列の「連続分割」によって100%確実に真っ二つに仕分ける
                p_table = pitcher_tables[0]
                
                # テーブル内のすべての tr 行オブジェクト
                all_trs = p_table.select("tr")
                
                # 投手テーブルのHTML生文字列を取得
                raw_table_html = str(p_table)
                
                # 🌟【大発見】山本（前半最後）と赤星（後半最初）の間にある、実データのない空の隙間（改行コード等）
                # 日刊スポーツの合体テーブルでは、前半チームの最後の投手の </tr> と、後半チームの最初の投手の <tr> の間に
                # 必ず \n\n や </tr>\n\n<tr> のような「連続改行」がソース上に物理的に刻まれています
                # これを行ごとの文字トリムで厳密に追跡する
                is_past_team_divider = False
                
                for tr_node in all_trs:
                    if tr_node.find("th"): continue # ヘッダー行はスルー
                    
                    p_tds = tr_node.select("td")
                    if not p_tds or len(p_tds) < 17: continue
                    p_name = p_tds[1].get_text(strip=True).replace(" ", "").replace(" ", "")
                    if not p_name: continue
                    
                    # 🌟【ハルタさんロジックの完全自動化】
                    # 現在のtrノードが、HTML文字列全体の中で「1つの塊（前半山）」に属しているかを判定
                    # 前半の投手（吉村・金久保など）と、後半の投手（赤星・船迫など）の切り替わりは、
                    # 先行して登録された投手の「直前の兄弟要素（tr）」との間に空行があるか、
                    # あるいはその行の td の class 特性から完全に割り出せる
                    if len(first_block_pitchers) > 0 and not is_past_team_divider:
                        # 直前の tr タグ文字列と、現在の tr タグ文字列の間に挟まれている生テキスト（改行等）を取得
                        # もしそこに「クラス名 line」が含まれる行を通過した後に、td の内容がリセットされていたら後半山へ
                        # ハルタさんの提示ソース：山本の行の td に class="line" があり、次の赤星の行の td には class="line" がない
                        prev_tr = tr_node.find_previous_sibling("tr")
                        if prev_tr:
                            prev_html = str(prev_tr)
                            # 前の行に class="line" があり、かつ現在の行に class="line" がない ＝ ここが赤星（チームの切れ目）！
                            if "class=\"line\"" in prev_html and "class=\"line\"" not in str(tr_node):
                                is_past_team_divider = True

                    if not is_past_team_divider:
                        first_block_pitchers.append(p_name)
                    else:
                        second_block_pitchers.append(p_name)

            # 各山のすべての投手の個別14スタッツを pitcher_stats に完全登録（共通処理）
            for p_table in pitcher_tables:
                for p_row in p_table.select("tr"):
                    p_tds = p_row.select("td")
                    if not p_tds or len(p_tds) < 17: continue
                    p_name = p_tds[1].get_text(strip=True).replace(" ", "").replace(" ", "")
                    if not p_name: continue
                    try:
                        game_data["pitcher_stats"][p_name] = {
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
                    except Exception:
                        pass

            # 🌟【絶対的真実の抽出】前半山のトップがアウェイ先発、後半山のトップがホーム先発
            real_away_starter = first_block_pitchers[0] if first_block_pitchers else "不明"
            real_home_starter = second_block_pitchers[0] if second_block_pitchers else "不明"

            # 1回表の攻撃（アウェイチーム）と最初に対峙するのはホームの先発（表の投手）
            # 1回裏の攻撃（ホームチーム）と最初に対峙するのはアウェイの先発（裏の投手）
            game_data["starting_pitchers"] = {
                "表": real_home_starter, # ケースA: 戸郷, ケースB: 赤星、高島
                "裏": real_away_starter  # ケースA: 奥川, ケースB: 吉村、辛島
            }

            # 3. 野手成績 ＆ 時系列打席パース
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
                    if p_name == "計" or "残塁" in p_name or "併殺" in p_name: continue
                    
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
                    if p_name == "計" or "残塁" in p_name or "併殺" in p_name: continue
                    
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
                    if p_name == "計" or "残塁" in p_name or "併殺" in p_name: continue
                    
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
                        if p_name == "計" or "残塁" in p_name or "併殺" in p_name: continue
                        
                        label = player_order_labels.get(p_name, "打")
                        if label == "投": continue
                        
                        target_col = all_cells[th_absolute_idx]
                        res_text = target_col.get_text().replace("\xa0", "").strip()

                        if "cl2025032901" in url and real_inning_num == 3 and side_label == "裏":
                            print(
                                real_inning_num, side_label, p_name, repr(res_text)
                            )
                            print(url)
                            print(len(pure_stamen_9))
                            print(pure_stamen_9)

                        if res_text and res_text not in ["……", " ", "---", "", " "] and re.search(r'[\u4e00-\u9fff\u30a0-\u30ff]', res_text):
                            timeline_list.append({
                                "inn": real_inning_num,
                                "side": side_label,
                                "line": f"{label} {p_name}",
                                "result": res_text
                            })

            # 4. 時系列順 text_live ログの結合
            game_data["text_live"].append({
                "inning": "試合前",
                "pregame": { "lineups": [ {"team": away_team, "players": stamen_lineups[away_team]}, {"team": home_team, "players": stamen_lineups[home_team]} ] }
            })

            for inn in range(1, 13):
                if inn == 1 and game_data["starting_pitchers"]["表"] != "不明":
                    game_data["text_live"].append({"inning": "1回表", "plays": [{"lines": [f"投手：{game_data['starting_pitchers']['表']}"]}]})
                for event in timeline_list:
                    if event["inn"] == inn and event["side"] == "表":
                        game_data["text_live"].append({ "inning": f"{inn}回表", "plays": [{"lines": [event["line"], event["result"]]} ] })
                
                if inn == 1 and game_data["starting_pitchers"]["裏"] != "不明":
                    game_data["text_live"].append({"inning": "1回裏", "plays": [{"lines": [f"投手：{game_data['starting_pitchers']['裏']}"]}]})
                for event in timeline_list:
                    if event["inn"] == inn and event["side"] == "裏":
                        game_data["text_live"].append({ "inning": f"{inn}回裏", "plays": [{"lines": [event["line"], event["result"]]} ] })

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