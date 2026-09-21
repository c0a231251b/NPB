import os
import json
import re
import requests
from bs4 import BeautifulSoup

def parse_single_game():
    url = "https://www.nikkansports.com/baseball/professional/score/2025/cl2025032802.html"
    headers = {"User-Agent": "Mozilla/5.0"}
    save_dir = "a"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🎯 開幕戦1試合の精密スクレイピングをテストします（日本語プレイ結果ポジティブマッチ版）...\nURL: {url}")
    
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = 'utf-8'
        if res.status_code != 200:
            print("Error: ページの取得に失敗しました。")
            return
            
        soup = BeautifulSoup(res.text, 'html.parser')
        team_tags = soup.select(".scoreTable .team")
        
        teams = [t.get_text(strip=True).replace("\xa0", "") for t in team_tags]
        runs = [s.get_text(strip=True) for s in soup.select(".scoreTable .totalScore")]

        game_data = {
            "url": url,
            "scoreboard": [{"team": teams[0], "R": runs[0]}, {"team": teams[1], "R": runs[1]}],
            "text_live": []
        }

        batter_tables = soup.select("table.batter")
        stamen_lineups = []

        for team_idx, table in enumerate(batter_tables):
            # ヘッダー（th）から「本当のイニング番号」を取得
            inning_numbers = [int(th.get_text().strip()) for th in table.select("tr th") if th.get_text().strip().isdigit()]
            total_innings = len(inning_numbers)

            rows = table.select("tr")
            
            pure_stamen_9 = []
            player_valid_plays = {} # { 選手名: [ (イニング, 結果), ... ] }
            
            for row in rows:
                name_td = row.select_one("td.left")
                if not name_td:
                    continue
                p_name = name_td.get_text(strip=True).replace(" ", "").replace(" ", "")
                if p_name == "計":
                    continue
                    
                cols = row.select("td")
                name_idx = cols.index(name_td)
                # 後ろからイニング列の数だけ正確に引き抜く
                inning_cols = cols[name_idx + 1:][-total_innings:]
                
                valid_plays = []
                for idx, col in enumerate(inning_cols):
                    real_inning_num = inning_numbers[idx]
                    res_text = col.get_text().replace("\xa0", "").strip()
                    
                    # ーーー ★ここがバグの完全修正ロジック★ ーーー
                    # 不確かな空白判定をすべて捨て、「漢字またはカタカナ（一ゴ、中飛、サンタナ等）」
                    # が含まれているマスだけを【有効な打席結果】として100%厳密に判定します。
                    # これにより、&nbsp; や空白、ドット、ハイフンなどのノイズは『文字数が0』、
                    # またはマッチしないため、山本・木澤らの投手陣は完全にここで「スルー（無視）」されます。
                    if not re.search(r'[\u4e00-\u9fff\u30a0-\u30ff]', res_text):
                        continue
                    
                    valid_plays.append((real_inning_num, res_text))
                
                # 本当に打席結果（プレイ）が存在した打者だけを上から順にスタメン登録
                if len(valid_plays) > 0 and len(pure_stamen_9) < 9:
                    if p_name not in pure_stamen_9:
                        pure_stamen_9.append(p_name)
                        player_valid_plays[p_name] = valid_plays

            stamen_lineups.append(pure_stamen_9)
            print(f"【{teams[team_idx]}の真のスタメン】: {pure_stamen_9}")

            # --- プレイログ（text_live）の生成 ---
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

        save_path = os.path.join(save_dir, "cl2025032802.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(game_data, f, ensure_ascii=False, indent=2)
            
        print("\n" + "="*50)
        print(f"💾 【ハルタさんロジック完全準拠】バグ修正版JSONを保存しました")
        print(f"保存先: {save_path}")
        print("="*50)

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    parse_single_game()