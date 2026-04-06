import json
import re
import requests
import os
import time
from bs4 import BeautifulSoup

# --- 設定 ---
START_ID = 2021040014
END_ID = 2021040114
OUTPUT_DIR = "game_data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
}

# ユーザーさんが作成した優秀な正規表現パターン
PREGAME_NAME_PATTERN = re.compile(r"\d番[:：]\s*([^、（）()]+)")

# --- 解析用関数群 ---

def normalize_name(name: str) -> str:
    return name.strip().replace(" ", "").replace("　", "")

def parse_pregame_text(lines: list[str]) -> dict:
    pitchers = []
    lineups = []
    player_names = []

    for line in lines:
        if "スターティングラインアップ" in line:
            lineup_players = []
            for name in PREGAME_NAME_PATTERN.findall(line):
                name = normalize_name(name)
                lineup_players.append(name)
                player_names.append(name)
            lineups.append({
                "side": "先攻" if line.startswith("先攻:") else "後攻" if line.startswith("後攻:") else "",
                "players": lineup_players,
                "text": line,
            })
            continue

        if "先発ピッチャー" in line:
            match = re.search(r"の\s*([^、\s]+).*?で\s*([^、\s]+)", line)
            if match:
                for name in match.groups():
                    name = normalize_name(name)
                    pitchers.append(name)
                    player_names.append(name)

    unique_player_names = list(dict.fromkeys(player_names))
    return {
        "raw_lines": lines,
        "pitchers": pitchers,
        "lineups": lineups,
        "player_names": unique_player_names,
    }

def extract_scoreboard(soup: BeautifulSoup) -> list[dict]:
    table = soup.select_one("table#ing_brd")
    if table is None: return []
    rows = []
    for tr in table.select("tbody tr"):
        team_tag = tr.select_one(".bb-gameScoreTable__team")
        team = team_tag.get_text(strip=True) if team_tag else ""
        inning_scores = [cell.get_text(strip=True) for cell in tr.select("td.bb-gameScoreTable__data")[1:10]]
        total_cells = tr.select("td.bb-gameScoreTable__total")
        rows.append({
            "team": team,
            "inning_scores": inning_scores,
            "R": total_cells[0].get_text(strip=True) if len(total_cells) > 0 else "",
            "H": total_cells[1].get_text(strip=True) if len(total_cells) > 1 else "",
            "E": total_cells[2].get_text(strip=True) if len(total_cells) > 2 else "",
        })
    return rows

def extract_text_live(soup: BeautifulSoup) -> list[dict]:
    sections = []
    for sec in soup.select("#text_live section.bb-liveText"):
        inning_tag = sec.select_one("h1.bb-liveText__inning")
        inning = inning_tag.get_text(strip=True) if inning_tag else ""
        plays = []
        for li in sec.select("ol.bb-liveText__orderedList > li.bb-liveText__item"):
            number = li.select_one("p.bb-liveText__number").get_text(strip=True) if li.select_one("p.bb-liveText__number") else ""
            lines = [p.get_text(" ", strip=True) for p in li.select("p.bb-liveText__batter, p.bb-liveText__summary, p.bb-liveText__summary--point, p.bb-liveText__summary--change") if p.get_text(strip=True)]
            plays.append({"number": number, "lines": lines})

        section_data = {"inning": inning, "detail": sec.select_one("p.bb-liveText__detail").get_text(strip=True) if sec.select_one("p.bb-liveText__detail") else "", "plays": plays}
        if inning == "試合前情報":
            section_data["pregame"] = parse_pregame_text([line for play in plays for line in play["lines"]])
        sections.append(section_data)
    return sections

def scrape_game_text(url: str) -> dict:
    res = requests.get(url, headers=HEADERS, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    return {
        "url": url,
        "title": soup.title.get_text(strip=True) if soup.title else "",
        "updated_at": soup.select_one("time.bb-tableNote__update").get_text(strip=True) if soup.select_one("time.bb-tableNote__update") else "",
        "scoreboard": extract_scoreboard(soup),
        "text_live": extract_text_live(soup),
    }

# --- メイン実行部 ---

def collect_games():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for game_id in range(START_ID, END_ID + 1):
        url = f"https://baseball.yahoo.co.jp/npb/game/{game_id}/text"
        print(f"Processing: {url}...")
        try:
            data = scrape_game_text(url)
            if not data.get("scoreboard"):
                print(f"  -> Skipping: No data (Game ID: {game_id})")
                continue

            filename = f"{OUTPUT_DIR}/game_{game_id}_text.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved: {filename}")
        except Exception as e:
            print(f"  -> Error: {e}")
        
        time.sleep(1.5) # サーバーへの優しさ

if __name__ == "__main__":
    collect_games()