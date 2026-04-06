import json
import re
import requests
from bs4 import BeautifulSoup


URL = "https://baseball.yahoo.co.jp/npb/game/2021038670/text"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
}


PREGAME_NAME_PATTERN = re.compile(r"\d番[:：]\s*([^、（）()]+)")


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
            lineups.append(
                {
                    "side": "先攻" if line.startswith("先攻:") else "後攻" if line.startswith("後攻:") else "",
                    "players": lineup_players,
                    "text": line,
                }
            )
            continue

        if "先発ピッチャー" in line:
            match = re.search(r"の\s*([^、\s]+).*?で\s*([^、\s]+)", line)
            if match:
                for name in match.groups():
                    name = normalize_name(name)
                    pitchers.append(name)
                    player_names.append(name)

    # 重複を落として順序を保つ
    unique_player_names = list(dict.fromkeys(player_names))

    return {
        "raw_lines": lines,
        "pitchers": pitchers,
        "lineups": lineups,
        "player_names": unique_player_names,
    }


def extract_scoreboard(soup: BeautifulSoup) -> list[dict]:
    table = soup.select_one("table#ing_brd")
    if table is None:
        return []

    rows = []
    for tr in table.select("tbody tr"):
        team_tag = tr.select_one(".bb-gameScoreTable__team")
        team = team_tag.get_text(strip=True) if team_tag else ""

        inning_scores = []
        score_cells = tr.select("td.bb-gameScoreTable__data")
        for cell in score_cells[1:10]:
            text = cell.get_text(" ", strip=True)
            inning_scores.append(text)

        total_cells = tr.select("td.bb-gameScoreTable__total")
        total = total_cells[0].get_text(strip=True) if len(total_cells) > 0 else ""
        hits = total_cells[1].get_text(strip=True) if len(total_cells) > 1 else ""
        errors = total_cells[2].get_text(strip=True) if len(total_cells) > 2 else ""

        rows.append(
            {
                "team": team,
                "inning_scores": inning_scores,
                "R": total,
                "H": hits,
                "E": errors,
            }
        )
    return rows


def extract_text_live(soup: BeautifulSoup) -> list[dict]:
    sections = []
    for sec in soup.select("#text_live section.bb-liveText"):
        inning_tag = sec.select_one("h1.bb-liveText__inning")
        detail_tag = sec.select_one("p.bb-liveText__detail")

        inning = inning_tag.get_text(strip=True) if inning_tag else ""
        detail = detail_tag.get_text(strip=True) if detail_tag else ""

        plays = []
        for li in sec.select("ol.bb-liveText__orderedList > li.bb-liveText__item"):
            num_tag = li.select_one("p.bb-liveText__number")
            number = num_tag.get_text(strip=True) if num_tag else ""

            # 1打席の中に複数の説明があるため、行ごとに抽出する
            lines = []
            for p in li.select(
                "p.bb-liveText__batter, p.bb-liveText__summary, p.bb-liveText__summary--point, p.bb-liveText__summary--change"
            ):
                text = p.get_text(" ", strip=True)
                if text:
                    lines.append(text)

            play = {"number": number, "lines": lines}
            plays.append(play)

        if inning == "試合前情報":
            sections.append(
                {
                    "inning": inning,
                    "detail": detail,
                    "pregame": parse_pregame_text([line for play in plays for line in play["lines"]]),
                    "plays": plays,
                }
            )
            continue

        sections.append(
            {
                "inning": inning,
                "detail": detail,
                "plays": plays,
            }
        )
    return sections


def scrape_game_text(url: str) -> dict:
    res = requests.get(url, headers=HEADERS, timeout=20)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")

    title_tag = soup.select_one("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    updated_tag = soup.select_one("time.bb-tableNote__update")
    updated_at = updated_tag.get_text(strip=True) if updated_tag else ""

    return {
        "url": url,
        "title": title,
        "updated_at": updated_at,
        "scoreboard": extract_scoreboard(soup),
        "text_live": extract_text_live(soup),
    }


if __name__ == "__main__":
    data = scrape_game_text(URL)
    output_file = "game_2021038670_text.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"saved: {output_file}")
    print(json.dumps(data, ensure_ascii=False, indent=2))