import json
import requests
from bs4 import BeautifulSoup


URL = "https://baseball.yahoo.co.jp/npb/game/2021038670/text"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
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

            plays.append({"number": number, "lines": lines})

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