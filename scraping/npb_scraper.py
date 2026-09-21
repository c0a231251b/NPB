import time
import csv
import re
import requests
from bs4 import BeautifulSoup

TEAMS = [
    ("巨人",         "https://www.my-favorite-giants.net/npb/roster/2025/G.htm"),
    ("ヤクルト",     "https://www.my-favorite-giants.net/npb/roster/2025/S.htm"),
    ("横浜DeNA",     "https://www.my-favorite-giants.net/npb/roster/2025/YB.htm"),
    ("中日",         "https://www.my-favorite-giants.net/npb/roster/2025/D.htm"),
    ("阪神",         "https://www.my-favorite-giants.net/npb/roster/2025/T.htm"),
    ("広島",         "https://www.my-favorite-giants.net/npb/roster/2025/C.htm"),
    ("日本ハム",     "https://www.my-favorite-giants.net/npb/roster/2025/F.htm"),
    ("楽天",         "https://www.my-favorite-giants.net/npb/roster/2025/E.htm"),
    ("西武",         "https://www.my-favorite-giants.net/npb/roster/2025/L.htm"),
    ("ロッテ",       "https://www.my-favorite-giants.net/npb/roster/2025/M.htm"),
    ("オリックス",   "https://www.my-favorite-giants.net/npb/roster/2025/Bs.htm"),
    ("ソフトバンク", "https://www.my-favorite-giants.net/npb/roster/2025/H.htm"),
]

OUTPUT_CSV    = "npb_roster_2025.csv"
REQUEST_DELAY = 2.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "Referer": "https://www.my-favorite-giants.net/npb/roster/2025/top.htm",
}

def is_ikanka(num_str):
    s = num_str.strip()
    if len(s) >= 3 and s.startswith("0"):
        return False
    return bool(re.fullmatch(r"\d{1,2}", s))

def scrape_team(session, team_name, url):
    res = session.get(url, timeout=15)
    res.raise_for_status()
    soup = BeautifulSoup(res.content, "lxml")

    table = soup.find("table", {"id": "roster"})
    if table is None:
        print(f"  [WARN] table#roster が見つかりません: {url}")
        return []

    tbody = table.find("tbody")
    if tbody is None:
        return []

    results = []
    for tr in tbody.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 8:
            continue
        num_str = cells[0].get_text(strip=True)
        name    = cells[2].get_text(strip=True)
        batting = cells[7].get_text(strip=True)
        if not num_str or not name:
            continue
        if not is_ikanka(num_str):
            continue
        name = name.replace("＊", "").strip()
        results.append({"チーム名": team_name, "背番号": num_str, "選手名": name, "投打": batting})

    return results

def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    all_rows = []
    for team_name, url in TEAMS:
        print(f"取得中: {team_name} ...")
        try:
            rows = scrape_team(session, team_name, url)
            print(f"  → {len(rows)} 名（支配下）")
            all_rows.extend(rows)
        except Exception as e:
            print(f"  [ERROR] {e}")
        time.sleep(REQUEST_DELAY)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["チーム名", "背番号", "選手名", "投打"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n完了: {len(all_rows)} 件を '{OUTPUT_CSV}' に保存しました。")

if __name__ == "__main__":
    main()