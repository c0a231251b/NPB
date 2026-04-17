# NPB
## 概要
打席履歴と投手の特徴量を用いた打順最適化を行う.

## 実行手順
### 1. データを取得
- 2024_initial_stats_scraper.py → initial_stats_2024.csv
- scrape_nikkan_2025_all.py → game_data_2025/.json
### 2. データ確認
- at_bats.py →ファイル数/総打席数を確認
### 3. 学習
- game_score_lstm.py 


## ファイル概要
### python
|ファイル名|概要|
|---|---|
|2024_initial_stats_scraper.py|2024年NPBの公式サイトから12球団の選手打撃成績をスクレイピング　してCSVに保存するスクリプト|
|2024_pitcher_stats_scraper.py|2024年NPBの公式サイトから12球団の選手投手成績をスクレイピングしてCSVに保存するスクリプト|
|2024_pitcher_stats_scraper_all.py|2024年NPB全12球団の投手成績（ERA・K/9・HR/9・球種割合）をスクレイピングしてCSVに保存するスクリプト|
|2025_initial_stats_scraper.py|2025年NPBの公式サイトから12球団の選手打撃成績をスクレイピングしてCSVに保存するスクリプト|
|at_bats.py|複数の試合JSONファイルから総打席数を集計して表示するスクリプト|
|calc_score_stats.py|game_data_2025 フォルダ内のJSONを用いて、2025年度NPBの全試合得点の標準偏差を求めるスクリプト|
|compare_models.py|打順の特徴量を平坦化し、重回帰とランダムフォレストでLSTMと得点予測精度を比較するベースライン評価スクリプト|
|game_score_lstm.py|打率・本塁打・長打率・OPSの4指標を動的に更新しながら標準LSTMで得点を予測し、結果をタイムスタンプ付きファイルに保存するスクリプト|
|scrape_nikkan_2025_all.py|日刊スコア速報から2025年度NPB公式戦全858試合（約6.4万打席）の打席履歴をスクレイピングしてCSVに保存するスクリプト|


### data
|ファイル/フィルダー名|概要|
|---|---|
|game_data_2025|2025年度打席履歴|
|url_list|日程ナビURLリストの保存先|
|initial_stats_2024.csv|2024年度個人打撃成績|
|initial_stats_2025.csv|2025年度個人打撃成績|
|pitcher_stats_2024.csv|2024年度個人投手成績|
|pitcher_stats_2024_all.csv|2024年度個人投手成績(球種割合含む)|
|result.txt|at_bats.pyの実行結果の保存先|



## 野球用語
### 使用する用語一覧
|用語|それ|
|---|---|
|AVG|打率|
|OBP|出塁率|
|SLG|長打率|
|OPS|打撃の総合評価指標|

### 詳細
#### AVG
$$
AVG = \frac{H}{AB}
$$

$$
打率 = \frac{安打}{打数}
$$

#### OBP
$$
OBP = \frac{H + BB + HBP}{AB + BB + HBP + SF}
$$

$$
OBP = \frac{安打 + 四球 + 死球}{打数 + 四球 + 死球 + 犠飛}
$$

##### SLG
$$
SLG = \frac{TB}{AB}
$$

$$
SLG = \frac{塁打}{打数}
$$

##### OPS
$$
OPS = \frac{OBP}{SLG}
$$

$$
打撃の総合評価指標 = \frac{出塁率}{長打率}
$$ 

## 対象データ
### ソース元
#### 2024年度個人投手成績
```txt=
https://npb.jp/bis/2024/stats/idp1_g.html
https://npb.jp/bis/2024/stats/idp1_t.html
https://npb.jp/bis/2024/stats/idp1_db.html
https://npb.jp/bis/2024/stats/idp1_c.html
https://npb.jp/bis/2024/stats/idp1_s.html
https://npb.jp/bis/2024/stats/idp1_d.html
https://npb.jp/bis/2024/stats/idp1_h.html
https://npb.jp/bis/2024/stats/idp1_f.html
https://npb.jp/bis/2024/stats/idp1_m.html
https://npb.jp/bis/2024/stats/idp1_e.html
https://npb.jp/bis/2024/stats/idp1_b.html
https://npb.jp/bis/2024/stats/idp1_l.html
```
```txt=
https://nf3.sakura.ne.jp/2024/Central/G/p/18_stat.htm
https://nf3.sakura.ne.jp/2024/Central/T/p/17_stat.htm
https://nf3.sakura.ne.jp/2024/Central/DB/p/17_stat.htm
https://nf3.sakura.ne.jp/2024/Central/C/p/17_stat.htm
https://nf3.sakura.ne.jp/2024/Central/S/p/17_stat.htm
https://nf3.sakura.ne.jp/2024/Central/D/p/17_stat.htm
https://nf3.sakura.ne.jp/2024/Pacific/B/p/13_stat.htm
https://nf3.sakura.ne.jp/2024/Pacific/M/p/14_stat.htm
https://nf3.sakura.ne.jp/2024/Pacific/H/p/14_stat.htm
https://nf3.sakura.ne.jp/2024/Pacific/E/p/14_stat.htm
https://nf3.sakura.ne.jp/2024/Pacific/L/p/14_stat.htm
https://nf3.sakura.ne.jp/2024/Pacific/F/p/14_stat.htm

```
#### 2024年度個人打撃成績
```txt=
https://npb.jp/bis/2024/stats/idb1_g.html
https://npb.jp/bis/2024/stats/idb1_t.html
https://npb.jp/bis/2024/stats/idb1_db.html
https://npb.jp/bis/2024/stats/idb1_c.html
https://npb.jp/bis/2024/stats/idb1_s.html
https://npb.jp/bis/2024/stats/idb1_d.html
https://npb.jp/bis/2024/stats/idb1_h.html
https://npb.jp/bis/2024/stats/idb1_f.html
https://npb.jp/bis/2024/stats/idb1_m.html
https://npb.jp/bis/2024/stats/idb1_e.html
https://npb.jp/bis/2024/stats/idb1_b.html
https://npb.jp/bis/2024/stats/idb1_l.html
```
#### 2025年度個人打撃成績
```txt
https://npb.jp/bis/2025/stats/idb1_g.html
https://npb.jp/bis/2025/stats/idb1_t.html
https://npb.jp/bis/2025/stats/idb1_db.html
https://npb.jp/bis/2025/stats/idb1_c.html
https://npb.jp/bis/2025/stats/idb1_s.html
https://npb.jp/bis/2025/stats/idb1_d.html
https://npb.jp/bis/2025/stats/idb1_h.html
https://npb.jp/bis/2025/stats/idb1_f.html
https://npb.jp/bis/2025/stats/idb1_m.html
https://npb.jp/bis/2025/stats/idb1_e.html
https://npb.jp/bis/2025/stats/idb1_b.html
https://npb.jp/bis/2025/stats/idb1_l.html
```

#### 2025年度打撃履歴
```txt
https://www.nikkansports.com/baseball/professional/schedule/cl03.html
https://www.nikkansports.com/baseball/professional/schedule/cl04.html
https://www.nikkansports.com/baseball/professional/schedule/cl05.html
https://www.nikkansports.com/baseball/professional/schedule/cl06.html
https://www.nikkansports.com/baseball/professional/schedule/cl07.html
https://www.nikkansports.com/baseball/professional/schedule/cl08.html
https://www.nikkansports.com/baseball/professional/schedule/cl09.html
https://www.nikkansports.com/baseball/professional/schedule/cl10.html
https://www.nikkansports.com/baseball/professional/schedule/pl03.html
https://www.nikkansports.com/baseball/professional/schedule/pl04.html
https://www.nikkansports.com/baseball/professional/schedule/pl05.html
https://www.nikkansports.com/baseball/professional/schedule/pl06.html
https://www.nikkansports.com/baseball/professional/schedule/pl07.html
https://www.nikkansports.com/baseball/professional/schedule/pl08.html
https://www.nikkansports.com/baseball/professional/schedule/pl09.html
https://www.nikkansports.com/baseball/professional/schedule/pl10.html
```

### データ形式
- CSV：個人打撃成績(1ファイル=1年間分)
- JSON：打撃履歴(1ファイル＝1試合分)

### 主な項目
#### game_data_2025/
|データ項目|用語|
|---|---|
|url|試合詳細ページのURL|
|scoreboard.team|チーム名|
|scoreboard.R|得点|
|text_live.inning|イニング（例：1回表、5回裏、試合前）|
|pregame.lineups.team|チーム名（スタメン情報）|
|pregame.lineups.players|スタメン選手名（打順順）|
|plays.lines[0]|打順＋選手名（例：1番 西川）|
|plays.lines[1]|打席結果（例：右安、三振、四球など）|

> ※ linesは文字列のため、打順・選手名・結果はパース処理が必要

#### initial_stats_2024.csv/initial_stats_2025.csv
|データ項目|用語|
|---|---|
|team|チーム名|
|name|選手名|
|hand|聞き手|
|AB|打数|
|H|安打|
|2B|2塁打|
|3B|3塁打|
|HR|本塁打|
|TB|塁打|
|BB|四球|
|HBP|死球|
|SF|犠飛|

#### pitcher_stats_2024.csv
|データ項目|用語|
|---|---|
|team|チーム名|
|name|選手名|
|G|登板数|
|IP|投球回|
|H|被安打|
|HR|被本塁打|
|SO|奪三振|
|ERA|防御率|

### pitcher_stats_2024_all.csv


|データ項目|用語|
|---|---|
|team|チーム名|
|name|選手名|
|hand|聞き手|
|K/9|奪三振率|
|HR/9|被本塁打率|
|pitch_〇〇_share|各球種割合(20種類)|











