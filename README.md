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
|2024_pitcher_stats_scraper.py|2024年NPBの公式サイトから12球団の選手投手成績をスクレイピングしてCSVに保存するスクリプト
|2025_initial_stats_scraper.py|2025年NPBの公式サイトから12球団の選手打撃成績をスクレイピングしてCSVに保存するスクリプト|
|at_bats.py|複数の試合JSONファイルから総打席数を集計して表示するスクリプト|
|calc_score_stats.py|game_data_2025 フォルダ内のJSONを用いて、2025年度NPBの全試合得点の標準偏差を求めるスクリプト|
|data_explorer.py|学習データの品質を診断する分析スクリプトで、得点分布・選手データのカバー率・特徴ベクトルのレンジを出力|
|fature_engineering.py|試合JSONから打席結果を累積しながら、スタメン打順の成績ベクトルと得点をセットにした学習データを構築するスクリプト|
|game_score_lstm.py|打率・本塁打・長打率・OPSの4指標を動的に更新しながら標準LSTMで得点を予測し、結果をタイムスタンプ付きファイルに保存するスクリプト|
|model_arch.py|スタメン打順の成績ベクトルを入力にLSTMで得点を予測する回帰モデルを定義・学習するコード|
|npb_bulk_scraper.py|Yahoo!野球のNPB試合テキスト中継ページをスクレイピングし、スコアボードと打席情報をJSON形式で保存するスクリプト(公式戦)|
|op_npb_bulk_scraper.py|Yahoo!野球のNPB試合テキスト中継ページをスクレイピングし、スコアボードと打席情報をJSON形式で保存するスクリプト(OP戦)|
|order_simulator.py|試合JSONからデータ構築・LSTM学習・打順最適化シミュレーションまでを一括実行する統合スクリプト|
|parser_utils.py|特定の1試合のYahoo!野球テキスト中継をスクレイピングしてJSONに保存する動作確認用スクリプト|
|scraper_sample.py|試合前情報のパース処理を省いた、スコアボードと打席テキストのみを取得する簡略版スクレイパー|
|stats_2024_train_model_attention.py|スタメン9人未満のデータを除外するバリデーションを追加した、LSTM+Attention打順予測モデルの改良版|
|stats_2024_train_model_attention_final.py|双方向LSTM＋Attentionモデル|
|stats_2025_train_model.py|2025年の実績CSVを初期値として球団別に選手を管理し、2026年試合データで累積更新しながらLSTMで打順の得点を予測・比較するシステム|
|stats_2025_train_model_attention.py|LSTMにAttention機構を追加し、打順ごとの重要度を可視化できるように拡張したモデル|
|text_pattern_analyzer.py|試合JSONから打席結果テキストを収集し、頻出表現をランキング表示する分析スクリプト|
|train_model.py|累積成績から打順ベクトルを生成してLSTMで得点を予測する、データ構築から学習までの一連のパイプライン|
|visualize_average_contribution.py|1試合あたりの平均的な貢献度を可視化|
|visualize_specific_game.py|特定の1試合の詳細な貢献度を可視化|
|visualize_total_contribution.py|全試合の合計得点貢献を可視化|

### data
|ファイル名|概要|
|---|---|
|game_data|2026年度打席履歴|
|game_data_2025|2025年度打席履歴|
|graph_data|グラフ保存用フォルダー|
|url_list|日程ナビURLリストの保存先|
|game_2021038670_text.json|4/5読売ジャイアンツvs横浜DeNAベイスターズ試合テキスト速報|
|initial_stats_2024.csv|2024年度個人打撃成績|
|initial_stats_2025.csv|2025年度個人打撃成績|
|pitcher_stats_2024.csv|2024年度個人投手成績|

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

#### 2026年度打撃履歴
```txt
https://baseball.yahoo.co.jp/npb/game/2021038622/text
```
**スポナビを利用しているが,日刊速報に変更の予定↓**

```txt
https://www.nikkansports.com/baseball/professional/score/2026/pf-score-20260327.html
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



## その他・あれこれ

### [研究ノート](https://hackmd.io/21dSIeucTJWG9MuyuxF3Eg)
### [コード詳細](https://hackmd.io/@Q6DZToC7RdeE3EFOW2XCAg/SkzxHR7hWe/edit)














