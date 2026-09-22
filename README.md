# NPB

## 概要

本リポジトリは、NPB（日本プロ野球）の試合・選手データを収集し、前処理・特徴量生成・モデル学習・評価を行うための実験環境です。

主に以下の処理を扱います。

- NPB選手成績・試合データのスクレイピング
- 打席履歴データの整形・選手IDの対応付け
- 直近成績・左右別成績などの特徴量生成
- LSTM、Factorization Machine（FM）、LightGBMによる予測実験
- 打席結果予測に向けた特徴量比較
- イニング単位の得点有無予測
- 選手埋め込みのクラスタリング・t-SNEによる可視化
- 打順評価・打順最適化に向けた分析



## リポジトリ構成

| ファイル名 | 内容 |
|---|---|
| `analysis/` | Lasso係数、特徴量、次元数などの分析用スクリプト・出力 |
| `checks/` | データ件数、欠損、カテゴリ数、pickle内容などを確認する補助スクリプト |
| `clustering/` | FMの埋め込みベクトルを用いた選手クラスタリング・相性分析 |
| `compare/` | LSTM、重回帰、Random Forestなど複数モデル・特徴量表現の比較 |
| `data/` | マスタデータおよび統合・加工済みデータ |
| `data/master/` | 選手ID、ロスター、試合情報などの参照用マスタデータ |
| `data/processed/` | 統合済みJSONや特徴量生成後のCSVなどの加工済みデータ |
| `evaluate/` | 打席結果予測に関するLightGBM・FM・SeqFM等の比較実験 |
| `evaluate/code/` | 打席結果予測の評価スクリプトおよび各実験結果 |
| `evaluate/data/` | 打席結果予測で使用する特徴量追加済みCSV |
| `evaluate_inning/` | イニング単位の得点有無・得点数予測に関する実験 |
| `evaluate_inning/code/` | イニング予測の評価スクリプトおよび各実験結果 |
| `fm/` | FM用データセット、スケーラー、学習・評価関連ファイル |
| `game_data_2025/` | 2025年度NPBの試合データ |
| `game_data_2025_match_results/` | 打席結果を含む2025年度NPBの試合データ |
| `game_data_2025_match_results_add_7day_stats/` | 直近7日間成績を追加した試合データ |
| `game_data_2025_match_results_add_30day_stats/` | 直近30日間成績を追加した試合データ |
| `game_data_2025_match_results_add_7day_stats_*/` | 7日間のBB、HR、ISO、K、OBP、OPS、SLG等を個別に追加した試合データ |
| `game_data_2025_match_results_add_30day_stats_*/` | 30日間のBB、HR、ISO、K、OBP、OPS、SLG等を個別に追加した試合データ |
| `game_data_2025_match_results_Just_before_the_turn_at_bat_stats/` | 各打席直前までの成績を追加した試合データ |
| `game_data_2025_updated/` | 更新処理を施した2025年度試合データ |
| `image/` | データ確認・検証用の画像 |
| `lightgbm/` | LightGBMの学習スクリプト |
| `lineup/` | スタメン打順の抽出・集計・打順最適化関連 |
| `logs/` | 名前対応失敗など、前処理・データ作成時のログ |
| `lstm/` | LSTMによる得点予測モデルの学習スクリプト・スケーラー |
| `prepare_fm_datasets/` | FM学習用データセット作成スクリプト |
| `prepare_fm_datasets_contrast9/` | contrast9条件でのFM学習用データセット作成スクリプト |
| `preprocessing/` | 選手ID作成、JSON結合、投打左右情報追加、時系列成績生成などの前処理 |
| `result/` | 過去の実験結果 |
| `scraping/` | NPB公式サイト・試合速報などからデータを取得するスクレイピングスクリプト |
| `stats/` | 打者・投手の基礎成績、左右別・対戦相手別成績 |
| `url_list/` | スクレイピング対象URLの一覧 |
| `visualize/` | t-SNE、クラスタリング結果、打順分布などの可視化 |
| `window_stats/` | 7日・30日・打席直前までの成績特徴量を付与するスクリプト |
| `.gitignore` | Git管理対象外ファイルの設定 |
| `README.md` | 本リポジトリの概要・構成・実行順序 |

## 実行順序

本リポジトリには過去の検証スクリプトも含まれているため、すべてのファイルを順番に実行する必要はありません。  
基本的な実験の流れは以下の通りです。

### 1. データ収集

NPB選手成績や試合データを取得します。

```powershell
python scraping/2025_initial_stats_scraper.py
python scraping/scrape_nikkan_2025_all.py
```

必要に応じて、`scraping/` 内の更新版・確認用スクリプトを使用します。

### 2. 前処理

取得したデータを統合し、選手IDや投打左右情報などを付与します。

```powershell
python preprocessing/create_player_ids.py
python preprocessing/merge_json_files.py
python preprocessing/add_handedness_to_game_json.py
python preprocessing/generate_time_series_stats.py
```

使用するデータや実験条件によって、必要な処理のみ実行します。

### 3. 時系列・直近成績特徴量の生成

7日間、30日間、打席直前までの成績などを追加します。

```powershell
python window_stats/add_window_7day_stats.py
python window_stats/add_window_30day_stats.py
python window_stats/add_window_Just_before_the_turn_at_bat_stats.py
```

BB、HR、ISO、K、OBP、OPS、SLGを個別に追加する場合は、`window_stats/` 内の対応するスクリプトを使用します。

### 4. FM用データセットの作成

FMを使用する場合は、試合データから学習用データセットを作成します。

```powershell
python prepare_fm_datasets/prepare_fm_dataset.py
```

7日・30日成績やcontrast9を使用する実験では、`prepare_fm_datasets/` または `prepare_fm_datasets_contrast9/` 内の対応するスクリプトを使用します。

### 5. モデル学習・比較

過去の得点予測実験では、以下のフォルダのスクリプトを使用します。

```powershell
python lstm/game_score_lstm.py
python lightgbm/train_lightgbm.py
```

FM、LSTM、重回帰、Random Forest等の比較は `fm/`、`lstm/`、`compare/` の各スクリプトを使用します。

### 6. 打席結果予測

打席結果予測の実験は `evaluate/` で行います。

例：

```powershell
python evaluate/code/evaluate_lightgbm_fm_37_features_4class.py --csv_path "evaluate/data/train_all_added_37_features.csv"
```

特徴量数、クラス定義、class weight、二段階分類、閾値調整などの条件ごとに評価スクリプトが分かれています。  
実験結果は各 `output_*` フォルダに保存されます。

### 7. イニング得点予測

イニング開始時点の情報から、得点有無・得点数を予測する実験は `evaluate_inning/` で行います。

代表的な検証スクリプト：

```powershell
python evaluate_inning/code/evaluate_step1_lightgbm_score_context.py
python evaluate_inning/code/evaluate_step2_lightgbm_lineup_strength.py
python evaluate_inning/code/evaluate_step2a_lightgbm_lineup_vs_pitcher_hand.py
python evaluate_inning/code/evaluate_step3_fm_binned_cross_step2a.py
python evaluate_inning/code/evaluate_step4_threshold_tuning_step2a.py
```

実験結果は各 `output_*` フォルダに保存されます。

### 8. 分析・可視化

モデル学習後の選手埋め込み、クラスタ、打順との関係などを分析・可視化します。

```powershell
python clustering/cluster_player_embeddings.py
python visualize/t-SNE_Visualization.py
```

用途に応じて `analysis/`、`clustering/`、`visualize/` 内のスクリプトを実行します。

> **注意**  
> 各スクリプトには、入力ファイルのパスや出力先がコード内で指定されているものがあります。  
> 実行前に、使用するデータファイルとパス設定を確認してください。




