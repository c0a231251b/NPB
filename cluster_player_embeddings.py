import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic'

# 1. モデルの定義（パラメータ読み込み用）
class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49):
        super(FactorizationMachineModel, self).__init__()
        self.player_w = nn.Embedding(num_players, 1)
        self.player_v = nn.Embedding(num_players, k)

def perform_clustering():
    # --- 1. マスターデータとモデルのロード ---
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}
    num_players = master["total_players"]

    model = FactorizationMachineModel(num_players=num_players)
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    
    # 16次元の潜在ベクトル（V）を抽出
    embeddings_v = model.player_v.weight.detach().numpy()
    
    # --- 【重要】投手ノイズを排除するためのフィルターデータの読み込み ---
    try:
        p_df = pd.read_csv("pitcher_stats_2024_all.csv")
        csv_pitchers = set(str(name).replace(" ", "").replace(" ", "") for name in p_df['name'].unique())
    except FileNotFoundError:
        print("Error: 'pitcher_stats_2024_all.csv' が見つかりません。投手フィルターを適用できません。")
        return

    # 0番インデックスや特殊なラベル、および「投手」を除外した【打者限定】のリストを作成
    valid_ids = []
    valid_embeddings = []
    
    for p_name, p_id in name_to_id.items():
        if p_id == 0 or p_name in ["Unknown", "LEAGUE_AVERAGE"]: 
            continue
            
        # 空白を除去して投手CSVとマッチング
        p_name_clean = p_name.replace(" ", "").replace(" ", "")
        if p_name_clean in csv_pitchers:
            continue # 投手は完全にスキップ
            
        valid_ids.append(p_id)
        valid_embeddings.append(embeddings_v[p_id])
        
    X = np.array(valid_embeddings) # クラスタリングに使う行列 [有効な打者数, 16]
    print(f"投手を除外しました。分析対象の総打者数: {X.shape[0]} 名")

    # --- 2. 各指標のループ計算を分離 ---
    # ① エルボー法 (K=1 〜 10)
    k_range_elbow = range(1, 11)
    inertias = []
    for k in k_range_elbow:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
    # ② シルエットスコア (K=2 〜 10)
    k_range_silhouette = range(2, 11)
    silhouettes = []
    for k in k_range_silhouette:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouettes.append(score)

    # --- 3. 左右に並べた2画面グラフの描画と保存 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左側：エルボー法（K=1からスタート）
    ax1.plot(k_range_elbow, inertias, marker='o', color='navy', linestyle='--')
    ax1.set_title('エルボー法による最適クラスター数の探索 (打者限定)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('クラスター数 (K)', fontsize=10)
    ax1.set_ylabel('クラスター内平方回 (Inertia)', fontsize=10)
    ax1.set_xticks(list(k_range_elbow))
    ax1.grid(True, alpha=0.3)

    # 右側：シルエットスコア（K=2からスタート）
    ax2.plot(k_range_silhouette, silhouettes, marker='s', color='crimson', linestyle='-')
    ax2.set_title('シルエットスコアによる品質評価 (打者限定)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('クラスター数 (K)', fontsize=10)
    ax2.set_ylabel('シルエット係数 (Silhouette Score)', fontsize=10)
    ax2.set_xticks(list(k_range_silhouette))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("clustering_metrics_evaluation.png")
    plt.close()
    print("投手を除外した、打者限定の左右併記グラフを保存しました: clustering_metrics_evaluation.png")

    # --- 4. 【暫定】数理の正解である K=3 で一度クラスタリングを実行し保存 ---
    """
    注意！！！！！！！
    ここを変更
    """
    chosen_k = 4  # ここをエルボー法とシルエットスコアの結果を見て適切に選択してください
    print(f"\n純粋な打者空間を K={chosen_k} でグルーピングします...")
    final_kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X)

    # 前回の線形項（W）の結果CSVがあればマージ（投手も含んだ全リストに対して打者のみ更新）
    try:
        df_weights = pd.read_csv("extracted_linear_weights.csv")
        id_to_cluster = {valid_ids[i]: cluster_labels[i] for i in range(len(valid_ids))}
        # マップされない選手（投手など）は一目でわかるように NaN または -1 になるように処理
        df_weights['Cluster_ID'] = df_weights['Player_ID'].map(id_to_cluster)
        df_res = df_weights
    except FileNotFoundError:
        plot_data = []
        for i, p_id in enumerate(valid_ids):
            plot_data.append({
                'Player_ID': p_id,
                'Player_Name': id_to_name.get(p_id, "Unknown"),
                'Cluster_ID': cluster_labels[i]
            })
        df_res = pd.DataFrame(plot_data)

    df_res.to_csv(f"extracted_player_clusters_K{chosen_k}.csv", index=False, encoding="utf-8-sig")
    print(f"打者のみにクラスターIDを付与して保存しました: extracted_player_clusters_K{chosen_k}.csv")

    print(f"\n打者クラスターごとの所属選手数 (K={chosen_k}):")
    print(df_res['Cluster_ID'].value_counts().sort_index())

if __name__ == "__main__":
    perform_clustering()