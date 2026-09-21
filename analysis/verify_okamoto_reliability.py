import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic'

# 1. モデルの定義
class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49):
        super(FactorizationMachineModel, self).__init__()
        self.player_v = nn.Embedding(num_players, k)

def load_model_compatibility(target_batter="岡本和真"):
    """モデルから対象打者と全投手の相性スコア（内積）を計算する関数"""
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    
    # 表記揺れ（スペースの有無）に対応してIDを取得
    batter_id = name_to_id.get(target_batter)
    if batter_id is None:
        batter_id = name_to_id.get("岡本 和真")
    
    if batter_id is None:
        print(f"Error: モデルのマスターデータに {target_batter} が見つかりません。")
        return None, None

    # モデルのロード
    num_players = master["total_players"]
    model = FactorizationMachineModel(num_players=num_players)
    model.load_state_dict(torch.torch.load("fm_model.pth"), strict=False)
    embeddings = model.player_v.weight.detach().numpy()
    
    v_batter = embeddings[batter_id]
    
    # 全投手のスコア（内積）を計算して辞書に格納
    compatibility_dict = {}
    for p_name, p_id in name_to_id.items():
        if p_id == 0 or p_name in ["Unknown", "LEAGUE_AVERAGE"]: continue
        v_pitcher = embeddings[p_id]
        score = np.dot(v_batter, v_pitcher) # 内積を相性スコアとする
        
        # 照合用に名前の空白を除去
        p_name_clean = p_name.replace(" ", "").replace("　", "")
        compatibility_dict[p_name_clean] = score
        
    return compatibility_dict, master

def verify_data(file_path, label, min_pa, compatibility_dict, target_metric='打率'):
    """CSVデータとモデルスコアをマージして相関を分析・プロットする関数"""
    # 文字コード揺れに対応して読み込み
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp932')
        
    # 名前のクリーニング
    df['投手_clean'] = df['投手'].str.replace(" ", "").str.replace("　", "")
    
    # 【重要】対戦打席数（ノイズ）のフィルタリング
    df_filtered = df[df['打席'] >= min_pa].copy()
    
    # モデルのスコアをマージ
    df_filtered['Model_Score'] = df_filtered['投手_clean'].map(compatibility_dict)
    
    # マッチしなかった投手（モデル側かCSV側にしかいない選手）を除外
    df_filtered = df_filtered.dropna(subset=['Model_Score', target_metric])
    
    if len(df_filtered) < 3:
        print(f"[{label}] フィルタリング後のデータ数が足りません（現在: {len(df_filtered)}件）。min_paを下げてください。")
        return
    
    # 相関係数の算出 (ピアソンの積率相関係数)
    r_val, p_val = pearsonr(df_filtered['Model_Score'], df_filtered[target_metric])
    print(f"■ 【{label}】分析結果 (対象投手: {len(df_filtered)}名 / 最小打席フィルター: {min_pa}打席)")
    print(f"  -> モデルスコア と 実際の{target_metric} の相関係数 R = {r_val:.3f} (p値: {p_val:.4f})")
    
    # 散布図のプロット
    plt.figure(figsize=(8, 6))
    sns.regplot(data=df_filtered, x='Model_Score', y=target_metric, 
                scatter_kws={'s': 60, 'alpha': 0.7, 'color': 'crimson'},
                line_kws={'color': 'navy', 'linestyle': '--'})
    
    plt.title(f"岡本和真 妥当性検証 [{label}]\n（R = {r_val:.3f}, フィルター: {min_pa}打席以上）", fontsize=12)
    plt.xlabel("モデルの相性スコア (予測値)", fontsize=10)
    plt.ylabel(f"実際の{target_metric} (実績値)", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"verification_{label}_{target_metric}.png"
    plt.savefig(filename)
    plt.close()
    print(f"  -> 散布図を保存しました: {filename}\n")

if __name__ == "__main__":
    # 1. モデルからスコアを抽出
    comp_dict, master_data = load_model_compatibility(target_batter="岡本和真")
    
    if comp_dict:
        # ハルタさんが用意してくれた3つのファイルパス
        """
        files = {
            "通算": ("KAZUMA_OKAMOTO_Statistics_by_Opposing_Pitcher_Total.csv", 10),  # 通算はデータが多いので10打席以上
            "2024年度": ("KAZUMA_OKAMOTO_Statistics_by_Opposing_Pitcher_2024.csv", 5), # 単年は5打席以上
            "2025年度": ("KAZUMA_OKAMOTO_Statistics_by_Opposing_Pitcher_2025.csv", 5)  # 未来予測も5打席以上
        }
        """
        files = {
            "通算_15打席以上": ("KAZUMA_OKAMOTO_Statistics_by_Opposing_Pitcher_Total.csv", 15),
            "通算_20打席以上": ("KAZUMA_OKAMOTO_Statistics_by_Opposing_Pitcher_Total.csv", 20),
            "通算_30打席以上": ("KAZUMA_OKAMOTO_Statistics_by_Opposing_Pitcher_Total.csv", 30)
        }
        
        # 2. 『打率』での検証を実行
        print("=== 【検証パターン1】モデルスコア vs 実際の打率 ===")
        for label, (path, min_pa) in files.items():
            verify_data(path, label, min_pa, comp_dict, target_metric='打率')
            
        # 3. 『OPS』での検証を実行（セイバーメトリクス的により精密な答え合わせ）
        print("=== 【検証パターン2】モデルスコア vs 実際のOPS ===")
        for label, (path, min_pa) in files.items():
            verify_data(path, label, min_pa, comp_dict, target_metric='OPS')