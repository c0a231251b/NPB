import torch
import torch.nn as nn
import json
import pandas as pd
import numpy as np

# 1. モデルの定義（学習済みパラメータを読み込むための器）
class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49):
        super(FactorizationMachineModel, self).__init__()
        # 線形項（選手個人の基本能力 w）
        self.player_w = nn.Embedding(num_players, 1)
        # 相互作用項（相性ベクトル v）
        self.player_v = nn.Embedding(num_players, k)

def extract_w_ranking():
    # --- 1. マスターデータの読み込み ---
    with open("player_id_master.json", "r", encoding="utf-8") as f:
        master = json.load(f)
    name_to_id = master["name_to_id"]
    id_to_name = {v: k for k, v in name_to_id.items()}
    num_players = master["total_players"]

    # --- 2. モデルのロードと重み（w）の抽出 ---
    model = FactorizationMachineModel(num_players=num_players)
    # strict=Falseにすることで、他の特徴量の重みがあっても player_w を確実に復元します
    model.load_state_dict(torch.load("fm_model.pth"), strict=False)
    
    # Embeddingから重みを取り出してNumPy配列に変換 (形状: [num_players, 1])
    weights_w = model.player_w.weight.detach().numpy().flatten()

    # --- 3. 2024年のスタッツから投手・野手の判定用リストを作成 ---
    # 手元のスタッツCSVから選手名（スペース除去）を収集
    try:
        h_df = pd.read_csv("initial_stats_2024.csv")
        p_df = pd.read_csv("pitcher_stats_2024_all.csv")
        csv_batters = set(str(name).replace(" ", "").replace(" ", "") for name in h_df['name'].unique())
        csv_pitchers = set(str(name).replace(" ", "").replace(" ", "") for name in p_df['name'].unique())
    except FileNotFoundError:
        print("Warning: スタッツCSVが見つかりません。全選手一括のランキングを作成します。")
        csv_batters = set()
        csv_pitchers = set()

    # --- 4. データの整理 ---
    all_data = []
    for p_name, p_id in name_to_id.items():
        if p_id == 0 or p_name in ["Unknown", "LEAGUE_AVERAGE"]: 
            continue
            
        w_val = weights_w[p_id]
        p_name_clean = p_name.replace(" ", "").replace(" ", "")
        
        # 役割の判定
        if p_name_clean in csv_pitchers:
            role = "投手"
        elif p_name_clean in csv_batters:
            role = "打者"
        else:
            role = "不明"
            
        all_data.append({
            "Player_ID": p_id,
            "Player_Name": p_name,
            "Weight_W": w_val,
            "Role": role
        })
        
    df_res = pd.DataFrame(all_data)

    # --- 5. ランキングの抽出と表示 ---
    # FMの式において、Y（得点）に貢献するほど w は大きくなります。
    # したがって：打者は「値が大きいほど優秀（基本打力が高い）」
    #      ：投手は「値が小さい（マイナスに大きい）ほど優秀（失点を抑える基本投球力が高い）」
    
    print("\n" + "="*50)
    print(" 【打者セクション】基本能力値（W） TOP 20")
    print(" ※ 値が大きいほど、平均的な得点貢献度（基本打力）が高い")
    print("="*50)
    df_batters = df_res[df_res['Role'] == "打者"].sort_values(by="Weight_W", ascending=False).head(20)
    for idx, row in df_batters.reset_index(drop=True).iterrows():
        print(f"{idx+1:2d}位: {row['Player_Name']:<12} (ID: {row['Player_ID']:3d}) -> W = {row['Weight_W']:.4f}")

    print("\n" + "="*50)
    print(" 【投手セクション】基本能力値（W） TOP 20")
    print(" ※ 値が小さい（マイナス）ほど、失点を防ぐ能力（基本投球力）が高い")
    print("="*50)
    df_pitchers = df_res[df_res['Role'] == "投手"].sort_values(by="Weight_W", ascending=True).head(20)
    for idx, row in df_pitchers.reset_index(drop=True).iterrows():
        print(f"{idx+1:2d}位: {row['Player_Name']:<12} (ID: {row['Player_ID']:3d}) -> W = {row['Weight_W']:.4f}")

    # 次のクラスタリング作業で使い回せるように、全選手のW一覧をCSVとして保存
    df_res.to_csv("extracted_linear_weights.csv", index=False, encoding="utf-8-sig")
    print("\n" + "-"*50)
    print("ベースデータを保存しました: extracted_linear_weights.csv")
    print("-"*50)

if __name__ == "__main__":
    extract_w_ranking()