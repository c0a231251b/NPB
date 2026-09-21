import os
import json
import glob
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# ============================================================
# 1. 連続値（得点差）対応型 Factorization Machines モデルの定義
# ============================================================
class RegressionFM(nn.Module):
    def __init__(self, num_features, k_dimensions=16):
        super(RegressionFM, self).__init__()
        # バイアス (w0)
        self.w0 = nn.Parameter(torch.zeros(1))
        # 1次結合の重み (w_i)
        self.w = nn.Linear(num_features, 1)
        # 2次相互作用の潜在ベクトル (V_i,k)
        self.V = nn.Parameter(torch.randn(num_features, k_dimensions) * 0.01)

    def forward(self, x):
        # 1次線形結合: w0 + sum(w_i * x_i)
        linear_terms = self.w0 + self.w(x)
        
        # 2次相互作用 (Interaction Terms) の数理高速計算
        # 1/2 * sum( (sum(v_i,k * x_i))^2 - sum(v_i,k^2 * x_i^2) )
        sum_vx = torch.mm(x, self.V)  # (batch_size, k)
        sum_vx_square = sum_vx ** 2
        
        squared_x = x ** 2
        squared_V = self.V ** 2
        sum_v_square_x_square = torch.mm(squared_x, squared_V)
        
        interaction_terms = 0.5 * torch.sum(sum_vx_square - sum_v_square_x_square, dim=1, keepdim=True)
        
        # 連続値（得点差）を予測するため、シグモイドはかけずにそのまま出力
        return linear_terms + interaction_terms

# ============================================================
# 2. 特徴量コンテキストベクトルのデータローダー
# ============================================================
def load_dataset_from_dir(json_dir):
    """各試合のJSONから対比特徴量（contrast_feature）と得点差（target）をマトリックス化"""
    file_paths = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    
    X_list = []
    y_list = []
    
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # 目的変数：得点差 (target) の計算
        scoreboard = data.get("scoreboard", [])
        if len(scoreboard) < 2: continue
        
        # 自球団（ホーム）の総得点 - 相手球団（ビジター）の総得点
        # 5.1節の定義通り、グラデーションを保持
        try:
            r_away = int(scoreboard[0]["R"])
            r_home = int(scoreboard[1]["R"])
            target_score_diff = float(r_home - r_away)
        except:
            continue
            
        # 1試合の全打席コンテキスト特徴量を配列化
        # ハルタさんが前処理で追加してくれた 'at_bat_features' を安全に展開
        features = data.get("at_bat_features", [])
        if not features: continue
        
        game_contrast_vector = []
        for feat in features:
            # 5.節の数式：打者打率 - 投手被打率 の差分値
            # もし前処理スクリプトのキー名が異なる場合は、対応するキー名（'advantage_diff'等）に微調整してください
            c_val = feat.get("contrast_feature", 0.0)
            if c_val is None: c_val = 0.0
            game_contrast_vector.append(c_val)
            
        # ニューラルネットワークの入力次元を一律に固定するため、最大打席数（例: 90打席）でパディング
        # 試合ごとの打席数のブレを 0.0（互角コンテキスト）で埋める防衛策
        MAX_BATTER_BOXES = 90
        if len(game_contrast_vector) < MAX_BATTER_BOXES:
            game_contrast_vector += [0.0] * (MAX_BATTER_BOXES - len(game_contrast_vector))
        else:
            game_contrast_vector = game_contrast_vector[:MAX_BATTER_BOXES]
            
        X_list.append(game_contrast_vector)
        y_list.append([target_score_diff])
        
    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32)

# ============================================================
# 3. 実験ランナー（トレーニングループ）
# ============================================================
def train_and_evaluate(json_dir, label_name):
    X, y = load_dataset_from_dir(json_dir)
    if len(X) == 0:
        print(f"❌ {label_name} のデータロードに失敗しました。パスを確認してください。")
        return
        
    # 時系列を崩さないよう、シーズン終盤の20%を検証（Val）データにスプリット
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    model = RegressionFM(num_features=X.shape[1], k_dimensions=16)
    criterion = nn.MSELoss() # 得点差回帰タスクのため、MSEを採用
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.01)
    
    best_val_mse = float('inf')
    
    for epoch in range(1, 51):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
        # 検証精度計測
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_mse = criterion(val_pred, y_val).item()
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                
    # 評価指標の可視化変換
    rmse = np.sqrt(best_val_mse)
    print(f"■ {label_name:<15} -> 最適最小 Val MSE: {best_val_mse:.4f} | 予測誤差 RMSE: {rmse:.3f} 点")
    return best_val_mse

# ============================================================
# 4. エントリポイント（3つの時間窓の一斉比較）
# ============================================================
if __name__ == "__main__":
    print("==================================================")
    print("📊 投手・野手対比特徴量 時間窓（Window）別 精度比較実証実験")
    print("==================================================")
    
    # フォルダパスの定義（ハルタさんのローカル環境のフォルダ名と一致させています）
    dirs = {
        #"パターン1 (前年度スタッツ)": "game_data_2025_updated_hoge",
        "パターン2 (直近30日)": "game_data_2025_match_results_add_30day_stats",
        "パターン3 (直近7日)" : "game_data_2025_match_results_add_7day_stats"
    }
    
    for label, path in dirs.items():
        if os.path.exists(path):
            train_and_evaluate(path, label)
        else:
            print(f"⚠️ パスが見つかりません: {path}")
    print("==================================================")