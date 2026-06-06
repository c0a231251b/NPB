import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==========================================
# 変更点: ファイル名および保存パスの一新
# スクリプト名: train_fm_model_score_diff.py
# ==========================================

class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=49): 
        super(FactorizationMachineModel, self).__init__()
        self.k = k
        self.player_linear = nn.Embedding(num_players, 1)
        self.player_v = nn.Embedding(num_players, self.k)

        # 重みを小さく初期化（勾配爆発防止）
        nn.init.normal_(self.player_linear.weight, std=0.01)
        nn.init.normal_(self.player_v.weight, std=0.01)

        self.num_linear = nn.Linear(num_num_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat, x_num):
        # 線形項の計算
        linear_part = self.num_linear(x_num) + torch.sum(self.player_linear(x_cat), dim=1)
        
        # 2次交互作用項（FMの核心部分）の計算
        v = self.player_v(x_cat)
        sum_of_v = torch.sum(v, dim=1)
        square_of_sum = sum_of_v ** 2
        sum_of_square = torch.sum(v ** 2, dim=1)
        interaction_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        # 回帰タスクのため、活性化関数（Sigmoid等）は通さず生の予測値を返す
        return linear_part + interaction_part + self.bias

def train_fm_score_diff():
    # 【確認】得点差をターゲットにしたデータセットのファイル名を指定してください
    # もし既存の 'fm_dataset.pkl' の 'target' カラムがすでに得点差に書き換わっている場合は、そのままで大丈夫です。
    dataset_file = "fm_dataset_score_diff.pkl" 
    
    try:
        df = pd.read_pickle(dataset_file)
        print(f"データセット '{dataset_file}' を読み込みました。")
    except FileNotFoundError:
        # バックアップとして従来のファイル名でも読み込めるように処理
        dataset_file = "fm_dataset.pkl"
        df = pd.read_pickle(dataset_file)
        print(f"警告: 新ファイルが見つからないため、既存の '{dataset_file}' を読み込みました。'target'が「得点差」になっているか確認してください。")

    X_cat = np.stack(df["cat_features"].values)
    X_num = np.stack(df["num_features"].values)
    
    # 目的変数 y (targetカラムに「自得点 - 相手得点」の連続値が入っている前提)
    y = df["target"].values.astype(np.float32)

    # 数値特徴量の標準化
    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    joblib.dump(scaler, "fm_scaler_score_diff.gz") # スケーラーの保存名も変更

    # 訓練データと検証データの分割
    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_cat, X_num, y, test_size=0.2, random_state=42
    )

    train_data = torch.utils.data.TensorDataset(
        torch.LongTensor(X_cat_train), torch.FloatTensor(X_num_train), torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # モデル・損失関数・最適化手法の定義
    model = FactorizationMachineModel(num_players=697, k=16, num_num_features=49)
    criterion = nn.MSELoss() # 連続値の誤差を測る平均二乗誤差
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1)

    print("---  得点差予測 FMモデル学習開始 ---")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0
        for cat, num, target in train_loader:
            optimizer.zero_grad()
            output = model(cat, num).squeeze()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 10エポックごとに進捗と検証データでのLoss（MSE）を表示
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(torch.LongTensor(X_cat_val), torch.FloatTensor(X_num_val)).squeeze()
                val_loss = criterion(val_output, torch.FloatTensor(y_val))
                print(f"Epoch {epoch:3}: Train MSE={total_loss/len(train_loader):.4f}, Val MSE={val_loss.item():.4f}")

    # 【変更】得点差モデル専用のパスで重みを保存
    output_model_path = "fm_model_score_diff.pth"
    torch.save(model.state_dict(), output_model_path)
    print(f"学習完了。 '{output_model_path}' を保存しました。")

if __name__ == "__main__":
    train_fm_score_diff()