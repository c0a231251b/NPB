import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # ← ここにまとめる

class FactorizationMachineModel(nn.Module):
    def __init__(self, num_players, k=16, num_num_features=4): # 4次元に修正
        super(FactorizationMachineModel, self).__init__()
        self.k = k
        self.player_linear = nn.Embedding(num_players, 1)
        self.player_v = nn.Embedding(num_players, self.k)

        # 【追加】重みを小さく初期化する（爆発防止）
        nn.init.normal_(self.player_linear.weight, std=0.01)
        nn.init.normal_(self.player_v.weight, std=0.01)

        self.num_linear = nn.Linear(num_num_features, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_cat, x_num):
        linear_part = self.num_linear(x_num) + torch.sum(self.player_linear(x_cat), dim=1)
        v = self.player_v(x_cat)
        sum_of_v = torch.sum(v, dim=1)
        square_of_sum = sum_of_v ** 2
        sum_of_square = torch.sum(v ** 2, dim=1)
        interaction_part = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return linear_part + interaction_part + self.bias

def train_fm():
    df = pd.read_pickle("fm_dataset_7d_30d.pkl")
    X_cat = np.stack(df["cat_features"].values)
    X_num = np.stack(df["num_features"].values)
    y = df["target"].values.astype(np.float32)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X_num)
    joblib.dump(scaler, "fm_scaler_7d_30d.gz")

    X_cat_train, X_cat_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
        X_cat, X_num, y, test_size=0.2, random_state=42
    )

    train_data = torch.utils.data.TensorDataset(
        torch.LongTensor(X_cat_train), torch.FloatTensor(X_num_train), torch.FloatTensor(y_train)
    )
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

    # 総選手数はマスターの数字(697)に合わせる
    model = FactorizationMachineModel(num_players=697, k=16, num_num_features=4)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.1)

    print("--- FMモデル学習開始 ---")
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

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(torch.LongTensor(X_cat_val), torch.FloatTensor(X_num_val)).squeeze()
                val_loss = criterion(val_output, torch.FloatTensor(y_val))
                print(f"Epoch {epoch:3}: Train MSE={total_loss/len(train_loader):.4f}, Val MSE={val_loss.item():.4f}")

    torch.save(model.state_dict(), "fm_model_7d_30d.pth")
    print("学習完了。 fm_model_7d_30d.pth を保存しました。")

   #============================
   #学習後の評価指標計算と表示
   #============================
    model.eval()

    with torch.no_grad():

        pred = model(
            torch.LongTensor(X_cat_val),
            torch.FloatTensor(X_num_val)
        ).squeeze().numpy()

    mae = mean_absolute_error(y_val, pred)
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    r2 = r2_score(y_val, pred)

    print()
    print("===== 評価結果 =====")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")

    #============================
    # ベースライン（平均値予測）との比較
    #============================

    pred = np.repeat(np.mean(y_train), len(y_val))

    mae = mean_absolute_error(y_val, pred)
    print()
    print("===== ベースラインとの比較 =====")
    print(f"ベースライン MAE = {mae:.4f}")

    print()
    print("====予測値の分布確認====")
    print(f"val_output[:20] = {val_output[:20]}")  # 最初の20件を表示
    print(f"y_val[:20] = {y_val[:20]}")      # 対応する実際の値も表示


if __name__ == "__main__":
    train_fm()