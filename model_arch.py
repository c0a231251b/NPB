import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- 1. データの準備 ---
# 前ステップの train_data を Tensor に変換
X = np.array([sample["lineup_vectors"] for sample in train_data], dtype=np.float32) # [106, 9, 4]
y = np.array([sample["score"] for sample in train_data], dtype=np.float32).reshape(-1, 1) # [106, 1]

# DatasetとDataLoaderの作成
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# --- 2. モデルの定義 (論文の図1を再現) ---
class BattingOrderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BattingOrderRNN, self).__init__()
        # RNN層 (LSTMを採用) 
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # 全結合層 (DENSE) [cite: 48, 57]
        self.fc = nn.Linear(hidden_size, 1)
        # 活性化関数 (ReLU) [cite: 48, 55, 98]
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTMの出力のうち、最後のステップの隠れ状態(h_n)を使用
        _, (h_n, _) = self.lstm(x)
        # h_nの形状は [num_layers, batch, hidden_size] なので [batch, hidden_size] に変換
        out = self.fc(h_n[-1])
        # 最後にReLUを適用してスコアを回帰 [cite: 70, 71]
        out = self.relu(out)
        return out

# パラメータ設定
INPUT_SIZE = 4   # AVG, OBP, SLG, ISO
HIDDEN_SIZE = 32 # 隠れ層の次元数
model = BattingOrderRNN(INPUT_SIZE, HIDDEN_SIZE)

# 損失関数と最適化手法の設定
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001) # 

# --- 3. 学習ループ ---
epochs = 100
print("学習開始...")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss (MSE): {total_loss/len(loader):.4f}")

print("学習完了！")