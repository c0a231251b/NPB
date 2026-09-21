
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack

# ============================================================
# 1. データ読み込み
# ============================================================
df = pd.read_csv("game_features_2025.csv")

# ============================================================
# 2. 特徴量分類
# ============================================================
one_hot_cols = [
    "season_id",
    "team_id",
    "opponent_team_id",
    "stadium_id",
    "home_away"
]

cat_cols = ["starter_hand"]

num_cols = [
    "team_avg_runs_7d",
    "opp_avg_runs_7d",
    "opp_starter_era"
]

target_col = "target_runs"

# ============================================================
# 3. one-hot + 数値の疎行列化
# ============================================================
# ============================================================
# 3. one-hot + 数値の疎行列化
# ============================================================
enc = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
X_cat = enc.fit_transform(df[one_hot_cols + cat_cols])

X_num = csr_matrix(df[num_cols].values)

X = hstack([X_cat, X_num]).astype(np.float32)
y = df[target_col].values.astype(np.float32)


# ============================================================
# 4. train/test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 5. FM（2次相互作用） NumPy 実装
# ============================================================

class FM:
    def __init__(self, n_features, k=8, lr=0.01, reg=0.01):
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(scale=0.1, size=(n_features, k))
        self.lr = lr
        self.reg = reg
        self.k = k

    def predict_row(self, x):
        # x: sparse row
        x = x.toarray().ravel()
        linear = self.w0 + np.dot(self.w, x)
        interaction = 0.5 * np.sum(
            (x @ self.V)**2 - (x**2) @ (self.V**2)
        )
        return linear + interaction

    def fit(self, X, y, epochs=10):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                x = X[i]
                pred = self.predict_row(x)
                err = y[i] - pred

                # update w0
                self.w0 += self.lr * err

                # update w
                x_arr = x.toarray().ravel()
                self.w += self.lr * (err * x_arr - self.reg * self.w)

                # update V
                for f in range(self.k):
                    v_f = self.V[:, f]
                    grad_v = err * (x_arr * (x_arr @ v_f) - v_f * (x_arr**2))
                    self.V[:, f] += self.lr * (grad_v - self.reg * v_f)

            print(f"Epoch {epoch+1} finished")

    def predict(self, X):
        return np.array([self.predict_row(X[i]) for i in range(X.shape[0])])


# ============================================================
# 6. FM 学習
# ============================================================
fm = FM(n_features=X.shape[1], k=8, lr=0.001, reg=0.01)
fm.fit(X_train, y_train, epochs=5)

# ============================================================
# 7. 評価
# ============================================================
y_pred = fm.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("====================================")
print("FM (NumPy implementation) Result")
print("====================================")
print(f"RMSE : {rmse:.4f}")
print("====================================")
