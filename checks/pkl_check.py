import pickle
df = pickle.load(open("fm_dataset_7d_OBP_contrast9.pkl", "rb"))
# 50行から100行までを表示
df = df.iloc[50:100]
print(df.head())
print(df.columns)

