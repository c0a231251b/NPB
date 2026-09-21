import pandas as pd

def extract_unique_labels(csv_path):
    # CSV読み込み
    df = pd.read_csv(csv_path)

    # label列が存在するかチェック
    if "label" not in df.columns:
        raise ValueError("CSVに 'label' 列が存在しません。")

    # ユニーク値と出現数を取得
    label_counts = df["label"].value_counts()

    print("=== 打席結果のユニーク表現一覧 ===")
    print("train_all_added_55_features.csv の label 列に含まれるユニークな打席結果とその出現数:")
    print(f"打席結果の数: {len(label_counts)}")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # 必要ならリストとして返す
    return label_counts

if __name__ == "__main__":
    csv_file = "c:/Users/Admin/Desktop/NPB/evaluate/data/train_all_added_55_features.csv"
    extract_unique_labels(csv_file)
