import pandas as pd
import numpy as np

def apply_batter_types(input_csv):
    df = pd.read_csv(input_csv)
    
    # 0除算を防ぐためABが0の選手は除外または補正
    df = df[df['AB'] > 0].copy()

    # 各種指標の計算
    df['AVG'] = df['H'] / df['AB']
    df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF'])
    df['SLG'] = df['TB'] / df['AB']
    df['IsoD'] = df['OBP'] - df['AVG']

    # --- 5分類のフラグ立て ---
    # 1. 長距離砲
    df['type_power'] = ((df['HR'] > 15) | (df['SLG'] > 0.400)).astype(int)
    
    # 2. アベレージ
    df['type_avg'] = (df['AVG'] > 0.290).astype(int)
    
    # 3. 俊足
    df['type_speed'] = ((df['SB'] > 10) | (df['3B'] > 3)).astype(int)
    
    # 4. 選球眼
    df['type_eye'] = (df['IsoD'] > 0.08).astype(int)

    # 5. オールラウンダーの判定
    # 上記4つのうち3つ以上を満たすか
    type_sum = df[['type_power', 'type_avg', 'type_speed', 'type_eye']].sum(axis=1)
    df['type_all'] = (type_sum >= 3).astype(int)

    # 戦略A: オールラウンダーの場合は他のフラグを落とす（重複を避ける場合）
    # df.loc[df['type_all'] == 1, ['type_power', 'type_avg', 'type_speed', 'type_eye']] = 0

    # 必要な列だけを保存
    output_columns = [
        'team', 'name', 'Hand', 'AVG', 'HR', 'SLG', 'SB', 'IsoD',
        'type_power', 'type_avg', 'type_speed', 'type_eye', 'type_all'
    ]
    
    output_file = "classified_batter_stats.csv"
    df[output_columns].to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"完了！ '{output_file}' を保存しました。")
    
    # どんな分布になったか確認
    print("\n--- 各タイプの人数 ---")
    print(df[['type_power', 'type_avg', 'type_speed', 'type_eye', 'type_all']].sum())

if __name__ == "__main__":
    apply_batter_types("initial_stats_2024.csv")