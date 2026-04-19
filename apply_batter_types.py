import pandas as pd
import numpy as np

def apply_batter_types(input_csv):
    df = pd.read_csv(input_csv)
    
    # 0除算を防ぐためABが0の選手は除外
    df = df[df['AB'] > 0].copy()

    # 各種指標の計算
    df['AVG'] = df['H'] / df['AB']
    df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / (df['AB'] + df['BB'] + df['HBP'] + df['SF'])
    df['SLG'] = df['TB'] / df['AB']
    df['IsoD'] = df['OBP'] - df['AVG']

    # --- 5分類のフラグ立て ---
    df['type_power'] = ((df['HR'] > 15) | (df['SLG'] > 0.400)).astype(int)
    df['type_avg'] = (df['AVG'] > 0.290).astype(int)
    df['type_speed'] = ((df['SB'] > 10) | (df['3B'] > 3)).astype(int)
    df['type_eye'] = (df['IsoD'] > 0.08).astype(int)

    # 5. オールラウンダーの判定
    type_cols = ['type_power', 'type_avg', 'type_speed', 'type_eye']
    type_sum = df[type_cols].sum(axis=1)
    df['type_all'] = (type_sum >= 3).astype(int)

    # 保存処理
    output_columns = [
        'team', 'name', 'Hand', 'AVG', 'HR', 'SLG', 'SB', 'IsoD',
        'type_power', 'type_avg', 'type_speed', 'type_eye', 'type_all'
    ]
    output_file = "classified_batter_stats.csv"
    df[output_columns].to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"完了！ '{output_file}' を保存しました。\n")

    # --- 各タイプの人数と選手名の表示 ---
    types = {
        'type_power': ' 長距離砲 (Power)',
        'type_avg': ' アベレージ (Average)',
        'type_speed': ' 俊足 (Speed)',
        'type_eye': ' 選球眼 (Eye)',
        'type_all': ' オールラウンダー (All-Rounder)'
    }

    print("="*50)
    print("打者タイプ別・所属選手一覧 (2024)")
    print("="*50)

    for col, label in types.items():
        matched_players = df[df[col] == 1]
        count = len(matched_players)
        
        print(f"\n【{label}】 合計: {count}名")
        print("-" * 30)
        
        if count > 0:
            # チームごとに並び替えて表示
            sorted_players = matched_players.sort_values('team')
            for _, row in sorted_players.iterrows():
                print(f"[{row['team']}] {row['name']}")
        else:
            print("該当者なし")
        print("-" * 30)

if __name__ == "__main__":
    apply_batter_types("initial_stats_2024.csv")