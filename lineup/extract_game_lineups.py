import os
import json
import glob
import pandas as pd
import re
import matplotlib.pyplot as plt
# --- 日本語文字化け対策 ---
plt.rcParams['font.family'] = 'MS Gothic'

def extract_pure_stamen_from_lines(game_data):
    """
    ハルタさんの指摘通り、プレイログ(lines)を全走査し、
    先攻・後攻それぞれの1番〜9番で『最初に打席に立った選手』を確実に仕留める関数
    """
    if 'scoreboard' not in game_data or len(game_data['scoreboard']) < 2:
        return {}
        
    visitor_team = game_data['scoreboard'][0].get('team')
    home_team = game_data['scoreboard'][1].get('team')
    
    # 各チームのスタメンマップを初期化
    team_lineups = {
        visitor_team: {num: "Unknown" for num in range(1, 10)},
        home_team: {num: "Unknown" for num in range(1, 10)}
    }
    
    if 'text_live' not in game_data:
        return team_lineups
        
    order_pattern = re.compile(r'([1-9])番\s+(.+)$')
    
    # ★重要: text_live 内の要素をすべて漏れなくループ
    for live_item in game_data['text_live']:
        inning_str = live_item.get('inning', '')
        
        # 攻撃チームの判定
        if '表' in inning_str:
            current_team = visitor_team
        elif '裏' in inning_str:
            current_team = home_team
        else:
            continue # 「試合前」などのノイズはスルー
            
        # ★大修正: 1つの要素に閉じ込めず、すべての plays と lines を完全にフラットになめる
        plays = live_item.get('plays', [])
        for play in plays:
            lines = play.get('lines', [])
            for line in lines:
                match = order_pattern.search(line.strip())
                if match:
                    order_num = int(match.group(1)) # 1〜9番
                    p_name = match.group(2).split()[0] # 選手名（苗字）
                    
                    # 各チーム、その打順で「人生（試合）で一番最初に出現した名前」をスタメンとしてロック
                    if team_lineups[current_team][order_num] == "Unknown":
                        team_lineups[current_team][order_num] = p_name
                        
    return team_lineups

def main():
    json_files = glob.glob("game_data_2025_updated/*.json")
    if not json_files:
        print("Error: 'game_data_2025_updated' フォルダ内にJSONファイルが見つかりません。")
        return

    all_rows = []

    print(f"🔍 {len(json_files)} 個のJSONファイルから、linesに基づく真のスターティングラインナップを抽出中...")

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        game_id = os.path.splitext(file_name)[0]

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                game_data = json.load(f)
            except Exception:
                continue

            # ハルタさんロジックでスタメンを抽出
            team_lineups = extract_pure_stamen_from_lines(game_data)
            
            if 'scoreboard' in game_data and len(game_data['scoreboard']) >= 2:
                visitor_team = game_data['scoreboard'][0].get('team')
                home_team = game_data['scoreboard'][1].get('team')
                game_teams = [visitor_team, home_team]
                
                for team_name in game_teams:
                    if team_name in team_lineups:
                        lineup = team_lineups[team_name]
                        
                        row_data = {
                            "Game_ID": game_id,
                            "Team": team_name,
                            "Opponent": home_team if team_name == visitor_team else visitor_team
                        }
                        
                        for num in range(1, 10):
                            row_data[f"Order_{num}"] = lineup.get(num, "Unknown")
                            
                        all_rows.append(row_data)

    if all_rows:
        df_res = pd.DataFrame(all_rows)
        columns_order = ["Game_ID", "Team", "Opponent"] + [f"Order_{num}" for num in range(1, 10)]
        df_res = df_res[columns_order]
        
        output_filename = "all_games_starting_lineups.csv"
        df_res.to_csv(output_filename, index=False, encoding="utf-8-sig")
        
        print("="*60)
        print(f"💾 【ハルタさんロジック完全準拠】スタメンCSVを保存しました: {output_filename}")
        print(f"📊 抽出された総打線データ行数: {len(df_res)} 行")
        
        # Unknownの残存数を最終カウント
        unknown_count = (df_res[[f"Order_{num}" for num in range(1, 10)]] == "Unknown").sum().sum()
        print(f"✨ 完全に回収できなかった残存Unknown数: {unknown_count} 箇所")
        print("="*60)
        
        print("\n💡 生成されたCSVの先頭サンプルの表示:")
        print(df_res.head(4).to_string(index=False))
    else:
        print("Error: データを抽出できませんでした。")

if __name__ == "__main__":
    main()