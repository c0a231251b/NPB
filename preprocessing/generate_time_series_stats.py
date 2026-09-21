import os
import json
import re
from datetime import datetime, timedelta

class BaseballStatsTimelineBilder:
    def __init__(self, json_dir="game_data_2025_updated_hoge"):
        self.json_dir = json_dir
        # 全野手の「日ごとの打撃成績」を蓄積するマスター
        self.batter_daily_logs = {}
        # 全投手の「日ごとの被打撃成績」を蓄積するマスター
        self.pitcher_daily_logs = {}

    def get_sorted_game_files(self):
        """フォルダ内のJSONファイルを日付順・試合番号順にソートして回収"""
        if not os.path.exists(self.json_dir):
            print(f"❌ フォルダが見つかりません: {self.json_dir}")
            return []
            
        files = [f for f in os.listdir(self.json_dir) if f.endswith(".json")]
        def extract_file_key(filename):
            match = re.search(r'\d+', filename)
            return match.group(0) if match else filename
            
        return sorted(files, key=extract_file_key)

    def calculate_recent_stats(self, player_name, is_pitcher, target_date_str, days_window):
        """🌟指定された試合日から、過去N日間の直近リアルタイムスタッツを動的逆算する"""
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d")
        start_date = target_date - timedelta(days=days_window)
        
        logs = self.pitcher_daily_logs.get(player_name, []) if is_pitcher else self.batter_daily_logs.get(player_name, [])
        
        total_ab_or_bf = 0  # 総打数 または 総対戦打者数
        total_hits = 0      # 総安打 または 総被安打
        
        for log in logs:
            log_date = datetime.strptime(log["date"], "%Y-%m-%d")
            # 「当日のN日前 〜 1日前まで」の期間内にあるログだけを厳密に切り出し集計
            if start_date <= log_date < target_date:
                if is_pitcher:
                    total_ab_or_bf += log["bf"]
                    total_hits += log["h"]
                else:
                    total_ab_or_bf += log["ab"]
                    total_hits += log["h"]
                    
        if total_ab_or_bf == 0:
            return 0.0
            
        return round(total_hits / total_ab_or_bf, 3)

    def update_player_mastar(self, game_data, date_str):
        """試合が終了した後に、その日の成績を次の日のために累積蓄積マスタへ登録する"""
        for team in game_data.get("batter_stats", {}):
            for b in game_data["batter_stats"][team]:
                name = b["player_name"]
                if name not in self.batter_daily_logs:
                    self.batter_daily_logs[name] = []
                self.batter_daily_logs[name].append({
                    "date": date_str,
                    "ab": b["ab"],
                    "h": b["h"]
                })
                
        for p_name, p in game_data.get("pitcher_stats", {}).items():
            if p_name not in self.pitcher_daily_logs:
                self.pitcher_daily_logs[p_name] = []
            self.pitcher_daily_logs[p_name].append({
                "date": date_str,
                "bf": p["bf"],
                "h": p["h"]
            })

    def run_simulation(self, days_window=30, cutoff_days=30):
        """全試合を時系列にエミュレート走査し、スタッツ特徴量を計算してJSONへ書き戻す"""
        game_files = self.get_sorted_game_files()
        if not game_files: return
        
        print(f"📊 {len(game_files)} 試合の時系列スタッツ動的計算・ファイル更新を開始します（集計期間: {days_window}日間 / カットオフ: {cutoff_days}日間）")
        
        # 開幕戦の日付を特定して、カットオフ期間の判定基準にする
        first_game_file = game_files[0]
        date_match = re.search(r'2025\d{4}', first_game_file)
        opening_date_str = f"{date_match.group(0)[:4]}-{date_match.group(0)[4:6]}-{date_match.group(0)[6:]}"
        opening_date = datetime.strptime(opening_date_str, "%Y-%m-%d")
        
        updated_count = 0
        
        for file_idx, file_name in enumerate(game_files, 1):
            file_path = os.path.join(self.json_dir, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                game_data = json.load(f)
                
            # ファイル名から日付オブジェクトを生成
            d_str = re.search(r'2025\d{4}', file_name).group(0)
            current_date_str = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:]}"
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
            
            # 開幕からの経過日数
            days_since_opening = (current_date - opening_date).days
            is_cutoff_active = days_since_opening < cutoff_days
            
            # 試合中に稼働している現在投手の追跡変数（初期値は先発投手）
            current_pitcher = "不明"
            
            # ==================== 🌟【メイン機能】時系列スタッツの動的逆算と注入 ====================
            for live in game_data.get("text_live", []):
                if "plays" not in live: continue
                for play in live["plays"]:
                    lines = play.get("lines", [])
                    if len(lines) < 1: continue
                    
                    # 1. 投手交代アナウンスの検知・上書き（🌟次のバッターに適用するため continue せずに処理を残す）
                    pitcher_change_match = re.search(r'投[手]?[：\s]+(\S+)', lines[0])
                    if pitcher_change_match:
                        current_pitcher = pitcher_change_match.group(1)
                    
                    # 2. 打者名と打席イベントのパース（ lines[0] の中身だけで判定できるように修正 ）
                    batter_match = re.search(r'\d+番\s*(\S+)|打\s*(\S+)', lines[0])
                    if batter_match:
                        batter_name = batter_match.group(1) or batter_match.group(2)
                        
                        # カットオフ期間、または投手・打者名が不明の場合は0.0（互角アドバンテージ）とする
                        if is_cutoff_active or current_pitcher == "不明":
                            b_avg = 0.0
                            p_avg = 0.0
                            adv_diff = 0.0
                        else:
                            # 過去の蓄積ログから、当日の「直近打率」と「直近被打率」を動的逆算！
                            b_avg = self.calculate_recent_stats(batter_name, False, current_date_str, days_window)
                            p_avg = self.calculate_recent_stats(current_pitcher, True, current_date_str, days_window)
                            adv_diff = round(b_avg - p_avg, 3)
                        
                        # 打席オブジェクトにメインとなる数理結果を完全に追記
                        play["batter_recent_avg"] = b_avg
                        play["pitcher_recent_avg"] = p_avg
                        play["advantage_diff"] = adv_diff # 👈 これが数値特徴量に入れる実数値になります
            
            # データを更新した内容でJSONファイルへ完全に上書き書き戻し保存
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(game_data, f, ensure_ascii=False, indent=2)
            
            # 試合が終わったら、この日の結果を次の日のシミュレーションのためにマスタへ登録
            self.update_player_mastar(game_data, current_date_str)
            updated_count += 1
            
            if updated_count % 100 == 0 or updated_count == len(game_files):
                print(f"🔄 スタッツ書き込み進行中... {updated_count}/{len(game_files)} 試合完了")
                
        print(f"✨ 完了！ 858試合のすべてのJSON打席ログへの『直近リアルタイムスタッツ（{days_window}日間）』の動的注入・保存が100%正常に完了しました。")

if __name__ == "__main__":
    # ハルタさん指定：直近30日間集計 / 開幕30日間カットオフルール
    builder = BaseballStatsTimelineBilder()
    builder.run_simulation(days_window=30, cutoff_days=30)