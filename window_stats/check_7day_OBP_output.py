# 確認後削除してよいスクリプト

import json
import os
from pathlib import Path

# 変更点1：出力フォルダ名を指定
OUTPUT_DIR = "game_data_2025_match_results_add_7day_stats_HR"

def check_ops_output(output_dir):

    json_files = sorted(Path(output_dir).glob("*.json"))

    if not json_files:
        print("[ERROR] 出力フォルダに JSON がありません")
        return

    print(f"[INFO] JSON ファイル数: {len(json_files)}")

    missing_at_bat = []
    missing_ops_fields = []
    summary = {
        "total_files": len(json_files),
        "files_with_at_bat": 0,
        "files_missing_at_bat": 0,
        "files_with_all_fields": 0,
        "files_missing_fields": 0,
    }

    for f in json_files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        at_bats = data.get("at_bat_features")

        if not at_bats:
            missing_at_bat.append(f.name)
            summary["files_missing_at_bat"] += 1
            continue

        summary["files_with_at_bat"] += 1

        # OPS フィールドが存在するかチェック
        ok = True
        for ab in at_bats:
            # 変更点2：出塁率 - 被出塁率の差分を特徴量として付与するバージョンに合わせて、必要なフィールド名を変更
            if not all(k in ab for k in ["batter_hr_7d", "pitcher_hr_7d", "contrast_feature"]):
                ok = False
                break

        if ok:
            summary["files_with_all_fields"] += 1
        else:
            summary["files_missing_fields"] += 1
            missing_ops_fields.append(f.name)

    # 結果表示
    print("\n===== 検証結果 =====")
    print(f"総ファイル数: {summary['total_files']}")
    print(f"at_bat_features があるファイル: {summary['files_with_at_bat']}")
    print(f"at_bat_features が無いファイル: {summary['files_missing_at_bat']}")
    print(f"HR フィールドが揃っているファイル: {summary['files_with_all_fields']}")
    print(f"HR フィールドが欠けているファイル: {summary['files_missing_fields']}")

    if missing_at_bat:
        print("\n--- at_bat_features が無いファイル ---")
        for name in missing_at_bat:
            print(name)

    if missing_ops_fields:
        print("\n--- HR フィールドが欠けているファイル ---")
        for name in missing_ops_fields:
            print(name)

    print("\n[INFO] 検証完了")


if __name__ == "__main__":
    check_ops_output(OUTPUT_DIR)
