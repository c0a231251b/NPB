import glob
import json
import os


JSON_DIR = "game_data" # JSONファイルが保存されているディレクトリ


def count_at_bats_in_game(game: dict) -> int:
	total = 0
	for section in game.get("text_live", []):
		inning = section.get("inning", "")
		if "回" not in inning:
			continue

		for play in section.get("plays", []):
			# 打席番号がある項目を1打席として数える
			if str(play.get("number", "")).strip():
				total += 1
	return total


def main() -> None:
	paths = sorted(glob.glob(os.path.join(JSON_DIR, "*.json")))
	if not paths:
		print(f"JSONが見つかりません: {JSON_DIR}")
		return

	grand_total = 0
	for path in paths:
		with open(path, "r", encoding="utf-8") as f:
			game = json.load(f)

		at_bats = count_at_bats_in_game(game)
		grand_total += at_bats
		print(f"{os.path.basename(path)}: {at_bats}打席")

	print("-" * 40)
	print(f"対象ファイル数: {len(paths)}")
	print(f"総打席数: {grand_total}")


if __name__ == "__main__":
	main()
