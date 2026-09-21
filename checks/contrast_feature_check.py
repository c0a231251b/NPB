import json

with open(
    "game_data_2025_match_results_add_30day_stats/cl2025032801.json",
    encoding="utf-8"
) as f:
    data = json.load(f)

print(data["at_bat_features"][0])