# %%
from pathlib import Path
import os
import pandas as pd
from nird.utils import load_config
import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])

intersections_list = []
for root, _, files in os.walk(
    base_path.parent / "outputs" / "damage_analysis" / "20241229" / "final"  # "updated"
):
    for file in files:
        intersection_path = Path(root) / file
        intersections_list.append(intersection_path)

event_list = []
min_cost_list = []
max_cost_list = []
for event_path in intersections_list:
    intersections = pd.read_csv(event_path)
    flood_key = event_path.stem.split("_")[1]
    min_cost = intersections.filter(like="damage_value_min").max(axis=1).sum()
    max_cost = intersections.filter(like="damage_value_max").max(axis=1).sum()

    # more attributes should be added
    event_list.append(flood_key)
    min_cost_list.append(min_cost)
    max_cost_list.append(max_cost)

temp = pd.DataFrame(
    {
        "event_id": event_list,
        "damage_cost_min": min_cost_list,
        "damage_cost_max": max_cost_list,
    }
)
temp["damage_cost_mean"] = temp[["damage_cost_min", "damage_cost_max"]].mean(axis=1)
