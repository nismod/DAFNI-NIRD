# %%
from pathlib import Path
import pandas as pd
from collections import defaultdict
import ast
import json
import warnings

warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")

# %%
timetable = pd.read_csv(
    INPUT_PATH_MACC / "data" / "incoming" / "rails" / "timetable" / "timetable.csv"
)
timetable = timetable[
    (timetable.cargo_type == "ExpP") | (timetable.cargo_type == "OrdP")
].reset_index()  # select the train timetable records
timetable["path"] = timetable["path"].apply(ast.literal_eval)
timetable["stops"] = timetable["stops"].apply(ast.literal_eval)

# %%
# apply the filter of selected date
selected_date = "2025-05-19"
with open(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "timetable"
    / "daily_train_schedule.json",
    "r",
) as f:
    daily_schedule = json.load(f)

train_list = set(daily_schedule[selected_date])
timetable = timetable[timetable["train_id"].isin(train_list)].reset_index()

# %%
# create a list of station lists for the selected date
station_lists = [
    [station for station, stop in zip(row["path"], row["stops"]) if stop == "S"]
    for _, row in timetable.iterrows()
]
# %%
origin_to_destinations = defaultdict(set)
for station_list in station_lists:
    for i, origin in enumerate(
        station_list[:-1]
    ):  # skip last since it has no downstream
        for j in range(i + 1, len(station_list)):
            destination = station_list[j]
            if destination != origin:
                origin_to_destinations[origin].add(destination)

# %%
# convert sets back to lists
origin_to_destinations = {k: list(v) for k, v in origin_to_destinations.items()}
print(len(origin_to_destinations.keys()))  # 2768 origin stations

# %%
with open(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "timetable"
    / f"origin_to_destinations_{selected_date}.json",
    "wt",
) as f:
    json.dump(origin_to_destinations, f)
