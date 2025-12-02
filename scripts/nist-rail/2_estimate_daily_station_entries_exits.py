# %%
from pathlib import Path
import datetime
from datetime import timedelta
import geopandas as gpd
import pandas as pd
from collections import defaultdict

import warnings

warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")

# user-defined variables
from_year = 2023
to_year = 2024

# %%
# load all stations
nationrail_nodes = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)
stations = nationrail_nodes[
    nationrail_nodes.station_label == "train_station"
].reset_index()

# load train station usage information
usage = pd.read_csv(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "table-1415-time-series-of-passenger-entries-and-exits.csv",
    encoding="ISO-8859-1",
)

# selected the year of interest
usage = usage[
    [
        "Sort",
        "Station name",
        "Three Letter Code (TLC)",
        "National Location Code (NLC)",
        "Region",
        "Local authority: district or unitary",
        f"Apr {from_year} to Mar {to_year}",
    ]
]
usage[f"Apr {from_year} to Mar {to_year}"] = usage[
    f"Apr {from_year} to Mar {to_year}"
].astype("float")
usage = usage[usage[f"Apr {from_year} to Mar {to_year}"] != -999].reset_index()

usage = usage[["Station name", f"Apr {from_year} to Mar {to_year}"]].rename(
    columns={
        "Station name": "station_name",
        f"Apr {from_year} to Mar {to_year}": "entries_and_exits",
    }
)
# to extrach only train station usage information (from all types of stations)
usage["station_name"] = usage.station_name.str.upper()
usage = usage.merge(stations[["station_name", "TIPLOC"]], on="station_name", how="left")
usage = usage[usage.TIPLOC.notnull()].reset_index()  # -> train-od matrix
usage.drop(columns=["index"], inplace=True)  # -> 2559 station usage

# %%
# estimate train station daily usage (based on daily # trains)
timetable = pd.read_csv(
    INPUT_PATH_MACC / "data" / "incoming" / "rails" / "timetable" / "timetable.csv"
)
timetable = timetable[
    (timetable.cargo_type == "ExpP") | (timetable.cargo_type == "OrdP")
].reset_index()  # select train-related records
timetable["from_date"] = timetable["date_from"].astype(str)
timetable["from_date"] = (
    "20"
    + timetable.from_date.str[0:2]
    + "-"
    + timetable.from_date.str[2:4]
    + "-"
    + timetable.from_date.str[4:6]
)
timetable["to_date"] = timetable["date_to"].astype(str)
timetable["to_date"] = (
    "20"
    + timetable.to_date.str[0:2]
    + "-"
    + timetable.to_date.str[2:4]
    + "-"
    + timetable.to_date.str[4:6]
)
timetable["from_date"] = timetable.from_date.apply(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
)
timetable["to_date"] = timetable.to_date.apply(
    lambda x: datetime.datetime.strptime(x, "%Y-%m-%d")
)

weekdays = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
weekly_schedule = (
    timetable[["train_id"] + weekdays].groupby("train_id", as_index=False).max()
)
weekly_schedule_dict = defaultdict(lambda: defaultdict(int))
for idx, row in weekly_schedule.iterrows():
    train_id = row["train_id"]
    for day in weekdays:
        weekly_schedule_dict[train_id][day] = row[day]

# %%
# estimate the operating trains of each day
"""
overall length of dates: 20250518-20251213
"""
daily_schedule = defaultdict(set)
start_date = datetime.datetime.strptime("2025-05-18", "%Y-%m-%d")
end_date = datetime.datetime.strptime("2025-12-14", "%Y-%m-%d")
current_date = start_date
while current_date < end_date:
    weekday = weekdays[current_date.weekday()]  # 0: monday, 6: sunday
    daily_schedule[current_date] = (
        timetable.loc[
            (timetable.from_date <= current_date)
            & (timetable.to_date >= current_date)
            & timetable[f"{weekday}"]
            == 1,
            "train_id",
        ]
        .unique()
        .tolist()
    )
    current_date += timedelta(days=1)

# convert datetime keys to string format
daily_schedule = {k: list(v) for k, v in daily_schedule.items()}
daily_schedule = {k.strftime("%Y-%m-%d"): v for k, v in daily_schedule.items()}

# %%
# estimate the number of trains run on each day
daily_schedule_count = {k: len(v) for k, v in daily_schedule.items()}
total_trains = sum(daily_schedule_count.values())
daily_schedule_count_norm = {
    k: v / total_trains for k, v in daily_schedule_count.items()
}
selected_dates = list(daily_schedule_count_norm.keys())
usage_dates = usage.copy()
for date in selected_dates:
    usage_dates[date] = (
        usage_dates["entries_and_exits"] * daily_schedule_count_norm[date] * 0.5
    )
    usage_dates[date] = usage_dates[date].round(0)

# %%
# export daily train station usage
usage_dates.to_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "timetable"
    / "station_daily_out_in_flows.pq",
)
