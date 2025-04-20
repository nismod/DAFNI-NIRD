# %%
from pathlib import Path
import pandas as pd
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt

from nird.utils import load_config

base_path = Path(load_config()["paths"]["output_path"])
base_path = base_path / "rerouting_analysis" / "20250302" / "trip_isolations"
# %%
event_index_to_name = {
    "19": "2015 December_UK Storm Desmond",
    "18": "2018 May_UK Midlands",
    "16": "1953 UK",
    "15": "2013 December_UK Storm Xaver",
    "14": "2014 February_UK Southern England",
    "13": "2015 December_UK Storm Frank",
    "12": "2015 December_UK Storm Eva",
    "11": "2016 January_UK Scotland",
    "10": "2020 February_UK Storm Dennis",
    "9": "2020 February_UK Storm Ciara",
    "8": "1998 Easter_UK Floods",
    "7": "2007 Summer_UK Floods/July",
    "6": "2007 Summer_UK Floods/June",
    "5": "2007 Summer_UK Floods/May",
    "4": "2019 November_UK Floods",
    "3": "2023 October_UK ROI Storm Babet",
    "1": "2024 January_UK Storm Henk_D",
}

iso_dict = defaultdict(lambda: defaultdict(list))
for scenario in ["15", "30", "60"]:
    for root, dirs, files in os.walk(base_path / scenario):
        for d in dirs:
            print(d)
            for i in range(1, 111):
                p = base_path / scenario / str(d) / f"trip_isolations_{str(i)}.csv"
                df = pd.read_csv(p)
                event = event_index_to_name[str(d)]
                if df.empty:
                    iso_dict[event][scenario].append(0)
                else:
                    iso_dict[event][scenario].append(df.Car21.sum())

# %%
tt_iso_dict_15 = {}
tt_iso_dict_30 = {}
tt_iso_dict_60 = {}
for event, v in iso_dict.items():
    for scenario, vList in v.items():
        if scenario == "15":
            tt_iso_dict_15[event] = sum(vList)
        elif scenario == "30":
            tt_iso_dict_30[event] = sum(vList)
        else:
            tt_iso_dict_60[event] = sum(vList)

iso_df = pd.DataFrame.from_dict(tt_iso_dict_15, orient="index")
iso_df.rename(columns={0: "isolated flows"}, inplace=True)

# %%
# plots
# Plot
events = [
    "2015 December_UK Storm Desmond",
    "2018 May_UK Midlands",
    "1953 UK",
    "2013 December_UK Storm Xaver",
    "2014 February_UK Southern England",
    "2015 December_UK Storm Frank",
    "2015 December_UK Storm Eva",
    "2016 January_UK Scotland",
    "2020 February_UK Storm Dennis",
    "2020 February_UK Storm Ciara",
    "1998 Easter_UK Floods",
    "2007 Summer_UK Floods/July",
    "2007 Summer_UK Floods/June",
    "2007 Summer_UK Floods/May",
    "2019 November_UK Floods",
    "2023 October_UK ROI Storm Babet",
    "2024 January_UK Storm Henk_D",
]
means = [tt_iso_dict_30[event] for event in events]
# %%
# indirect damage costs
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 26  # Title font size
plt.rcParams["axes.labelsize"] = 26
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

# Sort data by 'means' in descending order
sorted_indices = np.argsort(means)[::-1]  # Get indices for sorting in descending order
events_sorted = [events[i] for i in sorted_indices]
means_sorted = [means[i] for i in sorted_indices]

# Create horizontal bar plot
fig, ax = plt.subplots(figsize=(12, 10))
y_pos = np.arange(len(events_sorted))

ax.barh(
    y_pos,
    means_sorted,
    capsize=5,
    color="grey",
    alpha=0.7,
    edgecolor="black",
)

ax.set_yticks(y_pos)
ax.set_yticklabels(events_sorted)  # Use sorted event labels
ax.set_xlabel("The Number of Isolated OD Flows", fontsize=20)
ax.set_title("Isolated Origin-Destination Flows", fontweight="bold", pad=20)

# Improve aesthetics
ax.invert_yaxis()  # Ensure highest value is at the top
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
# plt.savefig(
#     r"C:\Oxford\Research\DAFNI\local\papers\figures\trip_isolations.tif", dpi=300
# )
# Show plot
plt.show()
