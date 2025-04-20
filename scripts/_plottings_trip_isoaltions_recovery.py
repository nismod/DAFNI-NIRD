# %%
from pathlib import Path
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from nird.utils import load_config

base_path = Path(load_config()["paths"]["output_path"])
base_path = base_path / "rerouting_analysis" / "20241229" / "trip_isolations" / "30"

event_index_to_name = {
    "19": "Thames Lloyd's RDS (synthetic event)",
    "18": "1953 UK",
    "17": "1998 Easter_UK Floods",
    "16": "2007 Summer_UK Floods/July",
    "15": "2007 Summer_UK Floods/June",
    "14": "2007 Summer_UK Floods/May",
    "13": "2013 December_UK Storm Xaver",
    "12": "2014 February_UK Southern England",
    "11": "2015 December_UK Storm Frank",
    "10": "2015 December_UK Storm Eva",
    "9": "2015 December_UK Storm Desmond",
    "8": "2016 January_UK Scotland",
    "7": "2018 May_UK Midlands",
    "6": "2019 November_UK Floods",
    "5": "2020 February_UK Storm Dennis",
    "4": "2020 February_UK Storm Ciara",
    "3": "2023 October_UK ROI Storm Babet",
    "2": "2024 January_UK Storm Henk_U",
    "1": "2024 January_UK Storm Henk_D",
}
iso_dict = defaultdict(list)
for root, dirs, files in os.walk(base_path):
    for d in dirs:
        print(d)
        for i in range(1, 111):
            p = base_path / str(d) / f"trip_isolations_{str(i)}.csv"
            df = pd.read_csv(p)
            event = event_index_to_name[str(d)]
            if df.empty:
                iso_dict[event].append(0)
            else:
                iso_dict[event].append(df.Car21.sum())

# %%
# plots
# indirect damage costs
plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 30  # Title font size
plt.rcParams["axes.labelsize"] = 16  # Axis labels
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

# Set a nicer style using seaborn
sns.set(style="whitegrid")

# Create a figure and axis for the plot
plt.figure(figsize=(12, 8))

# Plot data for each event (key in iso_dict)
for event_id, values in iso_dict.items():
    if event_id in [
        "2014 February_UK Southern England",
        "2013 December_UK Storm Xaver",
        "2007 Summer_UK Floods/July",
    ]:
        plt.plot(values, label=f"{event_id}", linewidth=4, alpha=0.7, linestyle="--")
    else:
        plt.plot(values, label=f"{event_id}", linewidth=2, alpha=0.7)

# Add labels and title with better font sizes
plt.xlabel("Days (from 1 to 110)", fontsize=18)
plt.ylabel("The number of isolated passenger flows", fontsize=18)
plt.title(
    "The dynamics of flow isolation during road restoration",
    fontsize=22,
    fontweight="bold",
)

# Customize the x and y limits for better clarity
plt.xlim(0, 109)
plt.xticks(ticks=range(0, 110, 10), labels=range(1, 111, 10), fontsize=12)
plt.ylim(min(min(iso_dict.values())), max(max(iso_dict.values())) + 10000)

# Add a grid for easier readability
plt.grid(True, linestyle="--", alpha=0.5)

# Display legend with a background for better visibility
plt.legend(
    title="Events", title_fontsize="14", loc="upper right", fontsize=11, shadow=True
)

# Tight layout to ensure everything fits well
plt.tight_layout()

# Display the plot
plt.show()
