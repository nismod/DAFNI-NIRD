# %%
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 26  # Title font size
plt.rcParams["axes.labelsize"] = 26  # Axis labels
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

# Data from the provided table
events = [
    "1953 UK",
    "1998 Easter_UK Floods",
    "2007 Summer_UK Floods/July",
    "2007 Summer_UK Floods/June",
    "2007 Summer_UK Floods/May",
    "2013 December_UK Storm Xaver",
    "2014 February_UK Southern England",
    "2015 December_UK Storm Frank",
    "2015 December_UK Storm Eva",
    "2015 December_UK Storm Desmond",
    "2016 January_UK Scotland",
    "2018 May_UK Midlands",
    "2019 November_UK Floods",
    "2020 February_UK Storm Dennis",
    "2020 February_UK Storm Ciara",
    "2023 October_UK ROI Storm Babet",
    "2024 January_UK Storm Henk_D",
]

mins = [
    329.4820981,
    29.22514093,
    70.63186313,
    46.48731279,
    18.12630973,
    94.74010432,
    51.54671064,
    0.001754147,
    0.008931398,
    8.839663903,
    12.29024379,
    9.689873517,
    4.153481686,
    85.12056683,
    42.41395348,
    26.00385339,
    19.07472797,
]

maxs = [
    444.1393108,
    41.34170173,
    96.28541314,
    63.75481296,
    23.32843194,
    139.7766137,
    74.17904342,
    0.00263122,
    0.013397097,
    11.54342231,
    17.64040142,
    13.01877389,
    5.545843834,
    120.2355598,
    58.7879092,
    36.18786773,
    27.48428994,
]

means = [
    386.8107045,
    35.28342133,
    83.45863814,
    55.12106287,
    20.72737083,
    117.258359,
    62.86287703,
    0.002192683,
    0.011164248,
    10.1915431,
    14.96532261,
    11.3543237,
    4.84966276,
    102.6780633,
    50.60093134,
    31.09586056,
    23.27950896,
]


errors = np.array(
    [
        np.array(means) - np.array(mins),
        np.array(maxs) - np.array(means),
    ]
)


# Setting up the figure
fig, ax = plt.subplots(figsize=(14, 10))
y_pos = np.arange(len(events))

# Plotting the values as a bar chart
ax.barh(
    events,
    means,
    xerr=errors,
    capsize=5,
    color="orange",
    alpha=0.7,
    edgecolor="black",
)


ax.set_yticks(y_pos)
ax.set_yticklabels(events)
ax.set_xlabel("Costs (£Million)", fontweight="bold")
ax.set_title(
    "Direct Asset Costs",
    fontweight="bold",
)

# Improve aesthetics
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()

plt.show()
