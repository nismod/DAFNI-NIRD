# %%
import numpy as np
import matplotlib.pyplot as plt


# indirect damage costs
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 26  # Title font size
plt.rcParams["axes.labelsize"] = 26  # Axis labels
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

# Data from the table
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

scenario_15 = [
    0.00,
    20.91,
    12.86,
    15.30,
    0.00,
    3.10,
    11.28,
    153.58,
    183.72,
    0.00,
    6.06,
    0.00,
    0.00,
    26.71,
    13.99,
    10.61,
    8.72,
]
scenario_30 = [
    0.00,
    54.16,
    23.99,
    16.59,
    0.00,
    3.10,
    11.28,
    153.58,
    183.72,
    0.00,
    6.07,
    0.00,
    0.00,
    30.63,
    13.99,
    1.49,
    8.73,
]
scenario_60 = [
    0.00,
    20.91,
    12.86,
    15.30,
    0.00,
    3.10,
    11.28,
    153.58,
    183.72,
    0.00,
    6.06,
    0.00,
    0.00,
    26.71,
    13.99,
    10.61,
    8.72,
]

# Compute mean and standard deviation (uncertainty)
means = np.mean([scenario_15, scenario_30, scenario_60], axis=0)
uncertainties = np.std([scenario_15, scenario_30, scenario_60], axis=0)

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(len(events))

ax.barh(
    y_pos,
    means,
    xerr=uncertainties,
    capsize=5,
    color="royalblue",
    alpha=0.7,
    edgecolor="black",
)
# ax.set_yticks(y_pos)
ax.yaxis.set_visible(False)
ax.set_xlim(0, 450)
ax.set_yticklabels(events)
ax.set_xlabel(
    "Costs (£Million)",
    fontweight="bold",
)
ax.set_title(
    "Indirect Rerouting Costs",
    fontweight="bold",
)

# Improve aesthetics
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()

# Show plot
plt.show()
