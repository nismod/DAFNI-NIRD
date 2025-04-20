# %%
import matplotlib.pyplot as plt
import numpy as np

# Data
plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 26  # Title font size
plt.rcParams["axes.labelsize"] = 26  # Axis labels
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20

events = [
    # "Thames Lloyd's RDS",
    "1953 UK",
    "1998 Easter_UK Floods",
    "2007 Summer_July",
    "2007 Summer_June",
    "2007 Summer_May",
    "2013 Dec_Storm Xaver",
    "2014 Feb_Southern England",
    "2015 Dec_Storm Frank",
    "2015 Dec_Storm Eva",
    "2015 Dec_Storm Desmond",
    "2016 Jan_Scotland",
    "2018 May_Midlands",
    "2019 Nov_Floods",
    "2020 Feb_Storm Dennis",
    "2020 Feb_Storm Ciara",
    "2023 Oct_Storm Babet",
    # "2024 Jan_Storm Henk_U",
    "2024 Jan_Storm Henk_D",
]

mean_indirect_costs = [
    # 3.76,
    0.00,
    32.00,
    16.57,
    15.73,
    0.00,
    3.10,
    11.28,
    153.58,
    183.72,
    0.00,
    6.07,
    0.00,
    0.00,
    28.02,
    13.99,
    7.57,
    # 8.37,
    8.72,
]

mean_direct_costs = [
    # 95.94,
    386.81,
    35.28,
    83.46,
    55.12,
    20.73,
    117.26,
    62.86,
    0.002,
    0.011,
    10.19,
    14.97,
    11.35,
    4.85,
    102.68,
    50.60,
    31.10,
    # 26.18,
    23.28,
]

# Sorting by total cost (optional, for better visual clarity)
sorted_indices = np.argsort(
    [d + i for d, i in zip(mean_direct_costs, mean_indirect_costs)]
)
events = [events[i] for i in sorted_indices]
mean_direct_costs = [mean_direct_costs[i] for i in sorted_indices]
mean_indirect_costs = [mean_indirect_costs[i] for i in sorted_indices]

# Plot
fig, ax = plt.subplots(figsize=(14, 10))

y_pos = np.arange(len(events))
ax.barh(
    y_pos,
    mean_direct_costs,
    label="Mean Direct Costs (£M)",
    color="orange",
    alpha=0.7,
    edgecolor="black",
)
ax.barh(
    y_pos,
    mean_indirect_costs,
    left=mean_direct_costs,
    label="Mean Indirect Costs (£M)",
    color="royalblue",
    alpha=0.7,
    edgecolor="black",
)

# Labels and formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(events)
ax.set_xlabel("Costs (£ Million)", fontweight="bold")
# ax.set_xlim(0, 400)
ax.set_title(
    "Direct vs Indirect Damage Costs",
    fontweight="bold",
)
ax.legend()

plt.gca().invert_yaxis()  # Invert y-axis so the highest cost is on top
plt.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
