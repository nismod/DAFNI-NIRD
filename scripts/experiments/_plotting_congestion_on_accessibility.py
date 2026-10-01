# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

outpath = Path(r"C:\Oxford\Research\NIST\papers\Figures")
# Load the dataset
# df = pd.read_csv('access_variations.csv')

# For demonstration, creating the top portion of your data
data = {
    "TCITY15NM": [
        "London",
        "Leeds",
        "Cambridge",
        "Birmingham",
        "Manchester",
        "Liverpool",
        "Leicester",
        "Bristol",
        "Luton",
        "Salford",
        "Sheffield",
        "Milton Keynes",
        "Nottingham",
        "Oxford",
        "Solihull",
        "Guildford",
        "Derby",
        "Watford",
        "Newcastle upon Tyne",
        "Northampton",
    ],
    "flow_change_2021": [
        333347,
        52156,
        420,
        49846,
        9606,
        24826,
        16924,
        9960,
        1,
        7990,
        3876,
        3250,
        17796,
        706,
        17108,
        1,
        3102,
        1,
        12524,
        3112,
    ],  # Replaced 0 with 1 for log scale compatibility
    "flow_change_2050": [
        2748468,
        190515,
        185283,
        155340,
        140587,
        129967,
        73077,
        65636,
        63851,
        61174,
        61094,
        59545,
        58011,
        54344,
        44374,
        44132,
        43390,
        37474,
        37376,
        34340,
    ],
}
df = pd.DataFrame(data)

# Sort by 2050 value to have a clean hierarchy
df = df.sort_values(by="flow_change_2050", ascending=True)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 22  # Set default font size
plt.rcParams["axes.titlesize"] = 22  # Title font size
plt.rcParams["axes.labelsize"] = 18  # Axis labels font size
plt.rcParams["xtick.labelsize"] = 16  # x-axis ticks font size
plt.rcParams["ytick.labelsize"] = 16  # y-axis ticks font size

# Initialize the plot
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 8))

# Draw the connecting lines (the "bar" of the dumbbell)
ax.hlines(
    y=df["TCITY15NM"],
    xmin=df["flow_change_2021"],
    xmax=df["flow_change_2050"],
    color="#afbec9",
    linewidth=2,
    zorder=1,
)

# Plot the 2021 points
ax.scatter(
    df["flow_change_2021"],
    df["TCITY15NM"],
    color="#1f77b4",
    label="2021",
    s=80,
    zorder=2,
)

# Plot the 2050 points
ax.scatter(
    df["flow_change_2050"],
    df["TCITY15NM"],
    color="#ff7f0e",
    label="2050",
    s=80,
    zorder=2,
)

# Set Log Scale to handle London vs the rest fairly
ax.set_xscale("log")

# Styling and Labels
ax.set_title(
    "Congestion impacts on accessibility loss (2021 vs 2050)\n (top 20 cities ranked by 2050 loss)",
    fontsize=20,
    pad=15,
    weight="bold",
)
ax.set_xlabel("Flow Change (Log Scale)", labelpad=10)
ax.set_ylabel("City")
ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="none",
    fontsize=14,
)

# Clean up layout
sns.despine(left=True, bottom=True)
plt.tight_layout()

# Save or display the figure
plt.savefig(outpath / "flow_change_dumbbell.png", dpi=300)
plt.show()
