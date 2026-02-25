# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from matplotlib.patches import Patch

base_path = Path(
    r"C:\Oxford\Research\MACCHUB\local\scripts\outputs\intersections\direct_damage"
)

# %%
intersections = pd.read_csv(
    base_path / "intersections_scotland_with_damage_basefld.csv"
)
cols = ["e_id"] + intersections.filter(like="river_damage_value").columns.tolist()
intersections_subset = intersections[cols]

gp = (
    intersections_subset.groupby("e_id", as_index=False)
    .sum(numeric_only=True)
    .reset_index(drop=True)
)
gp = gp.loc[:, (gp != 0).any(axis=0)]
max_cols = ["e_id"] + gp.filter(like="river_damage_value_max").columns.tolist()
min_cols = ["e_id"] + gp.filter(like="river_damage_value_min").columns.tolist()
max_cost = gp[max_cols].select_dtypes(include="number").max(axis=1).sum()
min_cost = gp[min_cols].select_dtypes(include="number").max(axis=1).sum()
ave_cost = np.mean([max_cost, min_cost])
print(f"Max cost: £{max_cost:,.2f}")
print(f"Min cost: £{min_cost:,.2f}")
print(f"Average cost: £{ave_cost:,.2f}")

# wales, england, scotland

# %%
# Updated Data definitions
# (Label, Start, End)
# Merged 'No damage' and 'Minor' into 'Minor'
sophisticated_data = [
    ("Minor", 0, 200),
    ("Moderate", 200, 600),
    ("Extensive", 600, 1000),  # Capped at 1000
]

other_data = [
    ("Minor", 0, 100),
    ("Moderate", 100, 200),
    ("Extensive", 200, 600),
    ("Severe", 600, 1000),  # Capped at 1000
]

# Color map (removed 'No damage')
colors = {
    "Minor": "#f1c40f",  # Yellow
    "Moderate": "#e67e22",  # Orange
    "Extensive": "#e74c3c",  # Red
    "Severe": "#8e44ad",  # Purple
}

fig, ax = plt.subplots(figsize=(12, 4.5))


def plot_road_type(data, y_pos):
    for label, start, end in data:
        width = end - start
        ax.broken_barh(
            [(start, width)],
            (y_pos - 0.35, 0.7),
            facecolors=colors[label],
            edgecolor="black",
            linewidth=0.8,
        )

        # Determine text label
        if start >= 600:
            text = ">600"
        else:
            text = f"{start}-{end}"

        # Add text if bar is wide enough
        if width >= 40:
            ax.text(
                start + width / 2,
                y_pos,
                text,
                ha="center",
                va="center",
                color="black",
                fontsize=10,
                fontweight="bold",
            )


plot_road_type(sophisticated_data, 1)
plot_road_type(other_data, 0)

# Aesthetics
ax.set_yticks([0, 1])
ax.set_yticklabels(
    ["Other Roads", "Sophisticated Roads"], fontsize=12, fontweight="bold"
)
ax.set_xlabel("Overtopping Floodwater Depth (cm)", fontsize=12)
ax.set_title(
    "Damage Level Thresholds by Road Type and Flood Depth", fontsize=14, pad=20
)

# Legend
legend_elements = [
    Patch(facecolor=colors[lvl], edgecolor="black", label=lvl) for lvl in colors
]
ax.legend(
    handles=legend_elements,
    title="Damage Level",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    frameon=False,
)

# X-axis ticks
ticks = [0, 100, 200, 600, 1000]
ax.set_xticks(ticks)
ax.set_xticklabels(["0", "100", "200", "600", "1000+"])

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="x", linestyle="--", alpha=0.3)

# Add Footnote
plt.figtext(
    0.5,
    0.01,
    "Note: range > 600 are open-ended and capped at 1000cm for visualisation",
    ha="center",
    fontsize=10,
    fontstyle="italic",
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig(
    r"C:\Oxford\Research\MACCHUB\MT\figures\road_damage_depth.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Manually structure data extracted from Excel
damage_levels = ["Minor", "Moderate", "Extensive", "Severe"]

data = {
    "England": {
        "Baseline": [352, 979, 247, 5],
        "Future": [371, 996, 253, 5],
    },
    "Wales": {
        "Baseline": [142, 141, 78, 3],
        "Future": [128, 188, 121, 8],
    },
    "Scotland": {
        "Baseline": [161, 158, 40, 0],
        "Future": [88, 252, 47, 0],
    },
}

output_paths = []

for country, vals in data.items():
    hist = np.array(vals["Baseline"])
    fut = np.array(vals["Future"])
    x = np.arange(len(damage_levels))
    width = 0.35

    max_val = max(hist.max(), fut.max())
    ylim = ceil(max_val * 1.15)

    fig, ax = plt.subplots(figsize=(9, 5))

    bars1 = ax.bar(
        x - width / 2, hist, width, label="Baseline", edgecolor="black", linewidth=0.9
    )
    bars2 = ax.bar(
        x + width / 2, fut, width, label="Future", edgecolor="black", linewidth=0.9
    )

    for bars in [bars1, bars2]:
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + ylim * 0.01,
                    f"{int(h)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(damage_levels, fontsize=11)
    ax.set_ylabel("Number of cases", fontsize=11)
    ax.set_title(f"Comparison of Baseline vs Future — {country}", fontsize=14, pad=12)
    ax.set_ylim(0, ylim)
    ax.legend(frameon=True, edgecolor="0.8")

    plt.tight_layout()

    outpath = Path(r"C:\Oxford\Research\MACCHUB\MT\figures")
    filename = outpath / f"damage_comparison_{country.lower()}_updated.png"

    fig.savefig(filename, dpi=200)
    output_paths.append(filename)
    plt.close(fig)

output_paths
