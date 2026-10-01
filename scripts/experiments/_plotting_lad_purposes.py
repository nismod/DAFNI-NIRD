# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

outpath = Path(r"C:\Oxford\Research\NIST\papers\Figures")
# %%
lad = gpd.read_parquet(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation\lad_shp.gpq"
)
temp = pd.read_parquet(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation\odpfc_od_agg_2050_edge_purpose_flow.pq"
)  # odpfc_od_agg_edge_purpose_flow / odpfc_od_agg_2050_edge_purpose_flow
roadcom = pd.read_parquet(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation\road_length_percentage_lad.pq"
)

# Merge edge-level flow/purpose onto LAD-road data
df = roadcom.merge(temp, left_on="e_id", right_on="edge", how="left")

# Allocate flow by the edge's proportion of total road length in the LAD
df["flow_alloc"] = df["flow"] * df["length"] / df["road_length"]

# Sum by LAD and purpose
lad_purpose_flow = (
    df.groupby(["LAD24CD", "purpose"], as_index=False)["flow_alloc"]
    .sum()
    .rename(columns={"flow_alloc": "flow"})
)

# lad_purpose_flow.to_csv(
#     r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation\lad_purpose_flow_2050.csv",
#     index=False,
# )

# %%
# ACTIVE BLOCK
lad_purpose_flow = pd.read_csv(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation\lad_purpose_flow_2021.csv"
)
lad_purpose_flow_future = pd.read_csv(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation\lad_purpose_flow_2050.csv"
)

# Use Times New Roman
mpl.rcParams["font.family"] = "Times New Roman"

purpose_map = {
    "commute": "Commute",
    "town": "Non-mandatory",
    "education": "Education",
    "employment": "Business",
}

# Set explicit colors so both subplots use the same palette
color_map = {
    "Commute": "#1f77b4",  # blue
    "Non-mandatory": "#d62728",  # red
    "Education": "#2ca02c",  # green
    "Business": "#ff7f0e",  # orange
}


def prepare_plot_pct(df):
    plot_df = df.pivot(index="LAD24CD", columns="purpose", values="flow").fillna(0)
    plot_pct = plot_df.div(plot_df.sum(axis=1), axis=0).fillna(0) * 100
    plot_pct = plot_pct.rename(columns=purpose_map)

    # Keep consistent order
    ordered_cols = ["Commute", "Non-mandatory", "Education", "Business"]
    plot_pct = plot_pct[[c for c in ordered_cols if c in plot_pct.columns]]
    return plot_pct


def annotate_low_commute_lads(ax, plot_pct, threshold=80):
    for i, lad in enumerate(plot_pct.index):
        commute_share = plot_pct.loc[lad, "Commute"]
        if commute_share < threshold:
            ax.text(
                i,
                commute_share + 1.0,
                lad,
                ha="center",
                va="bottom",
                fontsize=5,
                rotation=90,
                clip_on=False,
            )


plot_pct_2021 = prepare_plot_pct(lad_purpose_flow)
plot_pct_2050 = prepare_plot_pct(lad_purpose_flow_future)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300, sharey=True)

# Use the same colors in the same order for both panels
colors_2021 = [color_map[c] for c in plot_pct_2021.columns]
colors_2050 = [color_map[c] for c in plot_pct_2050.columns]

plot_pct_2021.plot(
    kind="bar",
    stacked=True,
    width=1.0,
    edgecolor="none",
    ax=axes[0],
    legend=False,
    color=colors_2021,
)

plot_pct_2050.plot(
    kind="bar",
    stacked=True,
    width=1.0,
    edgecolor="none",
    ax=axes[1],
    legend=False,
    color=colors_2050,
)

# annotate_low_commute_lads(axes[0], plot_pct_2021)
# annotate_low_commute_lads(axes[1], plot_pct_2050)

axes[0].set_title("2021", fontsize=14)
axes[1].set_title("2050", fontsize=14)

for ax in axes:
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.grid(axis="x", visible=False)

axes[0].set_ylabel("Share of allocated flow (%)", fontsize=14)
axes[1].set_ylabel("")

# Shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    title="Purpose",
    title_fontsize=14,
    fontsize=14,
    loc="upper center",
    ncol=4,
    frameon=False,
    bbox_to_anchor=(0.5, 1.05),
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig(outpath / "lad_purposes.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.show()

# %%
# Scatter plot: 2021 versus 2050 purpose shares by LAD
scatter_data = (
    plot_pct_2021.add_suffix("_2021")
    .join(plot_pct_2050.add_suffix("_2050"), how="outer")
    .fillna(0)
)

fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=300)

scatter_purposes = [
    c
    for c in color_map
    if f"{c}_2021" in scatter_data.columns and f"{c}_2050" in scatter_data.columns
]

for purpose in scatter_purposes:
    ax.scatter(
        scatter_data[f"{purpose}_2021"],
        scatter_data[f"{purpose}_2050"],
        s=20,
        alpha=0.70,
        color=color_map[purpose],
        label=purpose,
        edgecolors="white",
        linewidths=0.25,
    )

ax.plot(
    [0, 100],
    [0, 100],
    color="0.35",
    linewidth=0.9,
    linestyle=(0, (4, 3)),
    zorder=0,
)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks(range(0, 101, 20))
ax.set_yticks(range(0, 101, 20))
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel(
    "Share of distance-weighted network use, 2021 (%)", fontsize=12, labelpad=6
)
ax.set_ylabel(
    "Share of distance-weighted network use, 2050 (%)", fontsize=12, labelpad=6
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("0.25")
ax.spines["bottom"].set_color("0.25")
ax.tick_params(axis="both", labelsize=9, length=3, color="0.25")
ax.set_axisbelow(True)
ax.grid(color="0.88", linewidth=0.6)

# Magnify the low-share region where most non-commute changes occur.
inset_ax = ax.inset_axes([0.13, 0.50, 0.40, 0.40])
for purpose in scatter_purposes:
    inset_ax.scatter(
        scatter_data[f"{purpose}_2021"],
        scatter_data[f"{purpose}_2050"],
        s=20,
        alpha=0.70,
        color=color_map[purpose],
        edgecolors="white",
        linewidths=0.25,
    )

inset_ax.plot(
    [0, 10],
    [0, 10],
    color="0.35",
    linewidth=0.7,
    linestyle=(0, (4, 3)),
    zorder=0,
)
inset_ax.set_xlim(0, 10)
inset_ax.set_ylim(0, 10)
inset_ax.set_xticks([0, 5, 10])
inset_ax.set_yticks([0, 5, 10])
inset_ax.set_aspect("equal", adjustable="box")
inset_ax.tick_params(axis="both", labelsize=7, length=2)
inset_ax.grid(color="0.88", linewidth=0.5)
inset_ax.set_axisbelow(True)
inset_ax.set_facecolor("white")
for spine in inset_ax.spines.values():
    spine.set_color("0.25")
    spine.set_linewidth(0.8)

mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.35", lw=0.8)

ax.legend(
    title="Trip Purpose",
    title_fontsize=10,
    fontsize=10,
    ncol=4,
    loc="lower center",
    bbox_to_anchor=(0.5, 1.01),
    frameon=False,
    handletextpad=0.4,
    columnspacing=1.0,
)
fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.84)
plt.savefig(outpath / "lad_purposes_scatter.png", dpi=300, bbox_inches="tight")
plt.show()

# %%
mpl.rcParams["font.family"] = "Times New Roman"
# -------------------------------------------------
# 1) Prepare baseline and future shares
# -------------------------------------------------
baseline = lad_purpose_flow.copy()
future = lad_purpose_flow_future.copy()

baseline_pivot = baseline.pivot(
    index="LAD24CD", columns="purpose", values="flow"
).fillna(0)

future_pivot = future.pivot(index="LAD24CD", columns="purpose", values="flow").fillna(0)

# Make sure both datasets use the same LADs and purposes
all_lads = baseline_pivot.index.union(future_pivot.index)
all_purposes = baseline_pivot.columns.union(future_pivot.columns)

baseline_pivot = baseline_pivot.reindex(
    index=all_lads, columns=all_purposes, fill_value=0
)
future_pivot = future_pivot.reindex(index=all_lads, columns=all_purposes, fill_value=0)

# Convert to shares within each LAD
baseline_pct = baseline_pivot.div(baseline_pivot.sum(axis=1), axis=0).fillna(0)
future_pct = future_pivot.div(future_pivot.sum(axis=1), axis=0).fillna(0)

# Difference: future - baseline
diff_pct = future_pct - baseline_pct

# Optional: rename town to non-mandatory
diff_pct = diff_pct.rename(columns={"town": "non-mandatory", "employment": "business"})

# %%
# -------------------------------------------------
# 2) Plot spatial difference maps
# -------------------------------------------------
purposes = diff_pct.columns.tolist()

# Shared symmetric colour scale centered at zero
vmax = diff_pct.abs().max().max()
norm = mpl.colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

fig, axes = plt.subplots(
    1, len(purposes), figsize=(5 * len(purposes), 7), dpi=300, constrained_layout=True
)

if len(purposes) == 1:
    axes = [axes]

for ax, purpose in zip(axes, purposes):
    temp = diff_pct[[purpose]].reset_index()

    gdf = lad.merge(temp, on="LAD24CD", how="left")

    gdf.plot(
        column=purpose,
        ax=ax,
        cmap="RdBu_r",
        norm=norm,
        linewidth=0.2,
        edgecolor="white",
        legend=False,
        missing_kwds={"color": "lightgrey", "edgecolor": "white"},
    )

    ax.set_title(purpose.title(), fontsize=24, fontweight="bold")
    ax.set_axis_off()

# Shared colorbar
sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
sm.set_array([])

cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
# cbar.set_label("Change in share of flow (future - baseline)", fontsize=18)
# plt.savefig(outpath / "flow_change_spatial.png", dpi=300)
plt.show()

# %%
# -----------------------------
# 3) Prepare baseline and future
# -----------------------------
# Change in share: future - baseline
# diff_pct = (future_pct - baseline_pct) * 100  # percentage points

# # Optional rename for legend
# diff_pct = diff_pct.rename(columns={"town": "Non-mandatory"})

# -----------------------------
# 2) Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 4), dpi=300)

diff_pct.plot(kind="bar", stacked=True, width=1.0, edgecolor="none", ax=ax)

# Zero line
ax.axhline(0, color="black", linewidth=0.8)

# Hide x-axis labels and ticks
ax.set_xticks([])
ax.set_xlabel("")

# Y-axis
# ax.set_ylabel("Change in share of flow (future - baseline)", fontsize=14)

# Clean style
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", color="0.9", linewidth=0.8)
ax.grid(axis="x", visible=False)

# Legend
ax.legend(
    title="Purpose",
    title_fontsize=13,
    fontsize=12,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=False,
)
plt.tight_layout()
# plt.savefig(outpath / "flow_change_dumbbell.png", dpi=300)

plt.show()
