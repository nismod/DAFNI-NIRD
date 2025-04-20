# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx

import geopandas as gpd
from nird.utils import load_config
from shapely.geometry import Point, LineString

import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])
res_path = Path(load_config()["paths"]["output_path"])

# %%
res_flows = gpd.read_parquet(
    res_path / "base_scenario" / "edge_flows_validation_0205.gpq"
)
res_flows["acc_flow_cars"] = (res_flows["acc_flow"] / 1.06).round(0).astype(int)
res_flows_gp = res_flows.groupby(by=["id"], as_index=False).agg(
    {"acc_flow": "first", "acc_flow_cars": "first"}
)
obs_flows = gpd.read_parquet(
    base_path / "networks" / "road" / "link_traffic_counts.geoparquet"
)
res_merge = pd.merge(
    obs_flows[["id", "Cars_and_taxis"]], res_flows_gp, how="left", on="id"
)

res_merge_filter = res_merge[res_merge.Cars_and_taxis.notnull()]

# %%
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 20  # Title font size
plt.rcParams["axes.labelsize"] = 14  # Axis labels
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

plt.figure(figsize=(10, 5))
plt.hist(
    res_merge_filter["acc_flow_cars"],
    bins=34,
    alpha=0.6,
    label="Simulated",
    color="blue",
    density=True,
)
# Plot histograms
plt.hist(
    res_merge_filter["Cars_and_taxis"],
    bins=30,
    alpha=0.6,
    label="Observed",
    color="red",
    density=True,
)

# Labels and legend
plt.xlabel("Passenger Flow")
plt.ylabel("Density")
plt.legend(loc="best", fontsize=12)
plt.title(
    "Histogram distribution of Observed and Simulated Passenger Flows",
    fontweight="bold",
    fontsize=16,
)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.savefig(
    res_path.parent / "papers" / "figures" / "base_flows_hist.png",
)
plt.show()

# %%
# for simulated results
simu_notation_dict = {
    0: "0 - 6,110",
    1: "6,110 - 20,669",
    2: "20,669 - 46,725",
    3: "46,725 - 101,156",
    4: "101,156 - 171,447",
}
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 12  # Title font size
plt.rcParams["axes.labelsize"] = 10  # Axis labels
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

# network flow visualisation
gdf = gpd.read_parquet(res_path / "base_scenario" / "edge_flows_visualisation.pq")
gdf = gdf.to_crs(epsg=3857)

# Normalize line width (scaling values from 0-4 to 0.5-4 pt)
gdf["line_width"] = gdf["vis_label"].apply(lambda x: 0.1 + (x / 4) * (4 - 0.5))
gdf["flow_category"] = gdf["vis_label"].map(simu_notation_dict)

unique_categories = gdf.dropna(subset=["flow_category"])[
    ["flow_category", "line_width"]
].drop_duplicates()
unique_categories = unique_categories.sort_values(by="line_width")

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
for category, width in zip(
    unique_categories["flow_category"], unique_categories["line_width"]
):
    subset = gdf[gdf["flow_category"] == category]
    subset.plot(ax=ax, linewidth=width, color="blue", alpha=0.7, label=category)

# add boundary lines (for visualisation)
uk_bounds = {
    "xlim": (-1_000_000, 400_000),  # Adjust as per UK region
    "ylim": (6.2e6, 8.3e6),
}
corners_3857 = [
    (uk_bounds["xlim"][0], uk_bounds["ylim"][0]),
    (uk_bounds["xlim"][0], uk_bounds["ylim"][1]),
    (uk_bounds["xlim"][1], uk_bounds["ylim"][0]),
    (uk_bounds["xlim"][1], uk_bounds["ylim"][1]),
]
corner_points = [Point(x, y) for x, y in corners_3857]
boundary_links = [
    LineString([corner_points[0], corner_points[1]]),
    LineString([corner_points[0], corner_points[2]]),
    LineString([corner_points[1], corner_points[3]]),
    LineString([corner_points[2], corner_points[3]]),
]
boundary_data = {
    "origin_node": ["added_line1", "added_line2", "added_line3", "added_line4"],
    "flow": [0, 0, 0, 0],
    "geometry": boundary_links,
}
boundary_df = gpd.GeoDataFrame(boundary_data, crs="EPSG:3857")
gdf = pd.concat([gdf, boundary_df], ignore_index=True)

# Plot the GeoDataFrame
gdf.plot(ax=ax, linewidth=gdf["line_width"], color="blue", alpha=0.7)

# Add a light grey OSM basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)

# Create a legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    title="Passenger Flow (AADT)",
    loc="upper right",
    title_fontsize=10,
    fontsize=10,
    frameon=True,
    shadow=True,
)

ax.set_xlabel("Easting (meters)")
ax.set_ylabel("Northing (meters)")
# Show plot
plt.title(
    "Spatail Passenger-Flows by Roads in Great Britain (AADT, 2021) ",
    fontweight="bold",
)
# plt.show()
plt.savefig(
    res_path.parent / "papers" / "figures" / "base_flows_simu.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# for observation (AADT)
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 12  # Title font size
plt.rcParams["axes.labelsize"] = 10  # Axis labels
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

obs_notation_dict = {
    0: "0 - 4,580",
    1: "4,580 - 15,494",
    2: "15,494 - 35,026",
    3: "35,026 - 75,829",
    4: "75,829 - 128,540",
}
# network flow visualisation
gdf = gpd.read_parquet(res_path / "base_scenario" / "obs_flows_visualisation.pq")
# drop the null rows
gdf = gdf[gdf.vis_label.notnull()].reset_index(drop=True)

# %%
# Reproject to Mechator Prj System
gdf = gdf.to_crs(epsg=3857)

# Normalize line width (scaling values from 0-4 to 0.5-4 pt)
gdf["line_width"] = gdf["vis_label"].apply(lambda x: 0.1 + (x / 4) * (4 - 0.5))
gdf["flow_category"] = gdf["vis_label"].map(obs_notation_dict)

unique_categories = gdf.dropna(subset=["flow_category"])[
    ["flow_category", "line_width"]
].drop_duplicates()
unique_categories = unique_categories.sort_values(by="line_width")

# Create plot
fig, ax = plt.subplots(figsize=(10, 8))
for category, width in zip(
    unique_categories["flow_category"], unique_categories["line_width"]
):
    subset = gdf[gdf["flow_category"] == category]
    subset.plot(ax=ax, linewidth=width, color="red", alpha=0.7, label=category)

# add boundary lines (for visualisation)
uk_bounds = {
    "xlim": (-1_000_000, 400_000),  # Adjust as per UK region
    "ylim": (6.2e6, 8.3e6),
}
corners_3857 = [
    (uk_bounds["xlim"][0], uk_bounds["ylim"][0]),
    (uk_bounds["xlim"][0], uk_bounds["ylim"][1]),
    (uk_bounds["xlim"][1], uk_bounds["ylim"][0]),
    (uk_bounds["xlim"][1], uk_bounds["ylim"][1]),
]
corner_points = [Point(x, y) for x, y in corners_3857]
boundary_links = [
    LineString([corner_points[0], corner_points[1]]),
    LineString([corner_points[0], corner_points[2]]),
    LineString([corner_points[1], corner_points[3]]),
    LineString([corner_points[2], corner_points[3]]),
]
boundary_data = {
    "origin_node": ["added_line1", "added_line2", "added_line3", "added_line4"],
    "flow": [0, 0, 0, 0],
    "geometry": boundary_links,
}
boundary_df = gpd.GeoDataFrame(boundary_data, crs="EPSG:3857")
gdf = pd.concat([gdf, boundary_df], ignore_index=True)

# Plot the GeoDataFrame
gdf.plot(ax=ax, linewidth=gdf["line_width"], color="red", alpha=0.7)

# Add a light grey OSM basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)

# Create a legend
handles, labels = ax.get_legend_handles_labels()
legend = ax.legend(
    handles,
    labels,
    title="Passenger Flow (AADT)",
    loc="upper right",
    title_fontsize=10,
    fontsize=10,
    frameon=True,
    shadow=True,
)

ax.set_xlabel("Easting (meters)")
ax.set_ylabel("Northing (meters)")

# Show plot
plt.title(
    "UK Department of Transportation AADT Records (2021)",
    fontweight="bold",
)
# plt.show()
plt.savefig(
    res_path.parent / "papers" / "figures" / "base_flows_obs.png",
    dpi=300,
    bbox_inches="tight",
)
