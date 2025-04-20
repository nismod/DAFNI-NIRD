# %%
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
import pandas as pd
import geopandas as gpd

from pathlib import Path
from nird.utils import load_config
from shapely.geometry import Point

# Define paths
base_path = Path(load_config()["paths"]["output_path"])
input_path = base_path.parent / "processed_data" / "networks" / "road"
res_path = base_path / "rerouting_analysis" / "20250302" / "trip_isolations" / "30"

# %%
# Load road nodes and trip isolation data
nodes = gpd.read_parquet(input_path / "GB_road_nodes_with_bridges.gpq")

isolation = pd.DataFrame()
for i in range(1, 20):
    if i in [2, 17]:
        continue
    temp = pd.read_csv(res_path / f"{i}" / "trip_isolations_0.csv")
    isolation = pd.concat([isolation, temp], axis=0, ignore_index=True)

# Aggregate isolation data
isolation.rename(columns={"Car21": "flow"}, inplace=True)
isolation_gp = isolation.groupby(by=["origin_node"], as_index=False).agg(
    {"flow": "sum"}
)

# Merge with nodes to obtain geometries
isolation_gp = isolation_gp.merge(
    nodes[["id", "geometry"]].rename(columns={"id": "origin_node"}),
    how="left",
    on="origin_node",
)
isolation_gdf = gpd.GeoDataFrame(
    isolation_gp, geometry=isolation_gp.geometry, crs=nodes.crs
)
isolation_gdf = isolation_gdf.to_crs(epsg=3857)

# Define fixed UK map extent in Web Mercator (EPSG:3857)
uk_bounds = {
    "xlim": (-1_000_000, 400_000),
    "ylim": (6.2e6, 8.3e6),
}

# %%
# Add corner reference points
corner_coords = [
    (-1_000_000, 6.2e6),
    (-1_000_000, 8.3e6),
    (400_000, 6.2e6),
    (400_000, 8.3e6),
]
corner_data = {
    "origin_node": ["corner_1", "corner_2", "corner_3", "corner_4"],
    "flow": [0, 0, 0, 0],
    "geometry": [Point(x, y) for x, y in corner_coords],
}
corner_gdf = gpd.GeoDataFrame(corner_data, crs="EPSG:3857")
isolation_gdf = pd.concat([isolation_gdf, corner_gdf], ignore_index=True)

isolation_gdf["x"] = isolation_gdf.geometry.x
isolation_gdf["y"] = isolation_gdf.geometry.y

# %%
# Add major UK city reference points (approximate Web Mercator coordinates)
city_data = {
    "name": [
        "London",
        "Birmingham",
        "Manchester",
        "Leeds",
    ],
    "x": [
        -0.1278,
        -1.8985,
        -2.2426,
        -1.7457,
    ],
    "y": [
        51.5074,
        52.4862,
        53.4808,
        53.8258,
    ],
}
# Convert lat/lon to EPSG:3857
city_gdf = gpd.GeoDataFrame(
    city_data,
    geometry=gpd.points_from_xy(city_data["x"], city_data["y"], crs="EPSG:4326"),
).to_crs(epsg=3857)

# %%
# Create plot
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12
plt.rcParams["ytick.labelsize"] = 12

fig, ax = plt.subplots(figsize=(8, 10))

# KDE heatmap
sns.kdeplot(
    data=isolation_gdf,
    x="x",
    y="y",
    fill=True,
    cmap="Reds",
    levels=10,
    bw_adjust=0.7,
    clip=(uk_bounds["xlim"], uk_bounds["ylim"]),
    ax=ax,
)

# Plot city locations
ax.scatter(
    city_gdf.geometry.x,
    city_gdf.geometry.y,
    color="black",
    s=10,
    label="Major Cities",
    zorder=3,
)
for idx, row in city_gdf.iterrows():
    if row["name"] == "Leeds":
        ha = "left"
    else:
        ha = "right"
    ax.text(
        row.geometry.x,
        row.geometry.y,
        row["name"],
        fontsize=12,
        ha=ha,
        color="black",
        weight="bold",
        zorder=4,
    )

# Add basemap
ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.Positron, alpha=0.7)

# Labels and title
ax.set_xlabel("Easting (meters)")
ax.set_ylabel("Northing (meters)")
ax.set_title(
    "Spatial distribution of isolated flow hotspots", fontweight="bold", pad=15
)
# Save figure
plt.savefig(
    r"C:\Oxford\Research\DAFNI\local\papers\figures\heatmap-30.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
