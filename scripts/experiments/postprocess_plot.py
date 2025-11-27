# %%
from pathlib import Path
import os
import pandas as pd
import geopandas as gpd
from nird.utils import load_config
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])

# %%
# for trip isolation mapping
temp_path = (
    base_path.parent
    / "outputs"
    / "rerouting_analysis"
    / "20241229"
    / "60"
    / "17"
    / "dynamic"
    / "trip_isolations"
)
paths = []
for root, _, files in os.walk(temp_path):
    for file in files:
        paths.append(Path(root) / file)

file_dict = {}
for path in paths:
    file = pd.read_csv(path)
    k = path.stem.split("_")[2]
    v = file.Car21.sum()
    file_dict[k] = v

# %%
x = [i for i in range(110)]
y = [file_dict[str(i)] for i in x]

plt.figure(figsize=(8, 6))
plt.plot(x, y, marker="o", linestyle="-", color="blue", label="Isolated Trips")

# Add labels and title
plt.title("Number of Isolated Trips Over 110 Days", fontsize=16)
plt.xlabel("Days", fontsize=14)
plt.ylabel("Number of Isolated Trips", fontsize=14)

# Grid, legend, and tick adjustments
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()

# %%
# for edge flow mapping
edge_flows = pd.DataFrame(
    columns=["d0", "d1", "d2", "d3", "d4", "d5", "d15", "d30", "d60", "d90", "d110"]
    # columns=["d0", "d1", "d2", "d3", "d4", "d5", "d15", "d30", "d37"]
)

for i in [0, 1, 2, 3, 4, 5, 15, 30, 60, 90, 110]:
    # for i in [0, 1, 2, 3, 4, 5, 15, 30, 37]:
    temp = gpd.read_parquet(
        base_path.parent
        / "outputs"
        / "rerouting_analysis"
        / "20241229"
        / "60"
        / "17"
        / "dynamic"
        / "edge_flows"
        / f"edge_flows_{str(i)}.gpq"
    )
    edge_flows[f"d{str(i)}"] = temp.acc_flow

edge_flows2 = edge_flows.copy()

for i in [0, 1, 2, 3, 4, 5, 15, 30, 60, 90, 110]:
    # for i in [0, 1, 2, 3, 4, 5, 15, 30, 37]:
    edge_flows2[f"d{str(i)}"] = (
        edge_flows2[f"d{str(i)}"] / edge_flows2[f"d{str(i)}"].max()
    )

edge_flows3 = pd.concat([temp[["e_id", "geometry"]], edge_flows2], axis=1)
edge_flows_final = gpd.GeoDataFrame(
    edge_flows3, geometry="geometry", crs=temp.geometry.crs
)

# %%
columns = [f"d{val}" for val in [0, 1, 3, 5, 15, 30, 60, 90, 110]]
# columns = [f"d{val}" for val in [0, 1, 2, 3, 4, 5, 15, 30, 37]]
class_bins = [0.2, 0.4, 0.6, 0.8, 1.0]  # Classes to visualise
line_thickness = {0.2: 1, 0.4: 2, 0.6: 3, 0.8: 4, 1.0: 5}  # Thickness per class

# Create a 3x3 subplot
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
axes = axes.flatten()


# Function to reclassify y values
def reclassify(y):
    for threshold in class_bins:
        if y <= threshold:
            return threshold
    return None


# Visualise each column as a subplot
for i, col in enumerate(columns):
    ax = axes[i]

    # Create a new column for reclassified values
    edge_flows_final["class"] = edge_flows_final[col].apply(reclassify)

    # Filter out values below 0.2
    filtered_gdf = edge_flows_final[edge_flows_final["class"] > 0.2]

    # Plot each line with the corresponding thickness
    for geom, cls in zip(filtered_gdf.geometry, filtered_gdf["class"]):
        if geom.geom_type == "LineString":
            x_coords, y_coords = geom.xy
            ax.plot(x_coords, y_coords, linewidth=line_thickness[cls], color="blue")

    # Set title and remove axes for a cleaner look
    ax.set_title(f"{col}", fontsize=40)
    ax.set_axis_off()

# Hide unused subplots
for i in range(len(columns), len(axes)):
    axes[i].set_visible(False)

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
