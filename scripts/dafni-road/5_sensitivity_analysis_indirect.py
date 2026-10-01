"""
Xs:
- exposure portfolio: length, road_classification, form_of_way, trunk_road,
urban, lanes, averageWidth, road_label
- hazard characteristics: flood_depth
- vulnerability: damage_level, damage_ratio
- initial disruption: disrupted_flow
- flow change: change_flow
Y: indirect rerouting cost
"""

# %%
import os
from pathlib import Path

import pandas as pd
import numpy as np

import geopandas as gpd
from nird.utils import load_config
from SALib.analyze import morris

import seaborn as sns
import matplotlib.pyplot as plt

import gc
import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])  # local/processed_data
res_path = Path(load_config()["paths"]["output_path"])  # local/outputs

# %%
CONV_METER_TO_MILE = 0.000621371
CONV_MILE_TO_KM = 1.60934
CONV_KM_TO_MILE = 0.621371
PENCE_TO_POUND = 0.01


def voc_func(
    speed: float,
) -> float:
    s = speed * CONV_MILE_TO_KM  # km/hour
    lpkm = 0.178 - 0.00299 * s + 0.0000205 * (s**2)  # fuel consumption (liter/km)
    voc_per_km = 140 * lpkm * PENCE_TO_POUND  # average petrol cost: 140 pence/liter

    return voc_per_km  # £/km


def cost_func(
    distance: float,  # meter
    speed: float,  # mph
) -> float:
    if speed == 0:
        return np.nan
    time = distance * CONV_METER_TO_MILE / speed  # hour
    ave_occ = 1.06
    vot = 17.69  # £/hour
    voc_per_km = voc_func(speed)  # £/km
    c_time = time * ave_occ * vot  # £
    c_fuel = distance * CONV_METER_TO_MILE * CONV_MILE_TO_KM * voc_per_km  # £

    return c_time + c_fuel  # total cost per trip in £


def normalised(df):
    Result = df.copy()
    for col in df.columns:
        if (df[col].max() - df[col].min()) != 0:
            Result[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return Result


# %%
# load datasets
cols = [
    "e_id",
    "road_classification",
    "form_of_way",
    "trunk_road",
    "urban",
    "lanes",
    "averageWidth",
    "road_label",
    "flood_depth_max",
    "damage_level_max",
    "disrupted_flow",
    "change_flow",
    "rerouting_cost",
    "depth_thres",
]
edges = pd.DataFrame()
path = res_path / "rerouting_analysis" / "revision"

# %%
for depth_key in [15, 30, 60]:
    for event_key in range(1, 17):
        if event_key == 2:
            continue
        else:
            edge_flow_path = path / f"{depth_key}" / f"{event_key}"
            for _, _, files in os.walk(edge_flow_path):
                for file in files:
                    if file.startswith("edge_flows_"):
                        df = gpd.read_parquet(edge_flow_path / file)
                        # calculate rerouting cost per edge
                        pre_cost = df.apply(
                            lambda row: cost_func(
                                row["geometry"].length, row["initial_flow_speeds"]
                            ),
                            axis=1,
                        )
                        post_cost = df.apply(
                            lambda row: cost_func(
                                row["geometry"].length, row["acc_speed"]
                            ),
                            axis=1,
                        )
                        df["rerouting_cost"] = (post_cost - pre_cost).clip(
                            lower=0
                        ) * df["change_flow"]
                        df["depth_thres"] = depth_key
                        # only keep edges with rerouting cost > 0
                        df = df[df.rerouting_cost > 0].reset_index(drop=True)
                        df = df[cols]
                        print(
                            f"Depth: {depth_key}, Event: {event_key}, File: {file}"
                            f" Rows: {len(df)} Completed."
                        )

                        # preprocess to expand hazard and vulnerability attributes
                        edges = pd.concat([edges, df], axis=0, ignore_index=True)

                        del df
                        gc.collect()

# %%
edges = pd.read_parquet(path / "edges_revised.pq")

# Sensitivity analysis using Morris method
road_classification_mapping = {
    "B Road": 0,
    "A Road": 2,
    "Motorway": 1,
}
form_of_way_mapping = {
    "Single Carriageway": 0,
    "Collapsed Dual Carriageway": 1,
    "Slip Road": 2,
    "Dual Carriageway": 3,
    "Roundabout": 4,
}
trunk_road_mapping = {
    "True": 0,
    "False": 1,
}
road_label_mapping = {
    "road": 0,
    "bridge": 1,
    "tunnel": 2,
}
damage_level_mapping = {"no": 0, "minor": 1, "moderate": 2, "extensive": 3, "severe": 4}

edges["road_classification"] = edges["road_classification"].map(
    road_classification_mapping
)
edges["form_of_way"] = edges["form_of_way"].map(form_of_way_mapping)
edges["trunk_road"] = edges["trunk_road"].astype(str)
edges["trunk_road"] = edges["trunk_road"].map(trunk_road_mapping)
edges["road_label"] = edges["road_label"].map(road_label_mapping)
edges["damage_level_max"] = edges["damage_level_max"].map(damage_level_mapping)
edges.drop(columns=["e_id"], inplace=True)
edges = normalised(edges)

# %%
D = 9  # number of input factors
problem = {
    "num_vars": D,
    "names": [
        "Road Classification",  # e.g., A Road, B Road, Motorway
        "Carriageway Type",  #  e.g., Single Carriageway, Dual Carriageway, ...
        # "trunk",  # e.g., "True" or "False"
        "Location",  # e.g., 0 (non-urban) or 1 (urban)
        "Lanes",  # Number of lanes on the road
        "Road Width",  # Average width of the road link
        "Structure",  # e.g., road, bridge, tunnel
        "Flood Depth",  # Max flood depth
        "Damage Level",  # Max damage level
        # "disrupted_flow",  # Initial disrupted flow
        # "change_flow",  # Change in flow due to disruption
        "Speed-Depth Curve",
    ],
    "bounds": [
        [0, 1],
        [0, 1],
        # [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
    ],
}

# %%
inputs = (
    edges.drop(
        columns=["rerouting_cost", "disrupted_flow", "change_flow", "trunk_road"]
    )
    .to_numpy()
    .astype(np.float64)
)
outputs = edges["rerouting_cost"].to_numpy().astype(np.float64)
# outputs = edges["change_flow"].to_numpy().astype(np.float64)
valid_indices = ~np.isnan(outputs)
inputs = inputs[valid_indices, :]
outputs = outputs[valid_indices]

# reshape inputs and outputs
B = inputs.shape[0] // (D + 1)  # Calculate B from the number of rows
N = (D + 1) * B  # Ensure it's a multiple of (D+1)
# Trim inputs and outputs to the correct length
inputs = inputs[:N, :]
outputs = outputs[:N]

# %%
# Perform Morris Sensitivity Analysis
morris_results = morris.analyze(
    problem, inputs, outputs
)  # Analyze sensitivity of inputs

# Create a DataFrame for sensitivity analysis results
res_df = pd.DataFrame(
    {
        "Parameters": problem["names"],  # Input parameter names
        "S1": morris_results["mu"],  # First-order sensitivity indices
        "ST": morris_results["sigma"],  # Total sensitivity indices
        "S1_abs": morris_results["mu_star"],  # Absolute mean sensitivity
        "S1_abs_conf": morris_results["mu_star_conf"],  # Confidence intervals
    }
)
res_df

# %%
# plots
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 16  # Title font size
plt.rcParams["axes.labelsize"] = 14  # Axis labels
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
# Example: Plotting Morris Mean and Variance
morris_mean = morris_results["mu_star"] / (
    morris_results["mu_star"].sum()
)  # Morris mean (normalised)
morris_variance = morris_results["sigma"]  # Morris variance
# combine results into a single dataframe
df = pd.DataFrame(
    {"factor": problem["names"], "mean": morris_mean, "variance": morris_variance}
)
# sort values based on mean from high to low
df = df.sort_values(by="mean", ascending=False)


def create_gradient(values, cmap_name="Blues"):
    """
    values: numeric array (sorted high → low)
    returns: list of RGBA colours mapped to values
    """
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(values.min(), values.max())
    return [cmap(norm(v)) for v in values]


mean_colors = create_gradient(df["mean"].values, cmap_name="Blues")
var_colors = create_gradient(df["variance"].values, cmap_name="Reds")
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(8, 6))

# ----- Mean Plot -----
axes[0].barh(df["factor"], df["mean"], color=mean_colors)
axes[0].invert_yaxis()  # highest at top
axes[0].set_title("Morris Sensitivity: Mean (First-Order Effect)", fontsize=14)
axes[0].set_xlabel("Normalised Morris Mean")
axes[0].set_ylabel("Factors")

# ----- Variance Plot -----
axes[1].barh(df["factor"], df["variance"], color=var_colors)
axes[1].invert_yaxis()
axes[1].set_title(
    "Morris Sensitivity: Variance (Interaction / Nonlinearity)", fontsize=14
)
axes[1].set_xlabel("Morris Variance")
axes[1].set_ylabel("")
plt.tight_layout()
# plt.savefig(r"C:\Oxford\Research\DAFNI\local\papers\figures\morris_indirect.tif", dpi=300)
plt.show()

# %%
marker_map = {
    "Road Length": "o",
    "Road Classification": "s",
    "Carriageway Type": "D",
    "Location": "^",
    "Lanes": "v",
    "Road Width": "<",
    "Structure": ">",
    "Flood Type": "p",
    "Flood Depth": "*",
    "Damage Level": "X",
    "Damage Ratio": "h",
    "Unit Asset Value": "P",
    "Speed-Depth Curve": "d",
}

# Unified color dictionary for all parameters
color_map = {
    "Road Length": "#1f77b4",  # blue
    "Road Classification": "#ff7f0e",  # orange
    "Carriageway Type": "#2ca02c",  # green
    "Location": "#d62728",  # red
    "Lanes": "#9467bd",  # purple
    "Road Width": "#8c564b",  # brown
    "Structure": "#e377c2",  # pink
    "Flood Type": "#7f7f7f",  # gray
    "Flood Depth": "#bcbd22",  # olive
    "Damage Level": "#17becf",  # cyan
    "Damage Ratio": "#9edae5",  # light blue
    "Unit Asset Value": "#aec7e8",  # lighter blue
    "Speed-Depth Curve": "#c49c94",
}
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14

# Parameters for INDIRECT losses
parameters = [
    "Road Classification",
    "Carriageway Type",
    "Location",
    "Lanes",
    "Road Width",
    "Structure",
    "Flood Depth",
    "Damage Level",
    "Speed-Depth Curve",
]

S1_abs = res_df["S1_abs"].values / res_df["S1_abs"].values.sum()
ST = res_df["ST"].values

plt.figure(figsize=(6.5, 6))
handles = []

for i, param in enumerate(parameters):
    h = plt.scatter(
        S1_abs[i],
        ST[i],
        marker=marker_map[param],  # ← SAME marker as direct plot
        color=color_map[param],  # ← SAME color as direct plot
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
        s=70,
        label=param,
    )
    handles.append(h)

plt.xlabel(r"$\mu^*$")
plt.ylabel(r"$\sigma$")
plt.title("Rerouting Losses", pad=12, fontweight="bold")

plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(handles=handles, loc="lower right", borderaxespad=0.5)

plt.tight_layout()
plt.savefig(
    r"C:\Oxford\Research\DAFNI\local\papers\figures\for revision\morris_indirect.tif",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
