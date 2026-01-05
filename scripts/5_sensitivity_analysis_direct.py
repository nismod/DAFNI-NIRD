"""
temp_Xs:
- exposure portfolio: length, road_classification, form_of_way, trunk_road,
urban, lanes, averageWidth, road_label
- hazard characteristics: flood_type, flood_depth
- vulnerability: damage_level, damage_ratio
- unit repairing/maintenance cost: damage_value
Y: damage_cost
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

import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])  # local/processed_data
res_path = Path(load_config()["paths"]["output_path"])  # local/outputs


# %%
def normalised(df):
    Result = df.copy()
    for col in df.columns:
        if (df[col].max() - df[col].min()) != 0:
            Result[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return Result


def preprocess(intersections, road_links, lad_shp):
    # add geometry to intersections
    joined = intersections.copy()
    # intersections = intersections.copy()
    # # intersections = intersections.head(10)
    # intersections["_orig_index"] = np.arange(len(intersections))
    # intersections = intersections.merge(
    #     road_links[["e_id", "geometry"]],
    #     on="e_id",
    #     how="left",
    #     suffixes=("", "_road"),
    # )
    # intersections = gpd.GeoDataFrame(intersections, geometry="geometry")
    # # add admin info via spatial join
    # # ensure LAD shapefile has centroids
    # if "centroid" not in lad_shp.columns:
    #     lad_shp = lad_shp.assign(centroid=lad_shp.geometry.centroid)
    # else:
    #     # if centroid exists but not a geometry type, recompute to be safe
    #     if not isinstance(lad_shp.loc[0, "centroid"].__class__.__name__, str):
    #         lad_shp["centroid"] = lad_shp.geometry.centroid
    # joined = gpd.sjoin(
    #     intersections,
    #     lad_shp[["LAD21CD", "geometry", "centroid"]],
    #     how="left",
    #     predicate="intersects",
    # )
    # # compute distance from intersection geometry to LAD centroid — vectorized
    # joined["dist_to_centroid"] = joined.geometry.distance(
    #     joined["centroid"].fillna(joined.geometry)
    # )
    # # sort so the nearest centroid is first for each original intersection
    # joined = joined.sort_values(["_orig_index", "dist_to_centroid"])

    # # keep only the first match per original intersection (deterministic)
    # joined = (
    #     joined.drop_duplicates(subset="_orig_index", keep="first").reset_index(
    #         drop=False
    #     )
    # ).rename(columns={"index": "_orig_index"})

    # divide them into two sub-dataframes (surface, river)
    road_cols = [
        "length",
        "road_classification",
        "form_of_way",
        "trunk_road",
        "urban",
        "lanes",
        "averageWidth",
        "road_label",
        # "LAD21CD",
    ]

    # helper to build the surface/river DF
    def build_hazard_df(
        df: gpd.GeoDataFrame,
        hazard_prefix: str,
        depth_col: str,
        damage_level_col: str,
    ) -> pd.DataFrame:
        # base columns
        base = df[road_cols].copy()
        # include the depth and damage level if present
        extra_cols = [c for c in (depth_col, damage_level_col) if c in df.columns]
        if extra_cols:
            base = pd.concat([base, df[extra_cols]], axis=1)
        # unit costs
        uc_min_col = f"{hazard_prefix}_unit_cost_min"
        uc_max_col = f"{hazard_prefix}_unit_cost_max"
        uc_min = (
            df[uc_min_col]
            if uc_min_col in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        uc_max = (
            df[uc_max_col]
            if uc_max_col in df.columns
            else pd.Series([np.nan] * len(df), index=df.index)
        )
        base["damage_value"] = list(zip(uc_min, uc_max))

        # damage fraction columns (pattern matching)
        frac_cols = [
            c
            for c in df.columns
            if c.endswith(f"_{hazard_prefix}_damage_fraction")
            or f"_{hazard_prefix}_damage_fraction" in c
        ]
        # fallback: as in original, use filter(like=)
        if not frac_cols:
            frac_cols = [
                c for c in df.columns if f"_{hazard_prefix}_damage_fraction" in c
            ]

        if frac_cols:
            frac_df = df[frac_cols].replace(0, np.nan)
            frac_min = frac_df.min(axis=1, skipna=True)
            frac_max = frac_df.max(axis=1, skipna=True)
        else:
            frac_min = pd.Series([np.nan] * len(df), index=df.index)
            frac_max = pd.Series([np.nan] * len(df), index=df.index)
        base["damage_ratio"] = list(zip(frac_min, frac_max))

        # damage cost columns
        val_cols = [c for c in df.columns if f"_{hazard_prefix}_damage_value_" in c]
        if val_cols:
            val_df = df[val_cols].replace(0, np.nan)
            val_min = val_df.min(axis=1, skipna=True)
            val_max = val_df.max(axis=1, skipna=True)
        else:
            val_min = pd.Series([np.nan] * len(df), index=df.index)
            val_max = pd.Series([np.nan] * len(df), index=df.index)
        base["damage_cost"] = list(zip(val_min, val_max))
        base = base.reset_index(drop=True)
        base = base.explode(
            ["damage_value", "damage_ratio", "damage_cost"], ignore_index=True
        )
        return base

    # build surface and river dataframes
    surface_df = build_hazard_df(
        joined, "surface", "flood_depth_surface", "damage_level_surface"
    )
    surface_df.rename(
        columns={
            "flood_depth_surface": "flood_depth",
            "damage_level_surface": "damage_level",
        },
        inplace=True,
    )
    river_df = build_hazard_df(
        joined, "river", "flood_depth_river", "damage_level_river"
    )
    river_df.rename(
        columns={
            "flood_depth_river": "flood_depth",
            "damage_level_river": "damage_level",
        },
        inplace=True,
    )
    # Label each hazard type
    surface_df["flood_type"] = "surface"
    river_df["flood_type"] = "river"

    # Combine them into one DataFrame
    combined_df = pd.concat([surface_df, river_df], ignore_index=True)
    combined_df = combined_df[combined_df["damage_cost"].notnull()].reset_index(
        drop=True
    )
    combined_df.drop_duplicates(inplace=True)
    combined_df.trunk_road = combined_df.trunk_road.astype(str)

    return combined_df


# %%
# Load datasets
path = res_path / "damage_analysis" / "revision"
road_links = gpd.read_parquet(
    base_path / "networks" / "road" / "GB_road_links_with_bridges.gpq"
)
lad_shp = gpd.read_parquet(
    base_path
    / "census_datasets"
    / "admin_census_boundary_stats"
    / "gb_lad_2021_estimates.geoparquet"
)

intersections = pd.DataFrame()
for root, dirs, files in os.walk(path):
    for file in files:
        if file.startswith(
            "intersections"
        ):  # Load all intersections files from 17 historical flood events
            temp = pd.read_csv(os.path.join(root, file))
            intersections = pd.concat([intersections, temp], axis=0, ignore_index=True)

Xs = preprocess(intersections, road_links, lad_shp)
# %%
# temp_Xs = Xs.copy()
# remove damage level
temp_Xs = Xs[Xs.damage_level == "severe"].reset_index(drop=True)  #!!! update
# convert string objects to numeric values for sensitivity analysis
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

# admin_mapping = {v: idx for idx, v in enumerate(lad_shp["LAD21CD"].unique())}
flood_type_mapping = {"surface": 0, "river": 1}
damage_level_mapping = {"no": 0, "minor": 1, "moderate": 2, "extensive": 3, "severe": 4}

temp_Xs["road_classification"] = temp_Xs["road_classification"].map(
    road_classification_mapping
)
temp_Xs["form_of_way"] = temp_Xs["form_of_way"].map(form_of_way_mapping)
temp_Xs["trunk_road"] = temp_Xs["trunk_road"].map(trunk_road_mapping)
temp_Xs["road_label"] = temp_Xs["road_label"].map(road_label_mapping)
temp_Xs["flood_type"] = temp_Xs["flood_type"].map(flood_type_mapping)
temp_Xs["damage_level"] = temp_Xs["damage_level"].map(damage_level_mapping)
temp_Xs = normalised(temp_Xs)  # nomralize all input features between 0 and 1

# Morris
# Define the problem for Morris Sensitivity Analysis
D = 11  # Number of input variables
problem = {
    "num_vars": D,
    "names": [
        "Road Length",  # Length of the road link
        "Road Classification",  # e.g., A Road, B Road, Motorway
        "Carriageway Type",  #  e.g., Single Carriageway, Dual Carriageway
        "Location",  # Urban or rural classification (binary: 0 or 1)
        "Lanes",  # Number of lanes on the road
        "Road Width",  # Average width of the road link
        "Structure",  # Label indicating road structure (e.g., road, bridge, tunnel)
        "Flood Type",  # Type of flood (e.g., surface, river)
        "Flood Depth",  # Depth of flooding at the road link
        # "damage_level",  # Level of damage (e.g., no, minor, moderate, extensive, severe)
        "Damage Ratio",  # Damage ratio (min, max)
        "Unit Asset Value",  # Unit repairing/maintenance cost (min, max)"
    ],
    "bounds": [
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        [0, 1],
        # [0, 1],
        [0, 1],
        [0, 1],
    ],
}

# Convert input and output data to NumPy arrays
inputs = (
    temp_Xs.drop(columns=["trunk_road", "damage_cost", "damage_level"])
    .to_numpy()
    .astype(np.float64)
)  # Input variables
outputs = (
    temp_Xs["damage_cost"].to_numpy().astype(np.float64)
)  # Target variable (damage costs)

# Filter inputs and outputs to exclude NaN values
valid_indices = ~np.isnan(outputs)
inputs = inputs[valid_indices, :]
outputs = outputs[valid_indices]

# reshape inputs and outputs
B = inputs.shape[0] // (D + 1)  # Calculate B from the number of rows
N = (D + 1) * B  # Ensure it's a multiple of (D+1)
# Trim inputs and outputs to the correct length
inputs = inputs[:N, :]
outputs = outputs[:N]

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
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

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
# plt.savefig(r"C:\Oxford\Research\DAFNI\local\papers\figures\morris_direct.tif", dpi=300)
plt.show()

# %%
# Data
parameters = [
    "Road Length",
    "Road Classification",
    "Carriageway Type",
    "Location",
    "Lanes",
    "Road Width",
    "Structure",
    "Flood Type",
    "Flood Depth",
    "Damage Level",
    "Damage Ratio",
    "Unit Asset Value",
]
S1_abs = res_df["S1_abs"].values.tolist() / res_df["S1_abs"].values.sum()  # normalize
ST = res_df["ST"].values.tolist()

# Create scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(S1_abs, ST, color="b", alpha=0.7, marker="x")

# Annotate points
for i, param in enumerate(parameters):
    plt.annotate(
        param,
        (S1_abs[i], ST[i]),
        # fontsize=10,
        xytext=(5, 5),
        textcoords="offset points",
    )

# Labels and title
plt.xlabel(r"$\mu^*$")
plt.ylabel(r"$\sigma$")
plt.title(
    "Morris Sensitivity Analysis: Mean (abs) and Variance",
    pad=10,
    fontweight="bold",
)
plt.grid(True, linestyle="--", alpha=0.6)

# Show plot
plt.tight_layout()
# plt.savefig(r"C:\Oxford\Research\DAFNI\local\papers\figures\mean_var_direct.tif", dpi=300)
plt.show()


# %%
# Unified marker dictionary for all parameters
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
}
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 14


# Parameters for DIRECT damages
parameters = [
    "Road Length",
    "Road Classification",
    "Carriageway Type",
    "Location",
    "Lanes",
    "Road Width",
    "Structure",
    "Flood Type",
    "Flood Depth",
    #  "Damage Level",
    "Damage Ratio",
    "Unit Asset Value",
]

S1_abs = res_df["S1_abs"].values / res_df["S1_abs"].values.sum()
ST = res_df["ST"].values

plt.figure(figsize=(6.5, 6))
handles = []

for i, param in enumerate(parameters):
    h = plt.scatter(
        S1_abs[i],
        ST[i],
        marker=marker_map[param],
        color=color_map[param],  # ← unified color mapping
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
        s=70,
        label=param,
    )
    handles.append(h)

plt.xlabel(r"$\mu^*$")
plt.ylabel(r"$\sigma$")
plt.title("Severe Damage", pad=12, fontweight="bold")

plt.grid(True, linestyle="--", alpha=0.6)
# plt.legend(handles=handles, loc="lower right", borderaxespad=0.5)

plt.tight_layout()
plt.savefig(
    # r"C:\Oxford\Research\DAFNI\local\papers\figures\for revision\morris_direct.tif",
    r"C:\Oxford\Research\DAFNI\local\papers\figures\for revision\morris_direct_severe.tif",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
