import os
from pathlib import Path

import pandas as pd
import numpy as np

from nird.utils import load_config
from SALib.analyze import morris

import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])
res_path = Path(load_config()["paths"]["output_path"])

# %%
path = res_path / "damage_analysis" / "20241229" / "final"
test = pd.DataFrame()
for root, dirs, files in os.walk(path):
    for file in files:
        if file.startswith(
            "intersections"
        ):  # Optional: ensure you're reading only .csv files
            temp = pd.read_csv(os.path.join(root, file))
            test = pd.concat([test, temp], axis=0, ignore_index=True)

# damage fractions
fraction_cols = test.filter(like="_fraction").columns
test[fraction_cols] = test[fraction_cols].clip(upper=1.0)
damage_fraction_min = (
    test.filter(like="_fraction", axis=1).replace(0, np.nan).min(axis=1, skipna=True)
)
damage_fraction_min[damage_fraction_min.isnull()] = 0.0
damage_fraction_max = test.filter(like="_fraction", axis=1).max(axis=1)
test["damage_fractions"] = list(zip(damage_fraction_min, damage_fraction_max))

# damage values
damage_value_min = test.filter(like="_unit_cost_min", axis=1).max(axis=1, skipna=True)
damage_value_max = test.filter(like="_unit_cost", axis=1).max(axis=1)
test["damage_values"] = list(zip(damage_value_min, damage_value_max))

# damage costs
damage_cost_min = test.filter(like="_damage_value_min", axis=1).max(axis=1, skipna=True)
damage_cost_max = test.filter(like="_damage_value", axis=1).max(axis=1)
test["damage_costs"] = list(zip(damage_cost_min, damage_cost_max))

# %%
test = test[
    [
        "road_classification",
        "form_of_way",
        "trunk_road",
        "urban",
        "lanes",
        "road_label",
        "damage_fractions",
        "damage_values",
        "damage_costs",
    ]
]
test = test.explode(
    ["damage_fractions", "damage_values", "damage_costs"], ignore_index=True
)  # (49080, 0)

# %%
road_classification_mapping = {
    "B Road": 0,
    "A Road": 2,
    "Motorway": 1,
}
form_of_way_mapping = {
    "Single Carriageway": 0,
    "Collapsed Dual Carriageway": 1,
    "Slip Road": 1,
    "Dual Carriageway": 1,
    "Roundabout": 1,
}
trunk_road_mapping = {
    False: 1,
    True: 0,
}
road_label_mapping = {
    "road": 0,
    "tunnel": 2,
    "bridge": 1,
}

test["road_classification"] = test["road_classification"].map(
    road_classification_mapping
)
test["form_of_way"] = test["form_of_way"].map(form_of_way_mapping)
test["trunk_road"] = test["trunk_road"].map(trunk_road_mapping)
test["road_label"] = test["road_label"].map(road_label_mapping)

# %%
# Morris
# Define the problem for Morris Sensitivity Analysis
D = 8  # Number of input variables
problem = {
    "num_vars": D,
    "names": [
        "road_classification",  # e.g., A Road, B Road, Motorway
        "form_of_way",  #  e.g., Single Carriageway, Dual Carriageway
        "trunk_road",  # Whether the road is a trunk road (True/False)
        "urban",  # Urban or rural classification (binary: 0 or 1)
        "lanes",  # Number of lanes on the road
        "road_label",  # Label indicating road structure (e.g., road, bridge, tunnel)
        "damage_fraction",  # Fraction of damage (clipped between 0.0 and 1.0)
        "damage_values",  # Estimated damage values (scaled between 0.0 and 1.3)
    ],
    "bounds": [
        [0, 2],  # Bounds for road classification
        [0, 1],  # Bounds for form of way
        [0, 1],  # Bounds for trunk road
        [0, 1],  # Bounds for urban classification
        [2, 20],  # Bounds for number of lanes
        [0, 2],  # Bounds for road label
        [0.0, 1.0],  # Bounds for damage fraction
        [0.0, 1.3],  # Bounds for damage values
    ],
}

# Convert input and output data to NumPy arrays
inputs = test.iloc[:, :-1].to_numpy().astype(np.float64)  # Input variables
outputs = (
    test["damage_costs"].to_numpy().astype(np.float64)
)  # Target variable (damage costs)

# Filter inputs and outputs to exclude NaN values
valid_indices = ~np.isnan(outputs)  # Exclude rows with NaN in outputs
inputs = inputs[valid_indices, :]
outputs = outputs[valid_indices]

# %%
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

# Print results for Morris Sensitivity Analysis
print("Morris Mean (First-Order Effect):", morris_results["mu"])
print("Morris Variance (Total Effect):", morris_results["sigma"])
print("Morris Mean Confidence:", morris_results["mu_star"])
print("Variance Confidence Interval:", morris_results["mu_star_conf"])

# %%
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

# %%
# plots
# Example: Plotting Morris Mean and Variance
morris_mean = morris_results["mu_star"]  # Morris mean
morris_variance = morris_results["sigma"]  # Morris variance

# Set the style of the plots
sns.set(style="whitegrid")

# Create a figure and axis for both plots in one figure
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar plot for Morris Mean (First-Order Effect)
sns.barplot(x=morris_mean, y=problem["names"], ax=axes[0], color="blue")
axes[0].set_title("Morris Sensitivity Analysis: Mean (First-Order Effect)", fontsize=14)
axes[0].set_xlabel("Morris Mean", fontsize=12)
axes[0].set_ylabel("Factors", fontsize=12)

# Bar plot for Morris Variance (Total Effect)
sns.barplot(x=morris_variance, y=problem["names"], ax=axes[1], color="red")
axes[1].set_title("Morris Sensitivity Analysis: Variance (Total Effect)", fontsize=14)
axes[1].set_xlabel("Morris Variance", fontsize=12)
axes[1].set_ylabel("Factors", fontsize=12)

# Adjust spacing between plots
plt.tight_layout()

# Show the plots
plt.show()

# %%
# %%
# Data
plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 16  # Title font size
plt.rcParams["axes.labelsize"] = 14  # Axis labels
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
parameters = [
    "road_classification",
    "form_of_way",
    "trunk",
    "urban",
    "number_of_lanes",
    "road_structure",
    "damage_curve",
    "asset_value",
]
S1_abs = [
    0.01842870320064087,
    0.10553139363221894,
    0.031343725594092396,
    0.06921428982624625,
    0.19791185399459427,
    0.1796260999619082,
    0.30166573455315615,
    0.17707074252845398,
]
ST = [
    0.12676654944509624,
    1.519979698588346,
    0.30790500956508626,
    0.3121532357771505,
    2.933644899079449,
    3.6713228626584953,
    3.389077703765054,
    3.3449757874549415,
]

# Create scatter plot
plt.figure(figsize=(8, 5))
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
plt.savefig(r"C:\Oxford\Research\DAFNI\local\papers\figures\morris.tif", dpi=300)
plt.show()
