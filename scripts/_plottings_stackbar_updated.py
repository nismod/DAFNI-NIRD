# %% Import Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Read Data
df = pd.read_csv(
    r"C:\Oxford\Research\DAFNI\local\outputs\damage_analysis\20241229\final_damage_costs_updated.csv"
)

# Select Relevant Columns
data = df[
    ["events", "direct_min", "direct_max", "indirect_15", "indirect_30", "indirect_60"]
]

# Compute direct cost mean and uncertainty
df["direct_mean"] = (df["direct_min"] + df["direct_max"]) / 2
df["direct_uncertainty"] = (
    df["direct_max"] - df["direct_min"]
) / 2  # Half-range as std deviation

# Compute indirect cost mean and uncertainty (mean and variance-based approach)
df["indirect_mean"] = df[["indirect_15", "indirect_30", "indirect_60"]].mean(axis=1)
df["indirect_uncertainty"] = df[["indirect_15", "indirect_30", "indirect_60"]].std(
    axis=1
)  # Standard deviation

# Compute total cost (direct + indirect) for sorting
df["total_cost"] = df["direct_mean"] + df["indirect_mean"]

# Sort DataFrame by total cost in descending order
df = df.sort_values(by="total_cost", ascending=True).reset_index(drop=True)

# Set positions for bars
y = np.arange(len(df["events"]))  # Y-axis positions

# Create Horizontal Bar Plot
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 20  # Title font size
plt.rcParams["axes.labelsize"] = 16  # Axis labels
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

fig, ax = plt.subplots(figsize=(12, 8))

# Plot direct cost with uncertainty
ax.barh(y, df["direct_mean"], color="dodgerblue", label="Direct Cost")
ax.errorbar(
    df["direct_mean"],
    y,
    xerr=df["direct_uncertainty"],
    fmt="o",
    color="black",
    capsize=5,
    label="Uncertainty (Direct)",
)

# Stack indirect costs
left = df["direct_mean"]  # Left position for stacking
ax.barh(y, df["indirect_mean"], left=left, color="lightcoral", label="Indirect Cost")
ax.errorbar(
    left + df["indirect_mean"],
    y,
    xerr=df["indirect_uncertainty"],
    fmt="o",
    color="gray",
    capsize=5,
    label="Uncertainty (Indirect)",
)

# Labels and Formatting
ax.set_yticks(y)
ax.set_yticklabels(df["events"])
ax.set_xlabel("Cost (£Million)")
ax.set_title("Direct and Indirect Damage Costs with Uncertainties", fontweight="bold")
ax.legend(loc="lower right")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(
    r"C:\Oxford\Research\DAFNI\local\papers\figures\direct_indirect_damages.tif",
    dpi=300,
)
plt.show()
