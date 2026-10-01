# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.size"] = 18
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.labelsize"] = 18
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
plt.rcParams["legend.fontsize"] = 18

# %%
# Load data
# df = pd.read_csv(r"C:\Oxford\Research\NIST\local\final\flow_validation.csv")
df = pd.read_csv(
    r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation\flow_validation.csv"
)
df = df.rename(columns={"means": "observation", "acc_flow": "simulation"})

# Determine global range for identical bin edges
min_val = min(df["observation"].min(), df["simulation"].min())
max_val = max(df["observation"].max(), df["simulation"].max())

# %%
# Create 52 shared bins (using 53 edges)
shared_bins = np.linspace(min_val, max_val, 53)

# Plotting
plt.figure(figsize=(8, 8))

# Overlayed histograms with identical bins
plt.hist(
    df["observation"],
    bins=shared_bins,
    alpha=0.5,
    label="Observation (AADT)",
    color="blue",
    edgecolor="white",
    linewidth=0.5,
)
plt.hist(
    df["simulation"],
    bins=shared_bins,
    alpha=0.5,
    label="Simulation",
    color="red",
    edgecolor="white",
    linewidth=0.5,
)

plt.xlabel("Average daily trips (all purposes)")
plt.ylabel("Frequency")
# plt.title("Comparable Histogram: AADT vs Simulation")
plt.legend()
plt.grid(axis="y", alpha=0.3)

# Save the plot
# plt.savefig(r"C:\Oxford\Research\NIST\papers\Figures\validation_histogram.png")
# plt.close()
plt.show()
# Save the processed data
# df.to_csv("final_flow_validation_shared_bins.csv", index=False)

# %%
# Calculate regression
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df["observation"], df["simulation"]
)
r_squared = r_value**2

# Create regression line points
x_range = np.linspace(df["observation"].min(), df["observation"].max(), 100)
y_reg = slope * x_range + intercept

# Plotting
plt.figure(figsize=(8, 8))

# Scatter plot
plt.scatter(
    df["observation"],
    df["simulation"],
    alpha=0.3,
    s=10,
    color="teal",
    label="Data points",
)

# 1:1 Line
max_val = max(df["observation"].max(), df["simulation"].max())
plt.plot(
    [0, max_val],
    [0, max_val],
    color="black",
    linestyle="--",
    linewidth=2,
    label="1:1 Line (Ideal)",
)

# Regression Line
plt.plot(
    x_range,
    y_reg,
    color="darkorange",
    linewidth=3,
    label="Regression Line (Actual Fit)",
)

# Equation and R^2 text
equation_text = f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_squared:.3f}"
plt.text(
    0.05,
    0.95,
    equation_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.xlabel("Observation")
plt.ylabel("Simulation")
plt.title("Scatter Plot with Regression Analysis")
# plt.legend("upper right")
plt.grid(True, linestyle=":", alpha=0.6)
plt.gca().set_aspect("equal", adjustable="box")

# Save the plot
# plt.savefig(r"C:\Oxford\Research\NIST\local\final\flow_regression_analysis.png")
# plt.close()
plt.show()

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_squared}")
