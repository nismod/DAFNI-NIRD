import matplotlib.pyplot as plt
import pandas as pd
import warnings
from pathlib import Path

from nird.utils import load_config

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])

gp = pd.read_csv(
    base_path.parent / "outputs" / "base_scenario" / "origin_destination_flow_stats.csv"
)

# %%
# Set global font properties using plt.rcParams
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12  # Set default font size
plt.rcParams["axes.titlesize"] = 20  # Title font size
plt.rcParams["axes.labelsize"] = 16  # Axis labels font size
plt.rcParams["xtick.labelsize"] = 12  # x-axis ticks font size
plt.rcParams["ytick.labelsize"] = 12  # y-axis ticks font size

# Create the bar chart with distinct colors and clear labels
plt.figure(figsize=(12, 8))  # Adjust the figure size for better visibility
plt.bar(
    gp["region"],
    gp["origin_of_flows"],
    color="skyblue",
    width=0.4,
    label="From origins",
    align="center",
)
plt.bar(
    gp["region"],
    gp["destination_of_flows"],
    color="red",
    width=0.4,
    label="To destinations",
    align="edge",
)

# Add labels and title
plt.xlabel("Regions", fontweight="bold")  # Label for x-axis
plt.ylabel("Total Flows", fontweight="bold")  # Label for y-axis
plt.title("Flow Distribution by Region", fontweight="bold")  # Title for the plot

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha="right")

# Add a legend
plt.legend(fontsize=12)

# Display the plot
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()  # Ensure everything fits well in the figure
plt.savefig(
    base_path.parent / "papers" / "figures" / "Fig2.flow_distribution_by_region.tif",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
