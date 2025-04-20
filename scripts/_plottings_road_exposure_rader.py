# %%
import numpy as np
import matplotlib.pyplot as plt

# Re-define the datasets (only mean values)
datasets = [
    (
        "Road Classification",
        ["A Roads", "B Roads", "Motorway"],
        [23976.01, 17185.24, 6235.46],
    ),
    (
        "Form of Way",
        ["Collapsed Dual", "Dual", "Roundabout", "Single", "Slip Roads"],
        [6804.1, 444.88, 301.6, 39043.11, 803.01],
    ),
    ("Road Label", ["Ordinary", "Bridge", "Tunnel"], [45319.68, 2077.03, 0.00]),
    ("Trunk Road", ["Trunk", "Non-trunk"], [10406.51, 36990.20]),
    ("Urban Road", ["Urban", "Suburban"], [5963.3, 41433.41]),
]

# %%
# Pre-define font styles
plt.rcParams["font.size"] = 14
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.titlesize"] = 26  # Title font size
plt.rcParams["axes.labelsize"] = 14  # Axis labels
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14

# Create a figure with a 2x3 layout
fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(polar=True))

# Add a main title for the figure
fig.suptitle(
    "Comparison of Road Length Exposure (in meters) Across Categories",
    fontsize=24,
    fontweight="bold",
    y=1.02,
)

# Flatten axes for easy iteration
axes = axes.flatten()

# Define color for mean values
color = "blue"
label = "Mean"

# Iterate over each dataset and create a radar chart
for i, (title, categories, values) in enumerate(datasets):
    ax = axes[i]
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the radar chart loop

    # Close the loop for radar plot
    val = values + [values[0]]
    ax.plot(angles, val, color=color, linewidth=2, label=label)
    ax.fill(angles, val, color=color, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)

# Remove the empty subplot (since we have 5 datasets but 6 slots in 2x3 layout)
fig.delaxes(axes[-1])

# Add a legend
# handles, labels = axes[0].get_legend_handles_labels()  # Retrieve from first subplot
# fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.05, 1))

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
