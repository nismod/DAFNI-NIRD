# %%
from pathlib import Path
import pandas as pd
import numpy as np

import geopandas as gpd
import matplotlib.pyplot as plt
import pwlf

outpath = Path(r"C:\Oxford\Research\NIST\papers\Figures")
base_path = Path(r"C:\Oxford\Research\NIST\DfT Model\processed_data\simulation")

# %%
df_2021 = gpd.read_parquet(base_path / "lad_time_spd_baseline.gpq")
df_2050 = gpd.read_parquet(base_path / "lad_time_spd_od_agg_2050.gpq")


# %%
# -----------------------------
# 1. Prepare data
# -----------------------------
data = df_2050[["UR_lw_2021", "MPH_lw_2021"]].dropna().copy()
data = data[(data["UR_lw_2021"] > 0) & (data["MPH_lw_2021"] > 0)]

# -----------------------------
# 2. Bin capacity and compute medians
# -----------------------------
n_bins = 30
data["bin"] = pd.qcut(data["UR_lw_2021"], q=n_bins, duplicates="drop")

summary = (
    data.groupby("bin", observed=True)
    .agg(
        capacity=("UR_lw_2021", "median"),
        speed=("MPH_lw_2021", "median"),
        count=("MPH_lw_2021", "size"),
    )
    .reset_index(drop=True)
    .sort_values("capacity")
    .reset_index(drop=True)
)

x = summary["capacity"].values
y = summary["speed"].values

# -----------------------------
# 3. Fit piecewise linear regression
#    Use 3 segments by default
# -----------------------------
my_pwlf = pwlf.PiecewiseLinFit(x, y)

n_segments = 3  # change to 2 if you want a simpler model
breaks = my_pwlf.fit(n_segments)

# Predict fitted values on a fine grid
xx = np.linspace(x.min(), x.max(), 500)
yy = my_pwlf.predict(xx)

# -----------------------------
# 4. Extract segment slopes
# -----------------------------
slopes = my_pwlf.slopes
breakpoints = breaks  # includes endpoints

print("Breakpoints:")
for i, b in enumerate(breakpoints):
    print(f"  b{i}: {b:.4f}")

print("\nSegment slopes:")
for i, s in enumerate(slopes, start=1):
    print(f"  Segment {i}: slope = {s:.4f}")

# Steepest decreasing segment = most negative slope
steepest_idx = np.argmin(slopes)
steepest_slope = slopes[steepest_idx]

left = breakpoints[steepest_idx]
right = breakpoints[steepest_idx + 1]

print(f"\nSteepest decline is in segment {steepest_idx + 1}:")
print(f"  Capacity range: {left:.4f} to {right:.4f}")
print(f"  Slope: {steepest_slope:.4f} mph per unit capacity")

# -----------------------------
# 5. Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(
    summary["capacity"], summary["speed"], color="red", s=45, label="Binned medians"
)

ax.plot(xx, yy, color="blue", lw=2.5, label=f"Piecewise fit ({n_segments} segments)")

# Shade steepest segment
# ax.axvspan(left, right, color="orange", alpha=0.25, label="Steepest decline segment")

# Draw breakpoint lines
for b in breakpoints[1:-1]:
    ax.axvline(b, color="black", linestyle="--", alpha=0.8)

ax.set_xlabel("Capacity utilisation")
ax.set_ylabel("Median speed (mph)")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# %%
# --------------------------------------------------
# Settings
# --------------------------------------------------
x_col = "UR_lw_2021"
y_col = "MPH_lw_2021"
x_col_future = "UR_lw"
y_col_future = "MPH_lw"

n_bins = 30
n_segments = 3  # uncongested / transition / congested

plt.rcParams["font.family"] = "Times New Roman"


# --------------------------------------------------
# Helper: fit piecewise model on binned medians
# --------------------------------------------------
def fit_piecewise_on_medians(df, x_col, y_col, n_bins=30, n_segments=3):
    data = df[[x_col, y_col]].dropna().copy()
    data = data[(data[x_col] > 0) & (data[y_col] > 0)]

    # Bin x and compute medians in each bin
    data["bin"] = pd.qcut(data[x_col], q=n_bins, duplicates="drop")

    summary = (
        data.groupby("bin", observed=True)
        .agg(capacity=(x_col, "median"), speed=(y_col, "median"), count=(y_col, "size"))
        .reset_index(drop=True)
        .sort_values("capacity")
        .reset_index(drop=True)
    )

    x = summary["capacity"].values
    y = summary["speed"].values

    # Piecewise linear fit
    model = pwlf.PiecewiseLinFit(x, y)
    breaks = model.fit(n_segments)  # includes endpoints

    return data, summary, model, breaks


# --------------------------------------------------
# Helper: classify by baseline breakpoints
# --------------------------------------------------
def classify_capacity(x, b1, b2):
    return np.select(
        [x <= b1, (x > b1) & (x <= b2), x > b2],
        ["uncongested", "transition", "congested"],
        default=np.array(np.nan, dtype=object),
    )


# --------------------------------------------------
# Fit baseline model
# --------------------------------------------------
baseline_data, baseline_summary, baseline_model, breaks = fit_piecewise_on_medians(
    df_2021, x_col, y_col, n_bins=n_bins, n_segments=n_segments
)

# Breakpoints: [xmin, b1, b2, xmax]
b1, b2 = breaks[1], breaks[2]

print("Baseline breakpoints:")
print(f"  uncongested -> transition: {b1:.4f}")
print(f"  transition -> congested:  {b2:.4f}")

# Fitted line for plotting
xx = np.linspace(
    baseline_summary["capacity"].min(), baseline_summary["capacity"].max(), 500
)
yy = baseline_model.predict(xx)

# --------------------------------------------------
# Classify baseline and future scenario using baseline breakpoints
# --------------------------------------------------
baseline_data = baseline_data.copy()
baseline_data["regime"] = classify_capacity(baseline_data[x_col].values, b1, b2)

future_data = df_2050[[x_col_future, y_col_future]].dropna().copy()
future_data = future_data[
    (future_data[x_col_future] > 0) & (future_data[y_col_future] > 0)
]
future_data["regime"] = classify_capacity(future_data[x_col_future].values, b1, b2)

# Color map
colors = {
    "uncongested": "#2ca02c",  # green
    "transition": "#ff7f0e",  # orange
    "congested": "#d62728",  # red
}

# --------------------------------------------------
# Plot: 2 x 2 figure
# --------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey=False)

# (1) Baseline scatter plot
ax = axes[0, 0]
ax.scatter(
    baseline_data[x_col], baseline_data[y_col], s=18, alpha=0.5, color="steelblue"
)
ax.set_title("(1) Baseline scatter")
ax.set_xlabel("Capacity utilisation")
ax.set_ylabel("Speed (mph)")
ax.grid(alpha=0.2)

# (2) Piecewise fit with original dots in light grey
ax = axes[0, 1]
ax.scatter(
    baseline_data[x_col],
    baseline_data[y_col],
    s=16,
    alpha=0.25,
    color="lightgrey",
    label="Original dots",
)
ax.scatter(
    baseline_summary["capacity"],
    baseline_summary["speed"],
    s=45,
    color="red",
    label="Binned medians",
)
ax.plot(xx, yy, color="blue", lw=2.5, label="Piecewise fit")

ax.axvline(b1, color="black", linestyle="--", alpha=0.8)
ax.axvline(b2, color="black", linestyle="--", alpha=0.8)

ax.axvspan(
    baseline_summary["capacity"].min(), b1, color=colors["uncongested"], alpha=0.08
)
ax.axvspan(b1, b2, color=colors["transition"], alpha=0.08)
ax.axvspan(
    b2, baseline_summary["capacity"].max(), color=colors["congested"], alpha=0.08
)

ax.set_title("(2) Baseline piecewise fit")
ax.set_xlabel("Capacity utilisation")
ax.set_ylabel("Speed (mph)")
ax.legend()
ax.grid(alpha=0.2)

# (3) Baseline classified by baseline ranges
ax = axes[1, 0]
for regime in ["uncongested", "transition", "congested"]:
    sub = baseline_data[baseline_data["regime"] == regime]
    ax.scatter(
        sub[x_col], sub[y_col], s=18, alpha=0.75, color=colors[regime], label=regime
    )

ax.axvline(b1, color="black", linestyle="--", alpha=0.8)
ax.axvline(b2, color="black", linestyle="--", alpha=0.8)

ax.set_title("(3) Baseline classified by fitted ranges")
ax.set_xlabel("Capacity utilisation")
ax.set_ylabel("Speed (mph)")
ax.legend()
ax.grid(alpha=0.2)

# (4) Future scenario classified using the SAME baseline ranges
ax = axes[1, 1]
for regime in ["uncongested", "transition", "congested"]:
    sub = future_data[future_data["regime"] == regime]
    ax.scatter(
        sub[x_col_future],
        sub[y_col_future],
        s=18,
        alpha=0.75,
        color=colors[regime],
        label=regime,
    )

ax.axvline(b1, color="black", linestyle="--", alpha=0.8)
ax.axvline(b2, color="black", linestyle="--", alpha=0.8)

ax.set_title("(4) Future scenario classified by baseline ranges")
ax.set_xlabel("Capacity utilisation")
ax.set_ylabel("Speed (mph)")
ax.legend()
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(outpath / "operational regime congestion.png", dpi=300)

plt.show()
