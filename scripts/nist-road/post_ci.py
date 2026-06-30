# %%
# import sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from nird.utils import load_config
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

import seaborn as sns

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["nist_path"])


# %%
def compute_length_weighted_ci(flow):
    flow["length_m"] = flow.geometry.length
    flow["UR"] = flow.acc_flow / (
        flow.acc_flow + flow.acc_capacity
    )  # utilisation ratio
    flow["TL(sec/km)"] = 2236.94 * (
        1.0 / flow.acc_speed - 1.0 / flow.free_flow_speeds
    )  # time loss

    lad = gpd.read_parquet(nist_path / "processed" / "lad24_shp.gpq")
    flow_split = gpd.overlay(flow, lad[["LAD24CD", "geometry"]], how="intersection")
    flow_split["length_m"] = flow_split.geometry.length
    flow_split["length_km"] = flow_split["length_m"] / 1000.0
    flow_split = flow_split[flow_split["length_km"] > 0].copy()
    agg = (
        flow_split.groupby("LAD24CD")
        .apply(
            lambda g: pd.Series(
                {
                    "stress_ratio_lw": (g["stress_ratio"] * g["length_km"]).sum()
                    / g["length_km"].sum(),
                    "speed_loss_lw": (g["speed_loss"] * g["length_km"]).sum()
                    / g["length_km"].sum(),
                    "congestion_score_lw": (
                        g["congestion_score"] * g["length_km"]
                    ).sum()
                    / g["length_km"].sum(),
                }
            )
        )
        .reset_index()
    )
    lad_with_stats = lad.merge(agg, on="LAD24CD", how="left")
    lad_with_stats["stress_ratio_lw"] = lad_with_stats["stress_ratio_lw"].fillna(np.nan)
    lad_with_stats["speed_loss_lw"] = lad_with_stats["speed_loss_lw"].fillna(np.nan)
    lad_with_stats["congestion_score_lw"] = lad_with_stats[
        "congestion_score_lw"
    ].fillna(np.nan)

    return lad_with_stats[
        [
            "LAD24CD",
            "stress_ratio_lw",
            "speed_loss_lw",
            "congestion_score_lw",
            "geometry",
        ]
    ]


def compute_GMM_clusters(df):
    """
    Fit a Gaussian Mixture Model (GMM) on:
        - stress_ratio
        - speed_loss

    Returns
    -------
    result : dict
        {
            "df": input dataframe with added columns,
            "best_k": selected number of clusters,
            "gmm": fitted GaussianMixture model,
            "scaler": fitted StandardScaler,
            "centers": cluster centers in original scale,
            "bic_scores": list of BIC values,
            "threshold_curve": DataFrame with an approximate boundary curve
        }
    """

    required_cols = {"stress_ratio", "speed_loss"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[["stress_ratio", "speed_loss"]].dropna().copy()
    if len(data) < 10:
        raise ValueError("Not enough valid rows to fit GMM.")

    # Standardize features for GMM
    scaler = StandardScaler()
    X_std = scaler.fit_transform(data)

    # Select number of clusters using BIC
    bic_scores = []
    models = []
    max_k = min(6, len(data))  # avoid asking for more clusters than data can support

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=42, n_init=10
        )
        gmm.fit(X_std)
        bic_scores.append(gmm.bic(X_std))
        models.append(gmm)

    best_k = int(np.argmin(bic_scores) + 1)
    best_gmm = models[best_k - 1]

    # Cluster labels and probabilities
    labels = best_gmm.predict(X_std)
    probs = best_gmm.predict_proba(X_std)

    out = df.copy()
    out.loc[data.index, "gmm_cluster"] = labels

    for i in range(best_k):
        out.loc[data.index, f"gmm_p_{i}"] = probs[:, i]

    out["gmm_cluster"] = out["gmm_cluster"].astype("Int64")

    # Cluster centers in original scale
    centers_std = best_gmm.means_
    centers = scaler.inverse_transform(centers_std)
    centers_df = pd.DataFrame(
        centers, columns=["stress_ratio_center", "speed_loss_center"]
    )
    centers_df["cluster"] = range(best_k)
    centers_df = centers_df.sort_values(
        ["stress_ratio_center", "speed_loss_center"]
    ).reset_index(drop=True)

    # Approximate a boundary curve between the least and most "congested" regimes
    # (based on higher stress + higher speed loss)
    centers_df["congestion_score"] = centers_df["stress_ratio_center"].rank(
        method="dense"
    ) + centers_df["speed_loss_center"].rank(method="dense")
    low_regime = int(centers_df.sort_values("congestion_score").iloc[0]["cluster"])
    high_regime = int(centers_df.sort_values("congestion_score").iloc[-1]["cluster"])

    x_min, x_max = data["stress_ratio"].min(), data["stress_ratio"].max()
    y_min, y_max = data["speed_loss"].min(), data["speed_loss"].max()

    stress_vals = np.linspace(x_min, x_max, 200)
    boundary_points = []

    for s in stress_vals:
        y_vals = np.linspace(y_min, y_max, 400)
        grid_line = np.column_stack([np.full_like(y_vals, s), y_vals])
        grid_line_std = scaler.transform(grid_line)
        p = best_gmm.predict_proba(grid_line_std)

        diff = p[:, low_regime] - p[:, high_regime]
        sign_change = np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]

        if len(sign_change) > 0:
            j = sign_change[0]
            y0, y1 = y_vals[j], y_vals[j + 1]
            d0, d1 = diff[j], diff[j + 1]
            if d1 != d0:
                y_star = y0 - d0 * (y1 - y0) / (d1 - d0)
                boundary_points.append((s, y_star))

    threshold_curve = pd.DataFrame(
        boundary_points, columns=["stress_ratio", "speed_loss"]
    )

    return {
        "df": out,
        "best_k": best_k,
        "gmm": best_gmm,
        "scaler": scaler,
        "centers": centers_df,
        "bic_scores": bic_scores,
        "threshold_curve": threshold_curve,
    }


# %%
# Visualisation functions for GMM diagnostics
def plot_gmm_joint_diagnostics(results, df=None, sample_size=None):
    """
    Plot joint distribution diagnostics for stress_ratio and speed_loss
    using the fitted GMM.

    Parameters
    ----------
    results : dict
        Output of compute_GMM_clusters(df)
    df : pd.DataFrame, optional
        DataFrame to plot. Defaults to results["df"].
    sample_size : int, optional
        If given, randomly sample rows for plotting to keep plots lighter.
    """

    if df is None:
        df = results["df"]

    gmm = results["gmm"]
    scaler = results["scaler"]
    best_k = results["best_k"]

    data = df[["stress_ratio", "speed_loss", "gmm_cluster"]].dropna().copy()
    data["gmm_cluster"] = data["gmm_cluster"].astype(int)

    if sample_size is not None and len(data) > sample_size:
        data = data.sample(sample_size, random_state=42)

    X = data[["stress_ratio", "speed_loss"]].to_numpy()
    labels = data["gmm_cluster"].to_numpy()

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 250),
        np.linspace(y_min, y_max, 250),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Overall GMM density in original scale
    grid_std = scaler.transform(grid)
    log_density = gmm.score_samples(grid_std)
    density = np.exp(log_density).reshape(xx.shape)

    # -------------------------------------------------------
    # 1) Scatter + mixture density contours
    # -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    sc = ax.scatter(
        X[:, 0], X[:, 1], c=labels, cmap="tab10", s=18, alpha=0.65, edgecolor="none"
    )

    cs = ax.contour(xx, yy, density, levels=8, linewidths=1)
    ax.clabel(cs, inline=True, fontsize=8, fmt="%.2e")

    ax.set_xlabel("stress_ratio")
    ax.set_ylabel("speed_loss")
    ax.set_title("Joint distribution with GMM density contours")
    plt.colorbar(sc, ax=ax, label="Cluster")
    plt.show()

    # -------------------------------------------------------
    # 2) Per-cluster joint KDE plots
    # -------------------------------------------------------
    fig, axes = plt.subplots(1, best_k, figsize=(6 * best_k, 5), squeeze=False)

    for k in range(best_k):
        ax = axes[0, k]
        cluster_data = data[data["gmm_cluster"] == k]

        if len(cluster_data) < 3:
            ax.set_title(f"Cluster {k} (too few points)")
            continue

        sns.kdeplot(
            data=cluster_data,
            x="stress_ratio",
            y="speed_loss",
            fill=True,
            thresh=0.05,
            levels=8,
            cmap="Blues",
            ax=ax,
        )
        ax.scatter(
            cluster_data["stress_ratio"],
            cluster_data["speed_loss"],
            s=10,
            alpha=0.35,
            color="black",
        )
        ax.set_title(f"Cluster {k} joint KDE")
        ax.set_xlabel("stress_ratio")
        ax.set_ylabel("speed_loss")

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------
    # 3) Per-cluster marginal histograms
    # -------------------------------------------------------
    fig, axes = plt.subplots(best_k, 2, figsize=(12, 3.5 * best_k), squeeze=False)

    for k in range(best_k):
        cluster_data = data[data["gmm_cluster"] == k]

        ax1 = axes[k, 0]
        ax2 = axes[k, 1]

        sns.histplot(
            cluster_data["stress_ratio"], kde=True, stat="density", bins=25, ax=ax1
        )
        ax1.set_title(f"Cluster {k} - stress_ratio")
        ax1.set_xlabel("stress_ratio")

        sns.histplot(
            cluster_data["speed_loss"], kde=True, stat="density", bins=25, ax=ax2
        )
        ax2.set_title(f"Cluster {k} - speed_loss")
        ax2.set_xlabel("speed_loss")

    plt.tight_layout()
    plt.show()

    return data


def plot_gmm_component_ellipses(results, df=None):
    if df is None:
        df = results["df"]

    gmm = results["gmm"]
    scaler = results["scaler"]
    best_k = results["best_k"]

    data = df[["stress_ratio", "speed_loss", "gmm_cluster"]].dropna().copy()
    data["gmm_cluster"] = data["gmm_cluster"].astype(int)

    X = data[["stress_ratio", "speed_loss"]].to_numpy()
    labels = data["gmm_cluster"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", s=15, alpha=0.55)

    theta = np.linspace(0, 2 * np.pi, 200)

    for k in range(best_k):
        mean_std = gmm.means_[k]
        cov_std = gmm.covariances_[k]

        # back-transform mean and covariance to original scale
        mean = scaler.inverse_transform(mean_std.reshape(1, -1))[0]
        D = np.diag(scaler.scale_)
        cov = D @ cov_std @ D

        # ellipse from covariance eigenvectors
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        for n_std in [1, 2, 3]:
            ellipse = (
                vecs
                @ np.diag(np.sqrt(vals))
                @ np.vstack([np.cos(theta), np.sin(theta)])
                * n_std
            )
            ax.plot(
                ellipse[0, :] + mean[0],
                ellipse[1, :] + mean[1],
                linewidth=1.5,
                alpha=0.8,
            )

        ax.scatter(mean[0], mean[1], marker="x", s=120, linewidths=3)

    ax.set_xlabel("stress_ratio")
    ax.set_ylabel("speed_loss")
    ax.set_title("GMM component ellipses in original scale")
    plt.show()


# %%
CI2021 = gpd.read_parquet(
    nist_path.parent / "scripts" / "results" / "outputs" / "ci_2050_hhh.gpq"
)
# %%
# compute GMM clusters and add cluster labels to CI2021
results = compute_GMM_clusters(CI2021)
results["df"].head()

# %%
# plot_data = plot_gmm_joint_diagnostics(results, sample_size=5000)
# plot_gmm_component_ellipses(results)

# %%
df = results["df"]
cluster_to_score = results["centers"].set_index("cluster")["congestion_score"].to_dict()
df["congestion_score"] = df["gmm_cluster"].map(cluster_to_score)
# df.to_parquet(
#     nist_path.parent
#     / "scripts"
#     / "results"
#     / "outputs"
#     / "ci_2050_hhh_with_clusters.gpq"
# )

# %%
# compute length-weighted congestion indicators for 2021
lad_ci_2021 = compute_length_weighted_ci(df)
lad_ci_2021.head()

# %%
lad_ci_2021.to_parquet(
    nist_path.parent / "scripts" / "results" / "outputs" / "lad_ci_2050_hhh.gpq"
)
