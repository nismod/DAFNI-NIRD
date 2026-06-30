# %%
# import sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from nird.utils import load_config
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# import logging
import warnings

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["nist_path"])


# %%
def compute_length_weighted_congestion(flow):
    """Compute length-weighted congestion metrics for each LAD;
    merged with household growth statistics for future analysis."""

    # Compute length-weighted congestion metrics for each LAD.
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
                    "UR_lw": (g["UR"] * g["length_km"]).sum() / g["length_km"].sum(),
                    "TL_lw": (g["TL(sec/km)"] * g["length_km"]).sum()
                    / g["length_km"].sum(),
                    "MPH_lw": (g["acc_speed"] * g["length_km"]).sum()
                    / g["length_km"].sum(),
                    "CI_lw": (g["congestion_score"] * g["length_km"]).sum()
                    / g["length_km"].sum(),
                }
            )
        )
        .reset_index()
    )
    lad_with_stats = lad.merge(agg, on="LAD24CD", how="left")
    lad_with_stats["UR_lw"] = lad_with_stats["UR_lw"].fillna(np.nan)
    lad_with_stats["TL_lw"] = lad_with_stats["TL_lw"].fillna(np.nan)
    lad_with_stats["MPH_lw"] = lad_with_stats["MPH_lw"].fillna(np.nan)
    lad_with_stats["CI_lw"] = lad_with_stats["CI_lw"].fillna(np.nan)

    # Merge with household growth stats
    hh = gpd.read_parquet(nist_path / "processed" / "lad24_pop_hh.gpq")
    hh["HI(2021-2030)"] = hh["MHCLG_HH_2030"] - hh["VERISK_HH_2021"]
    hh["HI(2021-2050)"] = hh["MHCLG_HH_2050"] - hh["VERISK_HH_2021"]
    hh["HIR(2021-2030)"] = (hh["MHCLG_HH_2030"] - hh["VERISK_HH_2021"]) / hh[
        "VERISK_HH_2021"
    ]
    hh["HIR(2021-2050)"] = (hh["MHCLG_HH_2050"] - hh["VERISK_HH_2021"]) / hh[
        "VERISK_HH_2021"
    ]
    lad_with_stats = lad_with_stats.merge(
        hh[
            [
                "LAD24CD",
                "VERISK_HH_2021",
                "MHCLG_HH_2030",
                "MHCLG_HH_2050",
                "HI(2021-2030)",
                "HI(2021-2050)",
                "HIR(2021-2030)",
                "HIR(2021-2050)",
            ]
        ],
        on="LAD24CD",
        how="left",
    )

    return lad_with_stats


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


def compute_congestion(future_year, future_scenario):
    """Compute congestion metrics for a given future year and scenario, including
    volume-to-capacity ratio, time loss per km, stress ratio, and speed loss
    relative to reference speed."""

    flow = gpd.read_parquet(
        nist_path
        / "incoming"
        / "20260216 - inputs to OxfUni models"
        / "outputs"
        / "roads"
        / f"edge_flow_{future_year}_{future_scenario}.gpq"
    )
    flow["length_m"] = flow.geometry.length

    # volume-to-capacity ratio
    flow["UR"] = flow.acc_flow / (
        flow.acc_flow + flow.acc_capacity
    )  # edge utilisation ratio

    # time loss per km
    flow["TL(sec/km)"] = 2236.94 * (1.0 / flow.acc_speed - 1.0 / flow.free_flow_speeds)

    # stress ratio
    flow["stress_ratio"] = flow.acc_flow / (flow.acc_capacity + flow.acc_flow)

    # speed loss relative to reference speed
    flow["speed_loss"] = 1 - (flow.acc_speed / flow.initial_flow_speeds)

    return flow


# %%
def main(future_year, future_scenario):
    # compute edge-level congestion metrics
    flow = compute_congestion(future_year, future_scenario)
    # classify edges into congestion regimes using GMM clustering
    gmm_results = compute_GMM_clusters(flow)
    gmm_df = gmm_results["df"]
    cluster_to_score = (
        gmm_results["centers"][["cluster", "congestion_score"]]
        .set_index("cluster")["congestion_score"]
        .to_dict()
    )
    gmm_df["congestion_score"] = gmm_df["gmm_cluster"].map(cluster_to_score)

    # compute length-weighted congestion metrics for each LAD
    lad_with_stats = compute_length_weighted_congestion(gmm_df)

    # open baseline stats
    baseline = gpd.read_parquet(
        nist_path
        / "incoming"
        / "20260216 - inputs to OxfUni models"
        / "processed_data"
        / "lad_stats_2021.gpq"
    )

    lad_with_stats = lad_with_stats.merge(
        baseline[
            [
                "LAD24CD",
                "TL_lw_2021",
                "UR_lw_2021",
                "stress_ratio_lw_2021",
                "speed_loss_lw_2021",
                "congestion_score_lw_2021",
            ]
        ],
        on="LAD24CD",
        how="left",
    )

    # housing-normalised saturation index
    lad_with_stats["HNS"] = np.log(
        lad_with_stats["UR_lw"] / lad_with_stats["UR_lw_2021"]
    ) / np.log(
        lad_with_stats[f"MHCLG_HH_20{future_year}"] / lad_with_stats["VERISK_HH_2021"]
    )

    # # saturated capacity stress zones (? stress-threshold: vc ration > 0.1 )
    # lad_with_stats["ss"] = 0
    # lad_with_stats.loc[
    #     (lad_with_stats["HNS"] > 0)
    #     & (lad_with_stats["HNS"] < lad_with_stats["HNS"].quantile(0.33))  # 1
    #     & (lad_with_stats["UR_length_weighted"] > 0.1),
    #     "ss",
    # ] = 1

    # housing-normalised time loss ratio
    # with 888 for missing data and 999 for infinite values
    # lad_with_stats["TLwR"] = (
    #     lad_with_stats["TL_lw"] - lad_with_stats["TL_lw_2021"]
    # ) / lad_with_stats["TL_lw_2021"]

    # ratio = lad_with_stats["TLwR"] / lad_with_stats[f"HIR(2021-20{future_year})"]
    # lad_with_stats["TLwR/HIR"] = np.where(
    #     lad_with_stats["TLwR"].isna(), 888, np.where(np.isinf(ratio), 999, ratio)
    # )

    # housing-normalised delay index
    # lad_with_stats["TL/TL0"] = lad_with_stats["TL_lw"] / lad_with_stats["TL_lw_2021"]
    # lad_with_stats["H/H0"] = (
    #    lad_with_stats[f"MHCLG_HH_20{future_year}"] / lad_with_stats["VERISK_HH_2021"]
    # )
    lad_with_stats["HND"] = np.log(
        lad_with_stats["TL_lw"] / lad_with_stats["TL_lw_2021"]
    ) / np.log(
        lad_with_stats[f"MHCLG_HH_20{future_year}"] / lad_with_stats["VERISK_HH_2021"]
    )

    # # saturated congestion zones
    # lad_with_stats["sc"] = 0
    # lad_with_stats.loc[
    #     (lad_with_stats["HND"] > 0)
    #     & (lad_with_stats["HND"] < lad_with_stats["HND"].quantile(0.33))  # 1
    #     & (lad_with_stats["TL_length_weighted"] > 120),
    #     "sc",
    # ] = 1

    # # emerging congestion zones
    # lad_with_stats["ec"] = 0
    # lad_with_stats.loc[
    #     (np.isinf(lad_with_stats["HND"]) & (lad_with_stats["TL_length_weighted"] > 30)),
    #     "ec",
    # ] = 1

    # housing-normalised congestion index
    lad_with_stats["HNC"] = np.log(
        lad_with_stats["CI_lw"] / lad_with_stats["congestion_score_lw_2021"]
    ) / np.log(
        lad_with_stats[f"MHCLG_HH_20{future_year}"] / lad_with_stats["VERISK_HH_2021"]
    )

    # add labels
    lad_with_stats["congestion_label"] = "transitional"
    lad_with_stats.loc[
        (lad_with_stats["HNS"] >= 1) & (lad_with_stats["HND"] < 1), "congestion_label"
    ] = "pressure building"

    lad_with_stats.loc[
        (lad_with_stats["HNS"] >= 1)  # stress
        & (lad_with_stats["HNC"] >= 1)  # congestion
        & (lad_with_stats["HND"] >= 1),  # delay
        "congestion_label",
    ] = "saturated congestion"

    lad_with_stats.loc[np.isinf(lad_with_stats["HND"]), "congestion_label"] = (
        "emerging congestion"
    )
    lad_with_stats.to_parquet(
        nist_path.parent
        / "final"
        / f"lad_stats_{future_year}_{future_scenario}.gpq"
        # nist_path
        # / "incoming"
        # / "20260216 - inputs to OxfUni models"
        # / "outputs"
        # / "roads"
        # / f"lad_time_spd_{future_year}_{future_scenario}_Copy.gpq"
    )


if __name__ == "__main__":
    # logging.basicConfig(
    #     format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    # )
    # try:
    #     future_year, future_scenario = sys.argv[1:]
    #     main(int(future_year), future_scenario)
    # except (IndexError, NameError):
    #     logging.info("Provide input parameters!")

    main(future_year=50, future_scenario="hhh")
    # future_year: 30, 50
    # future_scenario: ppp, hhh
