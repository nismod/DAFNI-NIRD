# %%
# import sys
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
from nird.utils import load_config

# import logging
import warnings

warnings.simplefilter("ignore")
nist_path = Path(load_config()["paths"]["nist_path"])


# %%
def compute_congestion(future_year, future_scenario):
    flow = gpd.read_parquet(
        # nist_path
        # / "incoming"
        # / "20260216 - inputs to OxfUni models"
        # / "outputs"
        # / "roads"
        # / f"edge_flow_{future_year}_{future_scenario}.gpq"
        nist_path.parent
        / "scripts"
        / "results"
        / "outputs"
        / "edge_flow_2030.gpq"
    )
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
                    "UR_length_weighted": (g["UR"] * g["length_km"]).sum()
                    / g["length_km"].sum(),
                    # "UR_max": g["UR"].max(),
                    # "total_km": g["length_km"].sum(),
                    # "pct_saturated_km": 100.0
                    # * g.loc[g["UR"] > 0.85, "length_km"].sum()
                    # / g["length_km"].sum(),
                    "TL_length_weighted": (g["TL(sec/km)"] * g["length_km"]).sum()
                    / g["length_km"].sum(),
                    # "TL_max": g["TL(sec/km)"].max(),
                    # "MPH_length_weighted": (g["acc_speed"] * g["length_km"]).sum()
                    # / g["length_km"].sum(),
                }
            )
        )
        .reset_index()
    )
    lad_with_stats = lad.merge(agg, on="LAD24CD", how="left")
    lad_with_stats["UR_length_weighted"] = lad_with_stats["UR_length_weighted"].fillna(
        np.nan
    )
    # lad_with_stats["UR_max"] = lad_with_stats["UR_max"].fillna(np.nan)
    lad_with_stats["TL_length_weighted"] = lad_with_stats["TL_length_weighted"].fillna(
        np.nan
    )
    # lad_with_stats["MPH_length_weighted"] = lad_with_stats[
    #     "MPH_length_weighted"
    # ].fillna(np.nan)
    # lad_with_stats["total_km"] = lad_with_stats["total_km"].fillna(0.0)
    # lad_with_stats["pct_saturated_km"] = lad_with_stats["pct_saturated_km"].fillna(0.0)

    # for 2030 and 2050, merge with household growth stats
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

    lad_with_stats["URw/HIR"] = (
        lad_with_stats["UR_length_weighted"]
        / lad_with_stats[f"HIR(2021-20{future_year})"]
    )

    return lad_with_stats


def main(future_year, future_scenario):
    lad_with_stats = compute_congestion(future_year, future_scenario)
    # open baseline stats
    baseline = gpd.read_parquet(
        # nist_path
        # / "incoming"
        # / "20260216 - inputs to OxfUni models"
        # / "processed_data"
        # / "lad_stats_2021.gpq"
        nist_path.parent
        / "scripts"
        / "results"
        / "outputs"
        / "lad_stats_2021.gpq"
    )
    lad_with_stats = lad_with_stats.merge(
        baseline[["LAD24CD", "TL_length_weighted_2021", "UR_length_weighted_2021"]],
        on="LAD24CD",
        how="left",
    )
    lad_with_stats["TLwR"] = (
        lad_with_stats["TL_length_weighted"] - lad_with_stats["TL_length_weighted_2021"]
    ) / lad_with_stats["TL_length_weighted_2021"]
    ratio = lad_with_stats["TLwR"] / lad_with_stats[f"HIR(2021-20{future_year})"]

    # housing-normalised time loss ratio
    # with 888 for missing data and 999 for infinite values
    lad_with_stats["TLwR/HIR"] = np.where(
        lad_with_stats["TLwR"].isna(), 888, np.where(np.isinf(ratio), 999, ratio)
    )

    # housing-normalised delay index
    lad_with_stats["TL/TL0"] = (
        lad_with_stats["TL_length_weighted"] / lad_with_stats["TL_length_weighted_2021"]
    )
    lad_with_stats["H/H0"] = (
        lad_with_stats[f"MHCLG_HH_20{future_year}"] / lad_with_stats["VERISK_HH_2021"]
    )
    lad_with_stats["HND"] = np.log(lad_with_stats["TL/TL0"] / lad_with_stats["H/H0"])

    # saturated congestion zones
    lad_with_stats["sc"] = 0
    lad_with_stats.loc[
        (lad_with_stats["HND"] > 0)
        & (lad_with_stats["HND"] < lad_with_stats["HND"].quantile(0.33))  # 1
        & (lad_with_stats["TL_length_weighted"] > 120),
        "sc",
    ] = 1

    # emerging congestion zones
    lad_with_stats["ec"] = 0
    lad_with_stats.loc[
        (np.isinf(lad_with_stats["HND"]) & (lad_with_stats["TL_length_weighted"] > 30)),
        "ec",
    ] = 1

    # housing-normalised congestion index
    lad_with_stats["HNS"] = np.log(
        lad_with_stats["UR_length_weighted"] / lad_with_stats["UR_length_weighted_2021"]
    ) / np.log(
        lad_with_stats[f"MHCLG_HH_20{future_year}"] / lad_with_stats["VERISK_HH_2021"]
    )
    lad_with_stats["ss"] = 0
    lad_with_stats.loc[
        (lad_with_stats["HNS"] > 0)
        & (lad_with_stats["HNS"] < lad_with_stats["HNS"].quantile(0.33))  # 1
        & (lad_with_stats["UR_length_weighted"] > 0.1),
        "ss",
    ] = 1
    lad_with_stats.to_parquet(
        nist_path.parent
        / "scripts"
        / "results"
        / "outputs"
        / f"lad_time_spd_{future_year}_Copy.gpq"
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

    main(future_year=30, future_scenario="hhh")
    # future_year: 30, 50
    # future_scenario: ppp, hhh
