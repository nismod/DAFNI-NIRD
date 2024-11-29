# %%
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd

from snail import damages
from nird.utils import load_config

from collections import defaultdict
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])
raster_path = Path(load_config()["paths"]["JBA_data"])


# %%
def create_damage_curves(damage_ratio_df):
    list_of_damage_curves = []
    cols = damage_ratio_df.columns[1:]
    for col in cols:
        damage_ratios = damage_ratio_df[["intensity", col]]
        damage_ratios.rename(columns={col: "damage"}, inplace=True)
        damage_curve = damages.PiecewiseLinearDamageCurve(damage_ratios)
        list_of_damage_curves.append(damage_curve)

    damage_curve_dict = defaultdict()

    """
    MT: motorways and trunk roads
    O: other roads
    SA: sophisticated accessories
    NSA: non-sophisticated accessories
    HF: high flow
    LF: low flow
    """
    keys = ["MT_SA_LF", "MT_SA_HF", "MT_NSA_LF", "MT_NSA_HF", "O_LF", "O_HF"]
    for idx in range(len(keys)):
        key = keys[idx]
        damage_curve = list_of_damage_curves[idx]
        damage_curve_dict[key] = damage_curve

    return damage_curve_dict


def compute_damage_fractions(
    intersections_M,
    intersections_A,
    intersections_B,
    MA_curves,
    B_curves,
    raster_key,
    damage_curves,
):
    print("Computing damage fractions...")
    # motorways
    for curve in MA_curves:
        intersections_M[f"{raster_key}_{curve}_damage_fraction"] = damage_curves[
            curve
        ].damage_fraction(intersections_M[f"{raster_key}_depth"])

    # a/trunk roads:
    for curve in MA_curves:
        intersections_A[f"{raster_key}_{curve}_damage_fraction"] = damage_curves[
            curve
        ].damage_fraction(intersections_A[f"{raster_key}_depth"])

    # b/others roads
    for curve in B_curves:
        intersections_B[f"{raster_key}_{curve}_damage_fraction"] = damage_curves[
            curve
        ].damage_fraction(intersections_B[f"{raster_key}_depth"])

    return intersections_M, intersections_A, intersections_B


def compute_damage_values(
    intersections_M,
    intersections_A,
    intersections_B,
    MA_curves,
    B_curves,
    raster_key,
    cost_dict,
):
    print("Compute damage values...")
    # Motorway and A roads: 4 damage curves for M & A roads
    for case in MA_curves:
        intersections_M[f"{raster_key}_{case}_damage_value_L"] = (
            intersections_M.geometry.length
            * cost_dict["ML"]
            * intersections_M[f"{raster_key}_{case}_damage_fraction"]
        )
        intersections_M[f"{raster_key}_{case}_damage_value_U"] = (
            intersections_M.geometry.length
            * cost_dict["MU"]
            * intersections_M[f"{raster_key}_{case}_damage_fraction"]
        )
        intersections_A[f"{raster_key}_{case}_damage_value_L"] = (
            intersections_A.geometry.length
            * cost_dict["AL"]
            * intersections_A[f"{raster_key}_{case}_damage_fraction"]
        )
        intersections_A[f"{raster_key}_{case}_damage_value_U"] = (
            intersections_A.geometry.length
            * cost_dict["AU"]
            * intersections_A[f"{raster_key}_{case}_damage_fraction"]
        )

    # B Roads: 2 damage curves for B roads
    for case in B_curves:
        intersections_B[f"{raster_key}_{case}_damage_value_L"] = (
            intersections_B.geometry.length
            * cost_dict["BL"]
            * intersections_B[f"{raster_key}_{case}_damage_fraction"]
        )
        intersections_B[f"{raster_key}_{case}_damage_value_U"] = (
            intersections_B.geometry.length
            * cost_dict["BU"]
            * intersections_B[f"{raster_key}_{case}_damage_fraction"]
        )

    return intersections_M, intersections_A, intersections_B


def calculate_damages(intersections, raster_key, damage_curves, cost_dict):
    curves = ["MT_SA_LF", "MT_SA_HF", "MT_NSA_LF", "MT_NSA_HF", "O_LF", "O_HF"]
    MA_curves = ["MT_SA_LF", "MT_SA_HF", "MT_NSA_LF", "MT_NSA_HF"]
    B_curves = ["O_LF", "O_HF"]

    # append columns
    for col in curves:
        intersections[f"{raster_key}_{col}_damage_fraction"] = np.nan
    for col in curves:
        intersections[f"{raster_key}_{col}_damage_value_L"] = np.nan
    for col in curves:
        intersections[f"{raster_key}_{col}_damage_value_U"] = np.nan

    # extract rows
    intersections_M = intersections[intersections.road_classification == "Motorway"]
    intersections_A = intersections[intersections.road_classification == "A Road"]
    intersections_B = intersections[intersections.road_classification == "B Road"]

    # compute damage fractions
    intersections_M, intersections_A, intersections_B = compute_damage_fractions(
        intersections_M,
        intersections_A,
        intersections_B,
        MA_curves,
        B_curves,
        raster_key,
        damage_curves,
    )

    # compute damage costs
    intersections_M, intersections_A, intersections_B = compute_damage_values(
        intersections_M,
        intersections_A,
        intersections_B,
        MA_curves,
        B_curves,
        raster_key,
        cost_dict,
    )

    intersections = pd.concat(
        [intersections_M, intersections_A, intersections_B], axis=0, ignore_index=True
    )

    # calculate min and max damage values
    intersections[f"{raster_key}_min_damage_values"] = np.nan
    intersections[f"{raster_key}_max_damage_values"] = np.nan
    damage_value_columns_MA = [
        col
        for col in intersections.columns
        if ("damage_value" in col) & (f"{raster_key}_M" in col)
    ]
    intersections.loc[
        (intersections.road_classification == "Motorway")
        | (intersections.road_classification == "A Road"),
        f"{raster_key}_min_damage_values",
    ] = intersections[damage_value_columns_MA].min(axis=1)

    intersections.loc[
        (intersections.road_classification == "Motorway")
        | (intersections.road_classification == "A Road"),
        f"{raster_key}_max_damage_values",
    ] = intersections[damage_value_columns_MA].max(axis=1)

    damage_value_columns_B = [
        col
        for col in intersections.columns
        if ("damage_value" in col) & (f"{raster_key}_Other" in col)
    ]
    intersections.loc[
        intersections.road_classification == "B Road", f"{raster_key}_min_damage_values"
    ] = intersections[damage_value_columns_B].min(axis=1)
    intersections.loc[
        intersections.road_classification == "B Road", f"{raster_key}_max_damage_values"
    ] = intersections[damage_value_columns_B].max(axis=1)

    # extract partial datasets
    columns_to_extract = ["id", "road_classification", f"{raster_key}_depth"]
    columns_to_extract.extend(
        [col for col in intersections.columns if "damage_fraction" in col]
    )
    columns_to_extract.extend(
        [col for col in intersections.columns if "damage_value" in col]
    )
    intersections = intersections[columns_to_extract]
    print("The damage calculation is completed!")

    return intersections


def main():
    # damage curves
    """
    4 damage curvse for M and A roads
    2 damage curvse for B roads
    """
    damages_ratio_df = pd.read_excel(
        base_path / "disruption_analysis_1129" / "damage_ratio_road_flood.xlsx"
    )
    damage_curve_dict = create_damage_curves(damages_ratio_df)

    # damage costs (updated with UK values)
    """
    2 damage values (lower bound and upper bound) for M, A, B roads
    """
    damage_cost_dict = {
        "ML": 1500,
        "MU": 4000,
        "AL": 800,
        "AU": 2400,
        "BL": 400,
        "BU": 1200,
    }
    flood_key = "UK_2007_May_FLSW"
    # features
    road_links = gpd.read_parquet(
        base_path / "disruption_analysis_1129" / "road_links.pq"
    )
    disrupted_links = road_links[road_links[f"{flood_key}_depth"] > 0].reset_index(
        drop=True
    )

    disrupted_links_with_damage = calculate_damages(
        disrupted_links, flood_key, damage_curve_dict, damage_cost_dict
    )
    disrupted_links_with_damage.to_csv(
        base_path.parent / "outputs" / "disrupted_links_with_damage_values.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
