# %%
import os
import warnings
from pathlib import Path

from snail import io, intersection, damages
import pandas as pd
import numpy as np
import geopandas as gpd
from nird.utils import load_config

from collections import defaultdict

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["damage_analysis"])
raster_path = Path(load_config()["paths"]["JBA_data"])
raster_path = raster_path / "12-14_2007 Summer_UK Floods"


# %%
def calculate_damages(raster_path, raster_key, features, damage_curves, cost_dict):
    print(f"Intersecting features with raster {raster_key}...")
    # read the raster data: depth - meter
    raster = io.read_raster_band_data(raster_path)

    # run the intersection analysis
    grid, _ = io.read_raster_metadata(raster_path)
    prepared = intersection.prepare_linestrings(features)
    intersections = intersection.split_linestrings(prepared, grid)
    intersections = intersection.apply_indices(intersections, grid)

    intersections[f"{raster_key}_depth"] = intersection.get_raster_values_for_splits(
        intersections, raster
    )
    # reproject back
    intersections = intersections.to_crs("epsg:27700")

    # calculate damage fractions
    print("Computing direct damages...")
    # create damage curves (Motorways - 4 curves; others - 2 curves)
    curves = ["M_SA_LF", "M_SA_HF", "M_NSA_LF", "M_NSA_HF", "Other_LF", "Other_HF"]
    for col in curves:
        intersections[f"{raster_key}_{col}_damage_fraction"] = np.nan
    for col in curves:
        intersections[f"{raster_key}_{col}_damage_value_L"] = np.nan
    for col in curves:
        intersections[f"{raster_key}_{col}_damage_value_U"] = np.nan

    # Motorways and A roads
    MA_curves = ["M_SA_LF", "M_SA_HF", "M_NSA_LF", "M_NSA_HF"]
    intersections_M = intersections[intersections.road_classification == "Motorway"]
    for curve in MA_curves:
        intersections_M[f"{raster_key}_{curve}_damage_fraction"] = damage_curves[
            curve
        ].damage_fraction(intersections_M[f"{raster_key}_depth"])

    intersections_A = intersections[intersections.road_classification == "A Road"]
    for curve in MA_curves:
        intersections_A[f"{raster_key}_{curve}_damage_fraction"] = damage_curves[
            curve
        ].damage_fraction(intersections_A[f"{raster_key}_depth"])

    # B Roads
    B_curves = ["Other_LF", "Other_HF"]
    intersections_B = intersections[intersections.road_classification == "B Road"]
    for curve in B_curves:
        intersections_B[f"{raster_key}_{curve}_damage_fraction"] = damage_curves[
            curve
        ].damage_fraction(intersections_B[f"{raster_key}_depth"])

    # calculate cost of damage
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

    # combine results
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
        [col for col in intersections.columns if "damage_value" in col]
    )
    intersections = intersections[columns_to_extract]
    print("The damage calculation is completed!")
    return intersections


def create_damage_curves(damage_ratio_df):
    list_of_damage_curves = []
    cols = damage_ratio_df.columns[1:]
    for col in cols:
        damage_ratios = damage_ratio_df[["intensity", col]]
        damage_ratios.rename(columns={col: "damage"}, inplace=True)
        damage_curve = damages.PiecewiseLinearDamageCurve(damage_ratios)
        list_of_damage_curves.append(damage_curve)

    damage_curve_dict = defaultdict()
    keys = ["M_SA_LF", "M_SA_HF", "M_NSA_LF", "M_NSA_HF", "Other_LF", "Other_HF"]
    for idx in range(len(keys)):
        key = keys[idx]
        damage_curve = list_of_damage_curves[idx]
        damage_curve_dict[key] = damage_curve

    return damage_curve_dict


def clip_features(features, clip_path, raster_key):
    print(f"Clipping features based on {raster_key}...")
    clips = gpd.read_file(clip_path, engine="pyogrio")
    if clips.crs != "epsg:4326":
        clips = clips.to_crs("epsg:4326")
    # clipped_features = gpd.overlay(features, clips, how="intersection")
    assert (
        clips.crs == features.crs
    ), "CRS mismatch! Ensure both layers use the same CRS."
    clipped_features = gpd.sjoin(features, clips, how="inner", op="intersects")
    clipped_features = clipped_features[features.columns]
    clipped_features.reset_index(drop=True, inplace=True)
    return clipped_features


def main():
    # damage curves
    damage_ratio_df = pd.read_excel(
        base_path / "inputs" / "damage_ratio_road_flood.xlsx"
    )
    damage_curve_dict = create_damage_curves(damage_ratio_df)

    # damage costs
    damage_cost_dict = {
        "ML": 3470.158,
        "MU": 34701.582,
        "AL": 991.474,
        "AU": 2974.421,
        "BL": 495.737,
        "BU": 1487.211,
    }
    # read feature data
    road_links = gpd.read_parquet(base_path / "inputs" / "road_link_file.geoparquet")
    road_links = road_links.to_crs("epsg:4326")

    # classify flood events
    raster_path_with_vector = []
    raster_path_without_vector = []
    for root, dirs, _ in os.walk(raster_path):
        # check if both "Raster" and "Vector" directories exist
        if "Raster" in dirs:
            raster_dir = os.path.join(root, "Raster")
            vector_file_list = []
            # if "Vector" exists, collect vector files
            if "Vector" in dirs:
                vector_dir = os.path.join(root, "Vector")
                vector_file_list = [
                    f for f in os.listdir(vector_dir) if f.endswith(".shp")
                ]
            # process raster files in the "Raster" directory
            for raster_root, _, raster_files in os.walk(raster_dir):
                for raster_file in raster_files:
                    if (
                        raster_file.endswith(".tif")
                        and "RD" in raster_file  # RD: Raster Depth
                        and "IE" not in raster_file  # IE: Ireland
                    ):
                        file_path = os.path.join(raster_root, raster_file)
                        raster_key = raster_file.split(".")[0]
                        vector_file = f"{raster_key}.shp"
                        vector_file_VE = vector_file.replace("RD", "VE")
                        vector_file_PR = vector_file.replace("RD", "PR")
                        if (vector_file_VE in vector_file_list) or (
                            vector_file_PR in vector_file_list
                        ):
                            raster_path_with_vector.append(file_path)
                        else:
                            raster_path_without_vector.append(file_path)
    """
    # create vector data for rasters within raster_path_without_vector
    # gdal_polygonize xxx
    """
    # damage analysis (with vector data)
    for flood_path in raster_path_with_vector:
        # raster data path
        flood_key = flood_path.split("\\")[-1].split(".")[0]

        # clip data path, clip analysis
        clip_path = (
            flood_path.replace("Raster", "Vector")
            .replace("tif", "shp")
            .replace("RD", "VE")
        )
        if not os.path.exists(clip_path):
            clip_path = clip_path.replace("VE", "PR")
        clipped_features = clip_features(road_links, clip_path, flood_key)

        # output path
        intersections_path1 = base_path / "outputs" / f"{flood_key}_damages.csv"
        intersections_path2 = base_path / "outputs" / f"{flood_key}_damages_gp.csv"

        # damage analysis
        intersections_with_damages = calculate_damages(
            flood_path, flood_key, clipped_features, damage_curve_dict, damage_cost_dict
        )
        intersections_with_damages.to_csv(intersections_path1, index=False)

        intersections_with_damages = intersections_with_damages.groupby(
            by=["id"], as_index=False
        ).agg(
            {
                f"{flood_key}_depth": "max",
                f"{flood_key}_min_damage_values": "sum",
                f"{flood_key}_max_damage_values": "sum",
            }
        )
        intersections_with_damages.rename(
            columns={f"{flood_key}_depth": f"{flood_key}_max_depth"}, inplace=True
        )
        intersections_with_damages.to_csv(intersections_path2, index=False)


if __name__ == "__main__":
    main()
