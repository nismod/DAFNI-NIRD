import sys
from typing import Dict
from pathlib import Path

import geopandas as gpd
import numpy as np
from snail import io, intersection
import warnings
import logging
import rasterio
from shapely.geometry import box

warnings.filterwarnings("ignore")


def intersect_features_with_raster(
    raster_path: str,
    raster_key: str,
    features: gpd.GeoDataFrame,
    flood_type: str,
) -> gpd.GeoDataFrame:
    """
    Intersects vector features with a raster dataset to compute flood depth for each
        feature.

    Parameters:
        raster_path (str): Path to the raster file containing flood data.
        raster_key (str): Identifier for the raster dataset.
        features (gpd.GeoDataFrame): GeoDataFrame containing vector features (e.g.,
            road links).
        flood_type (str): Type of flood (e.g., "surface" or "river").

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of intersected features with flood depth values,
                          reprojected to EPSG:27700.
    """

    logging.info(f"Intersecting features with raster {raster_key}...")
    # read the raster data: depth (meter)
    raster = io.read_raster_band_data(raster_path)

    # run the intersection analysis
    grid, _ = io.read_raster_metadata(raster_path)
    prepared = intersection.prepare_linestrings(features)
    if prepared.crs != grid.crs:
        logging.info("Projecting Feature (clipped) CRS to Grid CRS...")
        prepared = prepared.to_crs(grid.crs)

    intersections = intersection.split_linestrings(prepared, grid)
    intersections = intersection.apply_indices(intersections, grid)
    intersections[f"flood_depth_{flood_type}"] = (
        intersection.get_raster_values_for_splits(intersections, raster)
    )

    # reproject back
    intersections = intersections.to_crs("epsg:27700")
    intersections["length"] = intersections.geometry.length  # segment length in meters

    return intersections


def clip_features(
    features: gpd.GeoDataFrame,
    clip_path: str,
    raster_key: str,
) -> gpd.GeoDataFrame:
    """
    Clips spatial features to the extent of a specified vector layer.

    Parameters:
        features (gpd.GeoDataFrame): GeoDataFrame containing the spatial features to be
            clipped.
        clip_path (str): Path to the vector file used for clipping.
        raster_key (str): Identifier for the raster dataset.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of features clipped to the extent of the clip
            layer.
    """

    logging.info(f"Clipping features based on {raster_key}...")
    clipper = gpd.read_file(clip_path, engine="pyogrio")  # grid's extent (vector)
    if features.crs != clipper.crs:
        logging.info("Projecting Feature CRS to match GRID CRS...")
        features = features.to_crs(clipper.crs)
    clipper = clipper.dissolve()
    clipped_features = gpd.clip(features, clipper)
    # clipped_features = gpd.sjoin(features, clips, how="inner", predicate="intersects")
    # clipped_features = clipped_features[features.columns]
    clipped_features.reset_index(drop=True, inplace=True)

    return clipped_features


def compute_maximum_speed_on_flooded_roads(
    depth: float,
    free_flow_speed: float,
    threshold=30,
) -> float:
    """
    Calculates the maximum allowable speed on flooded roads based on flood depth.

    Parameters:
        depth (float): Flood depth in meters.
        free_flow_speed (float): Free-flow speed under normal conditions (mph).
        threshold (float, optional): Depth threshold in centimeters for road closure
            (default is 30 cm).

    Returns:
        float: Maximum speed on the flooded road in miles per hour (mph).
    """

    depth = depth * 100  # m to cm
    if depth < threshold:  # cm
        value = free_flow_speed * (depth / threshold - 1) ** 2  # mph
        return value  # mph
    else:
        return 0.0  # mph


def compute_damage_level_on_flooded_roads(
    fldType: str,
    road_classification: str,
    trunk_road: str,
    road_label: str,
    fldDepth: float,
) -> str:
    """
    Determines the damage level of roads based on flood type, road classification,
        and flood depth.

    Parameters:
        fldType (str): Type of flood ("surface" or "river").
        road_classification (str): Classification of road (e.g., "Motorway", "A Road").
        trunk_road (bool): Indicates if the road is a trunk road (True/False).
        road_label (str): Label of the road (e.g., "road", "tunnel", "bridge").
        fldDepth (float): Flood depth in meters.

    Returns:
        str: Damage level categorized as "no", "minor", "moderate", "extensive",
            or "severe"
    """

    depth = fldDepth * 100  # convert from m to cm
    if fldType == "surface":
        if road_label == "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            elif 50 <= depth < 100:
                return "minor"
            elif 100 <= depth < 200:
                return "moderate"
            elif 200 <= depth < 600:
                return "extensive"
            elif depth >= 600:
                return "severe"
            else:
                return np.nan
        elif road_label != "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            elif 50 <= depth < 100:
                return "no"
            elif 100 <= depth < 200:
                return "no"
            elif 200 <= depth < 600:
                return "minor"
            elif depth >= 600:
                return "moderate"
            else:
                return np.nan
        else:
            if depth < 50:
                return "no"
            elif 50 <= depth < 100:
                return "no"
            elif 100 <= depth < 200:
                return "minor"
            elif 200 <= depth < 600:
                return "minor"
            elif depth >= 600:
                return "moderate"
            else:
                return np.nan

    elif fldType == "river":
        if road_label == "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            elif 50 <= depth < 100:
                return "minor"
            elif 100 <= depth < 200:
                return "minor"
            elif 200 <= depth < 600:
                return "moderate"
            elif depth >= 600:
                return "extensive"
            else:
                return np.nan
        elif road_label != "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            elif 50 <= depth < 100:
                return "minor"
            elif 100 <= depth < 200:
                return "moderate"
            elif 200 <= depth < 600:
                return "extensive"
            elif depth >= 600:
                return "severe"
            else:
                return np.nan
        else:
            if depth <= 0:
                return "no"
            elif 0 < depth < 50:
                return "minor"
            elif 50 <= depth < 100:
                return "moderate"
            elif 100 <= depth < 200:
                return "moderate"
            elif 200 <= depth < 600:
                return "extensive"
            elif depth >= 600:
                return "severe"
            else:
                return np.nan
    else:
        logging.info("Please enter the type of flood!")


def intersections_with_damage(
    road_links: gpd.GeoDataFrame,
    flood_key: str,
    flood_type: str,
    flood_path: str,
    clip_path: str,
) -> gpd.GeoDataFrame:
    """
    Computes flood depth and damage levels for road segments by intersecting them with
        flood data.

    Parameters:
        road_links (gpd.GeoDataFrame): GeoDataFrame of road links with geometries and
            classifications.
        flood_key (str): Identifier for the flood dataset.
        flood_type (str): Type of flood ("surface" or "river").
        flood_path (str): Path to the flood raster file.
        clip_path (str): Path to the vector file used for clipping.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of intersections with calculated flood depths
            and damage levels.
    """

    # Clip road links with features in the provided vector file
    features = road_links.copy()
    clipped_features = clip_features(features, clip_path, flood_key)
    if clipped_features.empty:
        logging.info("Warning: Clip features is None!")
        return None
    # Perform intersection analysis with the flood raster
    intersections = intersect_features_with_raster(
        flood_path,
        flood_key,
        clipped_features,
        flood_type,
    )
    intersections.reset_index(drop=True, inplace=True)
    # Adjust flood depths for embankment heights based on road classification
    """
    embankment against surface flood: 100 cm
    embankment against river flood: 200 cm
    """

    #!!! rerun analysis without embankment adjustment (for baseline scenario)
    if flood_type == "surface":
        intersections.loc[
            (intersections.road_classification == "Motorway")
            | (
                (intersections.road_classification == "A Road")
                & (intersections["trunk_road"])
            ),
            "flood_depth_surface",
        ] = (intersections["flood_depth_surface"] - 100).clip(lower=0)
    else:
        intersections.loc[
            (intersections.road_classification == "Motorway")
            | (
                (intersections.road_classification == "A Road")
                & (intersections["trunk_road"])
            ),
            "flood_depth_river",
        ] = (intersections["flood_depth_river"] - 200).clip(lower=0)

    # Compute damage levels for flooded road segments
    intersections[f"damage_level_{flood_type}"] = intersections.apply(
        lambda row: compute_damage_level_on_flooded_roads(
            flood_type,
            row["road_classification"],
            row["trunk_road"],
            row["road_label"],
            row[f"flood_depth_{flood_type}"],
        ),
        axis=1,
    )

    return intersections


def features_with_damage(
    features: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    damage_level_dict: Dict,
    damage_level_dict_reverse: Dict,
) -> gpd.GeoDataFrame:
    """
    Aggregates flood depth and damage levels for road links based on intersection data.

    Parameters:
        features (gpd.GeoDataFrame): GeoDataFrame of road links.
        intersections (gpd.GeoDataFrame): GeoDataFrame of intersections with flood data.
        damage_level_dict (Dict): Mapping of damage levels to numerical values.
        damage_level_dict_reverse (Dict): Reverse mapping of numerical values to damage
            levels.

    Returns:
        gpd.GeoDataFrame: Updated GeoDataFrame of road links with maximum flood depth
            and damage levels.
    """

    # Flood depth
    if (
        "flood_depth_surface" in intersections.columns
        and "flood_depth_river" in intersections.columns
    ):
        intersections["flood_depth_max"] = intersections[
            ["flood_depth_surface", "flood_depth_river"]
        ].max(axis=1)
    elif "flood_depth_surface" in intersections.columns:
        intersections["flood_depth_max"] = intersections.flood_depth_surface
    elif "flood_depth_river" in intersections.columns:
        intersections["flood_depth_max"] = intersections.flood_depth_river
    else:
        logging.info("Error: flood depth columns are missing!")
        sys.exit()

    # Damage level
    if (
        "damage_level_surface" in intersections.columns
        and "damage_level_river" in intersections.columns
    ):
        intersections["damage_level_surface"] = intersections[
            "damage_level_surface"
        ].map(damage_level_dict)
        intersections["damage_level_river"] = intersections["damage_level_river"].map(
            damage_level_dict
        )
        intersections["damage_level_max"] = intersections[
            ["damage_level_surface", "damage_level_river"]
        ].max(axis=1)
    elif "damage_level_surface" in intersections.columns:
        intersections["damage_level_surface"] = intersections[
            "damage_level_surface"
        ].map(damage_level_dict)
        intersections["damage_level_max"] = intersections.damage_level_surface
    elif "damage_level_river" in intersections.columns:
        intersections["damage_level_river"] = intersections["damage_level_river"].map(
            damage_level_dict
        )
        intersections["damage_level_max"] = intersections.damage_level_river
    else:
        logging.info("Error: damage level columns are missing!")

    intersections_gp = intersections.groupby("e_id", as_index=False).agg(
        {
            "flood_depth_max": "max",
            "damage_level_max": "max",
        }
    )
    intersections_gp["damage_level_max"] = intersections_gp.damage_level_max.astype(
        int
    ).map(damage_level_dict_reverse)

    features = features.merge(
        intersections_gp[["e_id", "flood_depth_max", "damage_level_max"]],
        how="left",
        on="e_id",
    )
    features["flood_depth_max"] = features["flood_depth_max"].fillna(0.0)
    features["damage_level_max"] = features["damage_level_max"].fillna("no")

    return features


def main(nation, flood_key, flow_key, scenario_key):
    # flood event inputs
    # nation = "scotland"
    flood_path = r"C:\Oxford\Research\MACCHUB\local\data\processed"
    flood_path = Path(flood_path) / f"{flood_key}"
    # flood_path = (
    #     r"C:\Oxford\Research\MACCHUB\local\data\processed\future_wales_medium.tif"
    # )
    # ["futrue_scotland_medium.tif", "future_thames_medium.tif", "future_wales_medium.tif"]
    # ["base_scotland.tif", "base_thames.tif", "dennis_clip.tif"]
    flood_type = "river"

    # damage level dicts
    damage_level_dict = {
        "no": 0,
        "minor": 1,
        "moderate": 2,
        "extensive": 3,
        "severe": 4,
    }
    damage_level_dict_reverse = {i: k for k, i in damage_level_dict.items()}
    out_path = Path(r"C:\Oxford\Research\MACCHUB\local\scripts\outputs\intersections")
    # out_path.mkdir(parents=True, exist_ok=True)

    # baseline flows
    flow_path = r"C:\Oxford\Research\MACCHUB\local\scripts\outputs\flows"
    flow_path = Path(flow_path) / f"{flow_key}"
    flows = gpd.read_parquet(
        flow_path
    )  # ["baseline_flows.gpq", "edge_flow_2050_ssp5.gpq"]
    flows.rename(
        columns={
            "acc_capacity": "current_capacity",
            "acc_speed": "current_speed",
            "acc_flow": "current_flow",
        },
        inplace=True,
    )

    # road links
    road_links = gpd.read_parquet(
        r"C:\Oxford\Research\DAFNI\local\processed_data\networks\road\GB_road_links_with_bridges.gpq"
    )
    features = road_links.copy()  # !!! to ensure road links crs not changed

    # clip features
    with rasterio.open(flood_path) as src:
        left, bottom, right, top = src.bounds
    bbox = box(left, bottom, right, top)
    if src.crs != features.crs:
        features = features.to_crs(src.crs)

    clipped = features.clip(bbox)
    clipped.reset_index(drop=True, inplace=True)

    # intersction analysis
    temp = intersect_features_with_raster(flood_path, "england", clipped, "river")
    temp.reset_index(drop=True, inplace=True)

    # adjust flood depths for embankment heights based on road classification
    if scenario_key == "base":  # for base scenario
        temp.loc[
            (temp.road_classification == "Motorway")
            | ((temp.road_classification == "A Road") & (temp["trunk_road"])),
            "flood_depth_river",
        ] = (temp["flood_depth_river"] - 200).clip(lower=0)

    temp[f"damage_level_{flood_type}"] = temp.apply(
        lambda row: compute_damage_level_on_flooded_roads(
            flood_type,
            row["road_classification"],
            row["trunk_road"],
            row["road_label"],
            row[f"flood_depth_{flood_type}"],
        ),
        axis=1,
    )
    temp = temp[temp["flood_depth_river"] >= 0].reset_index(
        drop=True
    )  #!!! "remove featrues with invalid flood depth -> intersect with no data cells"

    # save intersections
    intersections = gpd.GeoDataFrame(columns=["e_id", "length", "index_i", "index_j"])
    intersections = intersections.merge(
        temp[
            [
                "e_id",
                "length",
                "index_i",
                "index_j",
                "flood_depth_river",
                "damage_level_river",
            ]
        ],
        on=["e_id", "length", "index_i", "index_j"],
        how="outer",
    )
    intersections.to_parquet(out_path / f"intersections_{nation}_{scenario_key}.pq")

    # save road links
    # baseline flows with damage info (flood depth, damage level, max speed)
    temp_links = features_with_damage(
        road_links,
        intersections,
        damage_level_dict,
        damage_level_dict_reverse,
    )
    temp_links = temp_links.merge(
        flows[
            [
                "e_id",
                "combined_label",
                "free_flow_speeds",
                "initial_flow_speeds",
                "min_flow_speeds",
                "current_capacity",
                "current_speed",
                "current_flow",
            ]
        ],
        how="left",
        on="e_id",
    )

    temp_links["max_speed"] = temp_links.apply(
        lambda row: compute_maximum_speed_on_flooded_roads(
            depth=row["flood_depth_max"],
            free_flow_speed=row["free_flow_speeds"],
        ),
        axis=1,
    )

    temp_links.to_parquet(out_path / f"road_links_{nation}_{scenario_key}.gpq")
    print("completed!")
    return


if __name__ == "__main__":
    # # future
    flood_keys = {
        "england": "future_thames_medium.tif",
        "scotland": "future_scotland_medium.tif",
        "wales": "future_wales_medium.tif",
    }
    flow_key = "edge_flow_2050_ssp5.gpq"
    scenario_key = "future"

    # base
    # flood_keys = {
    #     "england": "base_thames.tif",
    #     "scotland": "base_scotland.tif",
    #     "wales": "dennis_clip.tif",
    # }
    # flow_key = "baseline_flows.gpq"
    # scenario_key = "base"
    for nation, flood_key in flood_keys.items():
        main(
            nation=nation,
            flood_key=flood_key,
            flow_key=flow_key,
            scenario_key=scenario_key,
        )
    # main(nation="wales", flood_key="wales", scenario="future")
