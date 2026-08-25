# %%
# ruff: noqa
import sys
from typing import Dict, Tuple
from pathlib import Path

import geopandas as gpd
import numpy as np
from snail import io, intersection
import warnings
import logging
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
from nird.utils import load_config

warnings.filterwarnings("ignore")

base_path = Path(load_config()["paths"]["MACCHUB"])


# %%
def intersect_features_with_raster(
    raster: np.ndarray,
    grid: gpd.GeoDataFrame,
    raster_key: str,  # this is used for logging purposes only
    features: gpd.GeoDataFrame,
    flood_type: str,
    scenario_key: str,
) -> gpd.GeoDataFrame:
    """
    Intersects vector features with a raster dataset to compute flood depth for each
        feature.

        raster (np.ndarray): Raster band data containing flood values.
    scenario_key: str,
        grid (gpd.GeoDataFrame): Raster grid metadata used for spatial operations.
        raster_key (str): Identifier for the raster dataset.
        features (gpd.GeoDataFrame): GeoDataFrame containing vector features (e.g.,
            road links).
        flood_type (str): Type of flood (e.g., "surface" or "river").

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of intersected features with flood depth values,
                          reprojected to EPSG:27700.
    """
    logging.info(f"Intersecting features with raster {raster_key}...")

    # run the intersection analysis
    prepared = intersection.prepare_linestrings(features)
    if prepared.crs != grid.crs:
        logging.info("Projecting Feature (clipped) CRS to Grid CRS...")
        prepared = prepared.to_crs(grid.crs)

    intersections = intersection.split_linestrings(prepared, grid)
    intersections = intersection.apply_indices(intersections, grid)

    # flood depth code: map to value (1: 0-0.3, 2:0.3-1, 3: >1m)
    # flood range -> depth (m)
    # baseline: (1->0.15, 2->0.65, 3->1.5)
    # future: (1->0.3, 2->1, 3->2)
    raster_values = intersection.get_raster_values_for_splits(intersections, raster)
    depth_mappings = {
        "base": {1: 0.15, 2: 0.65, 3: 1.5},
        "future": {1: 0.3, 2: 1.0, 3: 2.0},
    }
    try:
        depth_mapping = depth_mappings[scenario_key]
    except KeyError as error:
        raise ValueError("scenario_key must be 'base' or 'future'") from error

    intersections[f"flood_depth_{flood_type}"] = raster_values.replace(depth_mapping)

    # reproject back
    # intersections = intersections.to_crs("epsg:27700")
    intersections = intersections.to_crs(features.crs)
    intersections["length"] = intersections.geometry.length  # segment length in meters

    return intersections


def clip_features_with_polygon(
    features: gpd.GeoDataFrame,
    clip_path: str,
    raster_key: str,  # this is used for logging purposes only
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
    if clipper.crs is None:
        raise ValueError("The clipping layer has no CRS.")
    if features.crs is None:
        raise ValueError("The features layer has no CRS.")

    clipper = clipper[clipper.geometry.notna() & ~clipper.geometry.is_empty]
    if clipper.empty:
        raise ValueError("The clipping layer contains no usable geometries.")

    if features.crs != clipper.crs:
        logging.info("Projecting Feature CRS to match GRID CRS...")
        features = features.to_crs(clipper.crs)
    clipper = clipper.dissolve()
    clipped_features = gpd.clip(features, clipper)
    # clipped_features = gpd.sjoin(features, clips, how="inner", predicate="intersects")
    # clipped_features = clipped_features[features.columns]
    clipped_features.reset_index(drop=True, inplace=True)

    return clipped_features


def clip_raster_with_polygon(
    raster_path: str,
    clip_path: str,
    output_path: str,
) -> Tuple[np.ndarray, gpd.GeoDataFrame]:
    """Clip a raster to the geometry in a polygon shapefile.

    The clip geometry is reprojected to the raster CRS when necessary. The clipped
    raster keeps the source data type, CRS, and nodata value.

    Parameters:
        raster_path: Path to the input .tif or .tiff raster.
        clip_path: Path to the polygon shapefile used as the clipping boundary.
        output_path: Path where the clipped GeoTIFF will be written.

    Returns:
        Tuple containing the clipped raster data and its updated raster grid.
    """

    clipper = gpd.read_file(clip_path, engine="pyogrio")
    if clipper.crs is None:
        raise ValueError("The clipping layer has no CRS.")

    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError("The raster has no CRS.")
        if clipper.crs != source.crs:
            clipper = clipper.to_crs(source.crs)

        geometries = clipper.geometry[
            clipper.geometry.notna() & ~clipper.geometry.is_empty
        ]
        if geometries.empty:
            raise ValueError("The clipping shapefile contains no valid geometries.")

        nodata = source.nodata if source.nodata is not None else 0
        clipped_data, clipped_transform = mask(
            source,
            geometries.tolist(),
            crop=True,
            filled=True,
            nodata=nodata,
        )
        metadata = source.meta.copy()
        metadata.update(
            {
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clipped_transform,
                "nodata": nodata,
                "compress": "deflate",
                "zlevel": 9,
                "predictor": 3 if np.issubdtype(clipped_data.dtype, np.floating) else 2,
            }
        )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **metadata) as destination:
        destination.write(clipped_data)

    clipped_raster = io.read_raster_band_data(output_path)
    clipped_grid, _ = io.read_raster_metadata(output_path)
    return clipped_raster, clipped_grid


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
    road_label: str,
    fldDepth: float,
    trunk_road: bool = False,
) -> str:
    """
    Determines the damage level of roads based on flood type, road classification,
        and flood depth.

    Parameters:
        fldType (str): Type of flood ("surface" or "river").
        road_classification (str): Classification of road (e.g., "Motorway", "A Road").
        trunk_road (bool): Retained for compatibility but ignored for NI roads.
        road_label (str): Label of the road (e.g., "road", "tunnel", "bridge").
        fldDepth (float): Flood depth in meters.

    Returns:
        str: Damage level categorized as "no", "minor", "moderate", "extensive",
            or "severe"
    """

    depth = fldDepth * 100  # convert from m to cm
    if fldType == "surface":
        if road_label == "tunnel" and road_classification == "Motorway":
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
        elif road_label != "tunnel" and road_classification == "Motorway":
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
        if road_label == "tunnel" and road_classification == "Motorway":
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
        elif road_label != "tunnel" and road_classification == "Motorway":
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
        raise ValueError("flood type must be 'surface' or 'river'")


def features_with_damage(
    features: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    damage_level_dict: Dict,
    damage_level_dict_reverse: Dict,
) -> gpd.GeoDataFrame:
    """
    Aggregates flood depths and damage levels for road links based on intersection data.

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

    intersections = intersections.copy()

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


def main(
    nation: str,
    scenario_key: str,
    flood_path: str,
    flood_type: str,
    flow_path: str,
    link_path: str,
    out_path: str,
    clip_path: str = None,
):
    if flood_type not in {"surface", "river"}:
        raise ValueError("flood_type must be 'surface' or 'river'")

    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # damage level dicts
    damage_level_dict = {
        "no": 0,
        "minor": 1,
        "moderate": 2,
        "extensive": 3,
        "severe": 4,
    }
    damage_level_dict_reverse = {i: k for k, i in damage_level_dict.items()}

    # flows
    flows = gpd.read_parquet(flow_path)
    # check whether flows already has "current_capacity"
    if "current_capacity" in flows.columns:
        flows.rename(columns={"current_capacity": "total_capacity"}, inplace=True)

    flows = flows.rename(
        columns={
            "acc_capacity": "current_capacity",
            "acc_speed": "current_speed",
            "acc_flow": "current_flow",
        }
    )
    flows = flows.loc[:, ~flows.columns.duplicated()].copy()
    flow_columns = [
        "combined_label",
        "free_flow_speeds",
        "initial_flow_speeds",
        "min_flow_speeds",
        "current_capacity",
        "current_speed",
        "current_flow",
    ]
    missing_flow_columns = set(flow_columns) - set(flows.columns)
    if missing_flow_columns:
        raise ValueError(
            f"Flow data is missing required columns: {sorted(missing_flow_columns)}"
        )

    # network edges
    road_links = gpd.read_parquet(link_path)
    features = road_links.copy()  # to ensure road links crs not changed
    raster = io.read_raster_band_data(flood_path)
    grid, _ = io.read_raster_metadata(flood_path)

    # clip features and flood raster with the provided vector file
    if clip_path is not None:
        features = clip_features_with_polygon(features, clip_path, flood_path)
        clipped_raster_path = Path(out_path) / f"{nation}_{scenario_key}_clipped.tif"
        raster, grid = clip_raster_with_polygon(
            flood_path,
            clip_path,
            clipped_raster_path,
        )
    else:
        logging.info("No clipping path provided. Using full extent of input raster.")
        with rasterio.open(flood_path) as src:
            left, bottom, right, top = src.bounds
        bbox = box(left, bottom, right, top)
        if src.crs != features.crs:
            features = features.to_crs(src.crs)

        features = features.clip(bbox)
        features.reset_index(drop=True, inplace=True)

    # intersection analysis
    temp = intersect_features_with_raster(
        raster,
        grid,
        nation,
        features,
        flood_type,
        scenario_key,
    )
    temp.reset_index(drop=True, inplace=True)

    # adjust flood depths for embankment heights based on road classification
    # if scenario_key == "base":  # for base scenario
    #     if flood_type == "surface":
    #         temp.loc[
    #             (temp.road_classification == "Motorway"),
    #             "flood_depth_surface",
    #         ] = (temp["flood_depth_surface"] - 1.0).clip(lower=0)
    #     elif flood_type == "river":
    #         temp.loc[
    #             (temp.road_classification == "Motorway"),
    #             "flood_depth_river",
    #         ] = (
    #             temp["flood_depth_river"] - 2.0
    #         ).clip(lower=0)
    #     else:
    #         logging.info("Please enter the type of flood!")
    #         sys.exit()

    temp[f"damage_level_{flood_type}"] = temp.apply(
        lambda row: compute_damage_level_on_flooded_roads(
            flood_type,
            row["road_classification"],
            row["road_label"],
            row[f"flood_depth_{flood_type}"],
        ),
        axis=1,
    )
    temp = temp[temp[f"flood_depth_{flood_type}"] >= 0].reset_index(drop=True)

    # save intersections
    intersections = gpd.GeoDataFrame(columns=["e_id", "length", "index_i", "index_j"])
    depth_column = f"flood_depth_{flood_type}"
    damage_column = f"damage_level_{flood_type}"
    intersections = intersections.merge(
        temp[
            [
                "e_id",
                "length",
                "index_i",
                "index_j",
                depth_column,
                damage_column,
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
    temp_links = temp_links.drop(
        columns=[column for column in flow_columns if column in temp_links.columns]
    ).merge(
        flows[["e_id", *flow_columns]],
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


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"base", "future"}:
        raise SystemExit(
            "Usage: python scripts/macchub/2_intersection_analysis_NI.py "
            "<base|future>"
        )

    inputs_dict = {
        "flood_path_base": base_path / "hazards" / "all_Q100_Defended.tif",
        "flood_path_future": base_path / "hazards" / "all_Q100CC_Defended.tif",
        "link_path": base_path / "networks" / "edges_final.gpq",
        "flow_path": base_path.parent / "outputs" / "NI" / "edge_flow_NI.gpq",
        "out_path": base_path.parent / "outputs" / "NI",
        "clip_path": base_path / "hazards" / "ni-historical-flooding-aug2008.gpkg",
    }
    nation = "northern_ireland"
    scenario_key = sys.argv[1]
    flood_type = "river"  # surface or river
    main(
        nation=nation,
        scenario_key=scenario_key,
        flood_path=inputs_dict[f"flood_path_{scenario_key}"],
        flood_type=flood_type,
        flow_path=inputs_dict["flow_path"],
        link_path=inputs_dict["link_path"],
        out_path=inputs_dict["out_path"],
        clip_path=inputs_dict["clip_path"],
    )
