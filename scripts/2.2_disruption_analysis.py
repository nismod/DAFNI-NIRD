import logging
import os
import sys
from pathlib import Path

import pandas as pd
import geopandas as gpd

import snail.intersection
import snail.io
from nird.utils import load_config

import warnings

warnings.simplefilter("ignore")


def intersect_features_with_rasters(rasters, grid_id, grid, features):
    # convert feature crs to same as raster
    features_crs = features.crs
    features = features.to_crs(grid.crs)

    # optionally mask, but return all
    # to_intersect, to_skip = clip_features(features, clip_path, flood_key)

    prepared = snail.intersection.prepare_linestrings(features)
    intersections = snail.intersection.split_linestrings(prepared, grid)
    intersections = snail.intersection.apply_indices(
        intersections, grid, index_i=f"index_i_{grid_id}", index_j=f"index_j_{grid_id}"
    )

    for raster in rasters.itertuples():
        # read the raster data: depth (meter)
        raster_data = snail.io.read_raster_band_data(raster.path)
        intersections[raster.key] = snail.intersection.get_raster_values_for_splits(
            intersections,
            raster_data,
            index_i=f"index_i_{grid_id}",
            index_j=f"index_j_{grid_id}",
        ).fillna(0.0)

    intersections.to_crs(features_crs)

    return intersections


def clip_features(features, clip_path, raster_key):
    print(f"Clipping features based on {raster_key}...")
    clips = gpd.read_file(clip_path, engine="pyogrio")
    if clips.crs != "epsg:4326":
        clips = clips.to_crs("epsg:4326")
    assert (
        clips.crs == features.crs
    ), "CRS mismatch! Ensure both layers use the same CRS."
    clipped_features = gpd.sjoin(features, clips, how="inner", predicate="intersects")
    clipped_features = clipped_features[features.columns]
    clipped_features.reset_index(drop=True, inplace=True)
    return clipped_features


def compute_maximum_speed_on_flooded_roads(depth, free_flow_speed):
    depth = depth * 1000  # m to mm
    if depth < 300:  # mm: !!! uncertainty analysis (150, 300, 600)
        # value = 0.0009 * depth**2 - 0.5529 * depth + 86.9448  # kmph
        value = free_flow_speed * (depth / 300 - 1) ** 2  # mph
        return value  # mph
    else:
        return 0.0  # mph


def compute_damage_level_on_flooded_roads(
    fldType,
    road_classification,
    trunk_road,
    hasTunnel,
    fldDepth,
):
    depth = fldDepth * 100  # cm
    if fldType == "surface":
        if hasTunnel == 1 and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road == "true")
        ):
            if depth < 150:
                return "no"
            if 150 <= depth < 200:
                return "minor"
            elif 200 <= depth < 300:
                return "moderate"
            elif 300 <= depth < 700:
                return "extensive"
            else:
                return "severe"
        elif hasTunnel == 0 and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road == "true")
        ):
            if depth < 150:
                return "no"
            if 150 <= depth < 200:
                return "no"
            elif 200 <= depth < 300:
                return "no"
            elif 300 <= depth < 700:
                return "minor"
            else:
                return "moderate"
        else:
            if depth < 50:
                return "no"
            if 50 <= depth < 100:
                return "no"
            elif 100 <= depth < 200:
                return "minor"
            elif 200 <= depth < 600:
                return "minor"
            else:
                return "moderate"
    elif fldType == "river":
        if hasTunnel == 1 and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road == "true")
        ):
            if depth < 250:
                return "no"
            if 250 <= depth < 300:
                return "minor"
            elif 300 <= depth < 400:
                return "minor"
            elif 400 <= depth < 800:
                return "moderate"
            else:
                return "extensive"
        elif hasTunnel == 0 and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road == "true")
        ):
            if depth < 250:
                return "no"
            if 250 <= depth < 300:
                return "minor"
            elif 300 <= depth < 400:
                return "moderate"
            elif 400 <= depth < 800:
                return "extensive"
            else:
                return "severe"
        else:
            if depth < 50:
                return "minor"
            if 50 <= depth < 100:
                return "moderate"
            elif 100 <= depth < 200:
                return "moderate"
            elif 200 <= depth < 600:
                return "extensive"
            else:
                return "severe"
    else:
        print("Please enter the type of flood!")


def _join_dirname(path, dirname):
    return os.path.join(dirname, path)


def calculate_exposure(raster_path, features_path, exposure_path):
    rasters = pd.read_csv(raster_path / "rasters_RD.csv")

    rasters.path = rasters.path.apply(_join_dirname, args=(raster_path,))
    rasters, grids = snail.io.extend_rasters_metadata(rasters)
    logging.info("Read metadata for %s rasters.", len(rasters))
    # road links
    road_link_intersections = gpd.read_parquet(features_path)
    logging.info("Read %s road links.", len(road_link_intersections))

    for grid_id, grid in enumerate(grids):
        grid_rasters = rasters.query(f"grid_id == {grid_id}")
        logging.info("Intersecting %s rasters for grid %s", len(grid_rasters), grid_id)
        # intersection analysis
        road_link_intersections = intersect_features_with_rasters(
            grid_rasters, grid_id, grid, road_link_intersections
        )
    road_link_intersections.to_parquet(exposure_path)


def calculate_disruption(base_path, raster_path):
    rasters = pd.read_csv(raster_path / "rasters_RD.csv")

    road_links = gpd.read_parquet(
        base_path / "results" / "GB_road_link_file_intersections.pq"
    )
    base_scenario_links = gpd.read_parquet(
        base_path / "results" / "edge_flows_partial_1128.pq"
    )
    base_scenario_links.rename(
        columns={
            "acc_capacity": "current_capacity",
            "acc_speed": "current_speed",
        },
        inplace=True,
    )

    # attach capacity and speed info on D-zero (Edge_flows)
    road_links = road_links.merge(
        base_scenario_links[
            [
                "id",
                "current_capacity",
                "current_speed",
                "free_flow_speeds",
                "initial_flow_speeds",
                "min_flow_speeds",
            ]
        ],
        how="left",
        on="id",
    )

    for raster in rasters.itertuples():
        if "FLRF" in raster.key:
            flood_type = "river"
        else:
            flood_type = "surface"

        # calculate the maximum speeds on flooded road links
        road_links["max_speed"] = road_links.apply(
            lambda row: compute_maximum_speed_on_flooded_roads(
                row[raster.key], row["free_flow_speeds"]
            ),
            axis=1,
        )

        # estimate damage levels
        if flood_type == "surface":
            road_links["damage_level"] = road_links.apply(
                lambda row: compute_damage_level_on_flooded_roads(
                    "surface",
                    row["road_classification"],
                    row["trunk_road"],
                    row["hasTunnel"],
                    row[raster.key],
                ),
                axis=1,
            )
        elif flood_type == "river":
            road_links["damage_level"] = road_links.apply(
                lambda row: compute_damage_level_on_flooded_roads(
                    "river",
                    row.road_classification,
                    row.trunk_road,
                    row.hasTunnel,
                    row[raster.key],
                ),
                axis=1,
            )
        else:
            print(f"WARNING: unknown flood type {flood_type}")
            pass

    road_links.to_parquet(
        base_path / "results" / "disruption_analysis_1129" / "road_links.pq"
    )
    road_links.drop(columns="geometry", inplace=True)
    road_links.to_csv(
        base_path / "results" / "disruption_analysis_1129" / "road_links.csv",
        index=False,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        index_i = sys.argv[1]
        index_j = sys.argv[2]
    except IndexError:
        print(f"Usage: python {__file__} <index_i> <index_j> [<path/to/config.json>]")

    try:
        config_path = sys.argv[3]
    except IndexError:
        config_path = None

    base_path = Path(load_config(config_path)["paths"]["base_path"])
    raster_path = Path(load_config(config_path)["paths"]["JBA_data"])

    features_path = (
        base_path
        / "processed_data"
        / "networks"
        / "GB_road_link_file_100k.parquet"
        / f"{index_i}_{index_j}.parquet"
    )
    exposure_path = (
        base_path
        / "results"
        / "GB_road_link_file_intersections.parquet"
        / f"{index_i}_{index_j}.parquet"
    )

    if (exposure_path).exists():
        logging.info("Skipping exposure recalculation")
    else:
        calculate_exposure(raster_path, features_path, exposure_path)

    # calculate_disruption(base_path, raster_path)
