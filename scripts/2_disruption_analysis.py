# %%
import sys
from pathlib import Path

# import pandas as pd
import geopandas as gpd

from snail import io, intersection
from nird.utils import load_config

import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])
raster_path = Path(load_config()["paths"]["JBA_data"])


# %%
def intersect_features_with_raster(raster_path, raster_key, features):
    print(f"Intersecting features with raster {raster_key}...")
    # read the raster data: depth (meter)
    raster = io.read_raster_band_data(raster_path)

    # run the intersection analysis
    grid, _ = io.read_raster_metadata(raster_path)
    prepared = intersection.prepare_linestrings(features)
    intersections = intersection.split_linestrings(prepared, grid)
    if intersections is None:
        sys.exit()
    intersections = intersection.apply_indices(intersections, grid)
    intersections["flood_depth"] = intersection.get_raster_values_for_splits(
        intersections, raster
    )

    # reproject back
    intersections = intersections.to_crs("epsg:27700")
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


def identify_disrupted_links(road_links, flood_key, flood_path, clip_path, out_path):
    # convert feature crs to WGS-84 (same as raster)
    road_links = road_links.to_crs("epsg:4326")
    # mask
    clipped_features = clip_features(road_links, clip_path, flood_key)
    # intersection
    intersections = intersect_features_with_raster(
        flood_path, flood_key, clipped_features
    )
    # export intersected featrues
    intersections.to_csv(out_path, index=False)
    print("Complete identifying the disrupted links!")

    return intersections


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


def main(flood_type=None):
    # road links
    road_links = gpd.read_parquet(
        base_path / "networks" / "road" / "GB_road_link_file.pq"
    )
    base_scenario_links = gpd.read_parquet(
        base_path.parent / "outputs" / "edge_flows_partial_1128.pq"
    )
    base_scenario_links.rename(
        columns={
            "acc_capacity": "current_capacity",
            "acc_speed": "current_speed",
        },
        inplace=True,
    )
    # paths
    flood_path = (
        raster_path
        / "completed"
        / "12-14_2007 Summer_UK Floods"
        / "May"
        / "Raster"
        / "UK_2007_May_FLSW_RD_5m_4326.tif"
    )
    flood_key = flood_path.stem

    clip_path = (
        raster_path
        / "completed"
        / "12-14_2007 Summer_UK Floods"
        / "May"
        / "Vector"
        / "UK_2007_May_FLSW_VE_5m_4326.shp"
    )

    out_path = (
        base_path.parent
        / "outputs"
        / "disruption_analysis"
        / f"{flood_key}_fld_depth.csv"
    )

    # intersection analysis
    intersections = identify_disrupted_links(
        road_links,
        flood_key,
        flood_path,
        clip_path,
        out_path,
    )

    # for debug
    # flood_key = "UK_2007_May_FLSW"
    # intersections = pd.read_csv(
    #     base_path.parent
    #     / "outputs"
    #     / "disruption_analysis"
    #     / "UK_2007_May_FLSW_fld_depth.csv"
    # )

    intersections = intersections.groupby("id", as_index=False)["flood_depth"].max()

    road_links = road_links.merge(
        intersections[["id", "flood_depth"]], how="left", on="id"
    )
    road_links["flood_depth"] = road_links["flood_depth"].fillna(0.0)

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

    # calculate the maximum speeds on flooded road links
    road_links["max_speed"] = road_links.apply(
        lambda row: compute_maximum_speed_on_flooded_roads(
            row["flood_depth"], row["free_flow_speeds"]
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
                row["flood_depth"],
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
                row["flood_depth"],
            ),
            axis=1,
        )
    else:
        print("Please enter flood type!")

    road_links.to_parquet(base_path / "disruption_analysis_1129" / "road_links.pq")
    road_links.drop(columns="geometry", inplace=True)
    road_links.to_csv(
        base_path / "disruption_analysis_1129" / "road_links.csv", index=False
    )


if __name__ == "__main__":
    main(flood_type="surface")  # "surface"/"river"
