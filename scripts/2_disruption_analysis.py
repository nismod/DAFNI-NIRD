import sys
from typing import Dict
from pathlib import Path

import geopandas as gpd
from collections import defaultdict

from snail import io, intersection
from nird.utils import load_config
import warnings

warnings.filterwarnings("ignore")

base_path = Path(load_config()["paths"]["soge_clusters"])
raster_path = base_path / "hazards" / "JBA_data"


def intersect_features_with_raster(
    raster_path: str,
    raster_key: str,
    features: gpd.GeoDataFrame,
    flood_type: str,
) -> gpd.GeoDataFrame:
    """Intersects vector features with a raster dataset and computes flood depth for
    each intersected feature.

    Parameters
    ----------
    raster_path: str
        Path to the raster file containing flood data.
    raster_key: str
        Identifier for the raster dataset.
    features: gpd.GeoDataFrame
        Vector features (e.g., linestrings).
    flood_type: str
        Type of flood (e.g., "surface" or "river").

    Returns
    -------
    gpd.GeoDataFrame
        Intersected features with flood depth values, reprojected to EPSG:27700.
    """

    print(f"Intersecting features with raster {raster_key}...")
    # read the raster data: depth (meter)
    raster = io.read_raster_band_data(raster_path)

    # run the intersection analysis
    grid, _ = io.read_raster_metadata(raster_path)
    prepared = intersection.prepare_linestrings(features)
    intersections = intersection.split_linestrings(prepared, grid)
    # if intersections is None:
    #     print("Intersection Error: Linestring_features is empty!")
    #     sys.exit()
    intersections = intersection.apply_indices(intersections, grid)
    intersections[f"flood_depth_{flood_type}"] = (
        intersection.get_raster_values_for_splits(intersections, raster)
    )

    # reproject back
    intersections = intersections.to_crs("epsg:27700")
    intersections["length"] = intersections.geometry.length

    return intersections


def clip_features(
    features: gpd.GeoDataFrame,
    clip_path: str,
    raster_key: str,
) -> gpd.GeoDataFrame:
    """Extract features to the raster extent.

    Parameters
    ----------
    features: gpd.GeoDataFrame
        Contains the spatial features to be clipped.
    clip_path: str
        Path to the file containing the clipping features (shapefile).
    raster_key: str
        A descriptive raster key.

    Returns
    -------
    gpd.GeoDataFrame
        Contains the features clipped to the extent of the clip layer.
    """

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


def compute_maximum_speed_on_flooded_roads(
    depth: float,
    free_flow_speed: float,
    threshold=30,
) -> float:
    """Compute the maximum speed on flooded roads based on flood depth.

    Parameters
    ----------
    depth: float
        The depth of floodwater in meters.
    free_flow_speed: float
        The free-flow speed of the road under normal conditions in miles.
    threshold: float, optional
        The depth threshold in centimetres for road closure (default is 30 cm).

    Returns
    -------
    float
        The maximum speed on the flooded road in miles per hour (mph).
    """

    depth = depth * 100  # m to cm
    if depth < threshold:  # cm
        # value = 0.0009 * depth**2 - 0.5529 * depth + 86.9448  # kmph
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
    Computes the damage level of roads based on flood type, road classification,
    and flood depth.

    Parameters
    ----------
    fldType: str
        Type of flood, either "surface" or "river".
    road_classification: str
        Motorway, A Road or B Road.
    trunk_road: bool
        Whether the road is a trunk road (True/False).
    road_label: str
        Label of the road, e.g., road, tunnel, bridge.
    fldDepth: float
        Depth of the flood in metres.

    Returns
    -------
    str:
        Damage level as one of the following categories:
        - "no": No damage
        - "minor": Minor damage
        - "moderate": Moderate damage
        - "extensive": Extensive damage
        - "severe": Severe damage
    """

    depth = fldDepth * 100  # cm
    if fldType == "surface":
        if road_label == "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            if 50 <= depth < 100:
                return "minor"
            elif 100 <= depth < 200:
                return "moderate"
            elif 200 <= depth < 600:
                return "extensive"
            else:
                return "severe"
        elif road_label != "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            if 50 <= depth < 100:
                return "no"
            elif 100 <= depth < 200:
                return "no"
            elif 200 <= depth < 600:
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
        if road_label == "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            if 50 <= depth < 100:
                return "minor"
            elif 100 <= depth < 200:
                return "minor"
            elif 200 <= depth < 600:
                return "moderate"
            else:
                return "extensive"
        elif road_label != "tunnel" and (
            road_classification == "Motorway"
            or (road_classification == "A Road" and trunk_road)
        ):
            if depth < 50:
                return "no"
            if 50 <= depth < 100:
                return "minor"
            elif 100 <= depth < 200:
                return "moderate"
            elif 200 <= depth < 600:
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


def intersections_with_damage(
    road_links: gpd.GeoDataFrame,
    flood_key: str,
    flood_type: str,
    flood_path: str,
    clip_path: str,
) -> gpd.GeoDataFrame:
    """
    Calculate flood depth and damage levels for individual road segments.

    Parameters
    ----------
    road_links: gpd.GeoDataFrame
         Road links with geometries and classifications.
    flood_key: str
        Key identifier for the flood dataset.
    flood_type: str
        Type of flood ("surface" or "river").
    flood_path: str
        Path to the flood raster file.
    clip_path: str
        Path to the vector file used for clipping.

    Returns
    -------
    gpd.GeoDataFrame
        Intersections with calculated flood depths and damage levels.
    """

    # Clip road links with features in the provided vector file
    road_links = road_links.to_crs("epsg:4326")
    clipped_features = clip_features(road_links, clip_path, flood_key)
    if clipped_features.empty:
        print("Warning: Clip features is None!")
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
    Calculate damage levels and maximum flood depth for road links.

    Parameters
    ----------
    features: gpd.GeoDataFrame
        GeoDataFrame of road links.
    intersections: gpd.GeoDataFrame
        GeoDataFrame of intersections with flood data.
    damage_level_dict: Dict
        Mapping of damage levels to numerical values.
    damage_level_dict_reverse: Dict
        Reverse mapping of numerical values to damage levels.

    Returns
    -------
    gpd.GeoDataFrame
        Updated features with maximum flood depth and damage level.
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
        print("Error: flood depth columns are missing!")
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
        print("Error: damage level columns are missing!")

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


def main(depth_thres):
    """Main function

    Model Inputs
    ------
    - edge_flows_32p.gpq: base scenario output
    - GB_road_links_with_bridges.gpq: network element
    - Thames Lloyd's RDS (RASTER): JBA Flood Map
    - Thames Lloyd's RDS (Vector): Clip file

    Model Outputs
    -------
    - intersections_x.pq: feature intersections with flood depth and damage level
    - road_links_x.gpq: with maxmimum flood depth and damage level by aggregating the
        corresponding intersections.
    """

    # base scenario simulation results
    base_scenario_links = gpd.read_parquet(
        base_path.parent / "results" / "base_scenario" / "edge_flows_32p.gpq"
    )
    base_scenario_links.rename(
        columns={
            "acc_capacity": "current_capacity",
            "acc_speed": "current_speed",
            "acc_flow": "current_flow",
        },
        inplace=True,
    )

    # damage level dicts
    damage_level_dict = {
        "no": 0,
        "minor": 1,
        "moderate": 2,
        "extensive": 3,
        "severe": 4,
    }
    damage_level_dict_reverse = {i: k for k, i in damage_level_dict.items()}

    # flood event classification into surface/river flood
    flood_types = ["surface", "river", "both"]
    event_files = {flood_type: [] for flood_type in ["surface", "river"]}
    # Iterate through flood types and process files
    for flood_type in flood_types:
        folder_path = raster_path / flood_type
        if folder_path.exists():
            for raster_dir in folder_path.rglob(
                "Raster"
            ):  # Search for "Raster" directories
                for tif_file in raster_dir.rglob(
                    "*.tif"
                ):  # Find .tif files recursively
                    # Filter files with "RD" in the name and exclude those with "IE"
                    if "RD" in tif_file.name and "IE" not in tif_file.name:
                        if flood_type == "both":
                            if "FLSW" in tif_file.name:
                                target_flood_type = "surface"
                            elif "FLRF" in tif_file.name:
                                target_flood_type = "river"
                            else:
                                continue
                        else:
                            target_flood_type = flood_type

                        # Append the file path to the appropriate flood_type list
                        event_files[target_flood_type].append(str(tif_file))

    event_dict = defaultdict(lambda: defaultdict(list))
    for flood_type, list_of_events in event_files.items():
        for event_path in list_of_events:
            # event_key = "_".join(
            #     Path(event_path).stem.split("_")[:2]
            # )  # rename events under "both" category
            # event_key = event_path.split("\\")[6].split("_")[0]
            event_key = Path(event_path).parts[9].split("_")[0]
            event_dict[event_key][flood_type].append(event_path)
    print(event_dict)
    # analysis
    for flood_key, v in event_dict.items():
        # road links
        road_links = gpd.read_parquet(
            base_path / "networks" / "GB_road_links_with_bridges.gpq"
        )

        # out path
        out_path = (
            base_path.parent / "results" / "disruption_analysis" / str(depth_thres)
        )

        intersections = gpd.GeoDataFrame(
            columns=["e_id", "length", "index_i", "index_j"]
        )

        for flood_type, flood_paths in v.items():
            for flood_path in flood_paths:
                # clip path
                clip_path = Path(
                    flood_path.replace("Raster", "Vector").replace(".tif", ".shp")
                )
                clip_path1 = clip_path.with_name(clip_path.name.replace("_RD_", "_VE_"))
                clip_path2 = clip_path.with_name(clip_path.name.replace("_RD_", "_PR_"))
                if clip_path1.exists():
                    clip_path = clip_path1
                elif clip_path2.exists():
                    clip_path = clip_path2
                else:
                    print(f"Cannot find vector file for: {flood_path}")
                    # breakpoint()
                    continue  # Skip further processing for this file

                # intersections
                temp_file = intersections_with_damage(
                    road_links, flood_key, flood_type, flood_path, clip_path
                )
                if temp_file is None:
                    # breakpoint()
                    continue
                intersections = intersections.merge(
                    temp_file[
                        [
                            "e_id",
                            "length",
                            "index_i",
                            "index_j",
                            f"flood_depth_{flood_type}",
                            f"damage_level_{flood_type}",
                        ]
                    ],
                    on=["e_id", "length", "index_i", "index_j"],
                    how="outer",
                )

        # save intersectiosn for damage analysis
        if intersections.empty:
            print("Warning: intersections result is empty!")
            # breakpoint()
            continue
        intersections.to_parquet(out_path / f"intersections_{flood_key}.pq")

        # road integrations
        road_links = features_with_damage(
            road_links,
            intersections,
            damage_level_dict,
            damage_level_dict_reverse,
        )

        # max_speed estimation
        """
        Uncertainties of flood depth threshold for road closure (cm): 15, 30, 60
        """
        # attach capacity and speed info on D-0
        road_links = road_links.merge(
            base_scenario_links[
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

        # compute maximum speed restriction on individual road links
        road_links["max_speed"] = road_links.apply(
            lambda row: compute_maximum_speed_on_flooded_roads(
                row["flood_depth_max"],
                row["free_flow_speeds"],
                threshold=depth_thres,
            ),
            axis=1,
        )
        road_links.to_parquet(out_path / f"road_links_{flood_key}.gpq")


if __name__ == "__main__":
    try:
        depth_thres = int(sys.argv[1])
    except (IndexError, ValueError):
        print(
            "Error: Please provide the flood depth for road closure (e.g., 30 or "
            "60 cm)!"
        )
        sys.exit(1)
    main(depth_thres)
