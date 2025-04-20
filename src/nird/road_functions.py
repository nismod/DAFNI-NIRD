import sys
import json

from typing import Dict, Tuple
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

from collections import defaultdict
import geopandas as gpd

from snail import io, intersection
from snail import damages

import warnings

warnings.filterwarnings("ignore")


# %%
# for disruption analysis
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
    # damage_level_dict: Dict,
    # damage_level_dict_reverse: Dict,
) -> gpd.GeoDataFrame:
    """
    Calculate damage levels and maximum flood depth for road links.

    Parameters
    ----------
    features: gpd.GeoDataFrame
        GeoDataFrame of road links.
    intersections: gpd.GeoDataFrame
        GeoDataFrame of intersections with flood data.

    Returns
    -------
    gpd.GeoDataFrame
        Updated features with maximum flood depth and damage level.
    """
    damage_level_dict = {
        "no": 0,
        "minor": 1,
        "moderate": 2,
        "extensive": 3,
        "severe": 4,
    }
    damage_level_dict_reverse = {i: k for k, i in damage_level_dict.items()}

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


# %%
# for damage analysis
def create_damage_curves(damage_ratio_df: pd.DataFrame) -> Dict:
    """Create a dictionary of piecewise linear damage curves for various road
    classifications and flow conditions based on damage ratio data.

    Parameters
    ----------
    damage_ratio_df: pd.DataFrame
        A sample of flood depths and their corresponding road flood damage ratios.

    Returns
    -------
    Dict
        A dictionary containing damage curves for different road and flow conditions.

    Notes:
    C1: motorways & trunk roads, sophisiticated accessories, low flow
    C2: motorways & trunk roads, sophisiticated accessories, high flow
    C3: motorways & trunk roads, non sophisticated accessories, low flow
    C4: motorways & trunk roads, non sophisticated accessories, high flow
    C5: other roads, low flow
    C6: other roads, high flow
    """

    list_of_damage_curves = []
    cols = damage_ratio_df.columns[1:]
    for col in cols:
        damage_ratios = damage_ratio_df[["intensity", col]]
        damage_ratios.rename(columns={col: "damage"}, inplace=True)
        damage_curve = damages.PiecewiseLinearDamageCurve(damage_ratios)
        list_of_damage_curves.append(damage_curve)

    damage_curve_dict = defaultdict()
    keys = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for idx in range(len(keys)):
        key = keys[idx]
        damage_curve = list_of_damage_curves[idx]
        damage_curve_dict[key] = damage_curve

    return damage_curve_dict


def compute_damage_fraction(
    road_classification: str,
    trunk_road: bool,
    road_label: str,
    flood_depth: float,
    damage_curves: Dict,
) -> Tuple[str, float, str, float]:
    """Compute the damage fraction for a road asset based on its classification,
    label, and flood depth.

    Parameters
    ----------
    road_classification : str
        The classification of the road, such as "Motorway", "A Road", or "B Road".
    trunk_road : bool
        Specifies whether the road is a trunk road (True) or not (False).
    road_label : str
        The type of infrastructure, such as "bridge", "tunnel", or "road".
    flood_depth : float
        The depth of floodwater on the road asset in meters.
    damage_curves : Dict
        A dictionary containing damage curves for different road and flow conditions.

    Returns
    -------
    Tuple[str, float, str, float]
        A tuple containing:
        - The first damage curve label (e.g., "C1", "C3", or "C5").
        - The computed damage fraction from the first curve.
        - The second damage curve label (e.g., "C2", "C4", or "C6").
        - The computed damage fraction from the second curve.
    """

    if road_label == "tunnel" and (
        road_classification == "Motorway"
        or (road_classification == "A Road" and trunk_road)
    ):
        C1_damage_fraction = damage_curves["C1"].damage_fraction(flood_depth)
        C2_damage_fraction = damage_curves["C2"].damage_fraction(flood_depth)
        return ("C1", C1_damage_fraction, "C2", C2_damage_fraction)

    elif road_label != "tunnel" and (
        road_classification == "Motorway"
        or (road_classification == "A Road" and trunk_road)
    ):
        C3_damage_fraction = damage_curves["C3"].damage_fraction(flood_depth)
        C4_damage_fraction = damage_curves["C4"].damage_fraction(flood_depth)
        return ("C3", C3_damage_fraction, "C4", C4_damage_fraction)

    else:
        C5_damage_fraction = damage_curves["C5"].damage_fraction(flood_depth)
        C6_damage_fraction = damage_curves["C6"].damage_fraction(flood_depth)
        return ("C5", C5_damage_fraction, "C6", C6_damage_fraction)


def compute_damage_values(
    length: float,  # in meters
    flood_type: str,
    damage_fraction: float,
    road_classification: str,
    form_of_way: str,
    urban: int,
    lanes: int,
    road_label: str,
    damage_level: str,
    damage_values: Dict,  # million £/unit
    bridge_width=None,
) -> Tuple[float, float, float]:
    """
    Calculate the damage costs (minimum, maximum, and mean) for different types of road
    infrastructure (bridges, tunnels, and ordinary roads) caused by flooding.

    Parameters:
    ----------
    length : float
        The length of the infrastructure (in meters).
    flood_type : str
        The type of flood (e.g., "river", "surface").
    damage_fraction : float
        The fraction of damage to the infrastructure (e.g., 0.5 for 50% damage).
    road_classification : str
        The classification of the road (e.g., "Motorway", "A Road", "B Road").
    form_of_way : str
        The configuration of the road (e.g., "Single Carriageway", "Dual Carriageway").
    urban : int
        Binary indicator for urban (1) or suburban (0) areas.
    lanes : int
        The number of lanes on the road.
    road_label : str
        The type of infrastructure ("bridge", "tunnel", or "road").
    damage_level : str
        The severity level of damage (e.g., "minor", "major", "catastrophic").
    damage_values : dict
        A nested dictionary containing damage cost values (in million £/unit) for
        different infrastructure types,
        flood types, and damage levels. The dictionary should have the structure:
        {
            "bridge_flood_type": {"damage_level": {"min": value, "max": value}},
            "tunnel": {"specific_key": {"min": value, "max": value}},
            "road": {"specific_key": {"min": value, "max": value}}
        }.
    bridge_width : float, optional
        The width of the bridge (in meters). Required if `road_label` is "bridge".

    Returns:
    -------
    Tuple [float, float, float]
        A tuple containing:
        - min_cost (float): The minimum estimated damage cost.
        - max_cost (float): The maximum estimated damage cost.
        - mean_cost (float): The mean estimated damage cost.
    """

    def compute_bridge_damage(length, width, flood_type, damage_level):
        """Calculate min and max damage for bridges."""
        min_damage = (
            width * length * damage_values[f"bridge_{flood_type}"][damage_level]["min"]
        )
        max_damage = (
            width * length * damage_values[f"bridge_{flood_type}"][damage_level]["max"]
        )
        return min_damage, max_damage

    def compute_tunnel_or_road_damage(length, lanes, key1, key2, damage_fraction):
        """Calculate min and max damage for tunnels or roads."""
        min_damage = (
            length * 1e-3 * lanes * damage_values[key1][key2]["min"] * damage_fraction
        )
        max_damage = (
            length * 1e-3 * lanes * damage_values[key1][key2]["max"] * damage_fraction
        )
        return min_damage, max_damage

    if road_label == "bridge":
        if bridge_width is None:
            raise ValueError("Bridge width is required for bridges!")
        min_cost, max_cost = compute_bridge_damage(
            length, bridge_width, flood_type, damage_level
        )
    elif road_label in ["tunnel", "road"]:
        urban_key = "urb" if urban == 1 else "sub"
        if road_classification == "Motorway":
            lane_key = "ge8" if lanes >= 8 else "lt8"
            key = f"m_{lane_key}_{urban_key}"
        elif form_of_way == "Single Carriageway":
            road_type = "a" if road_classification == "A Road" else "b"
            key = f"{road_type}single_{urban_key}"
        else:  # Dual Carriageway
            lane_key = "ge6" if lanes >= 6 else "lt6"
            key = f"abdual_{lane_key}_{urban_key}"

        min_cost, max_cost = compute_tunnel_or_road_damage(
            length, lanes, road_label, key, damage_fraction
        )
    else:
        raise ValueError("Invalid road_label. Must be 'bridge', 'tunnel', or 'road'.")

    mean_cost = np.mean([min_cost, max_cost])

    return min_cost, max_cost, mean_cost


def calculate_damage(
    disrupted_links: pd.DataFrame,
    damage_curves: Dict,
    damage_values: Dict,
) -> pd.DataFrame:
    """
    Calculate damage fractions and costs for disrupted road links based on flood depth.

    Parameters
    ----------
    disrupted_links: pd.DataFrame
        A DataFrame containing disrupted road links and their attributes with required
        columns such as "flood_depth_surface" and "flood_depth_river".
    damage_curves: Dict
        A dictionary containing damage curves.
    damage_values: Dict
        A dictionary containing damage cost values for different damage levels,
        road asset types, and flood types.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with calculated damage fractions and costs. Includes columns
        for damage fractions and cost stats (min, max, mean).
    """

    # Validate required columns
    required_columns = {"flood_depth_surface", "flood_depth_river"}
    missing_columns = required_columns - set(disrupted_links.columns)
    assert not missing_columns, f"Missing required columns: {missing_columns}"

    # Define damage categories and flood types
    curves = ["C1", "C2", "C3", "C4", "C5", "C6"]
    flood_types = ["surface", "river"]

    # Add default columns using MultiIndex for efficiency
    col_tuples = [
        (f"{curve}_{flood_type}_damage_fraction", np.nan)
        for curve in curves
        for flood_type in flood_types
    ] + [
        (f"{curve}_{flood_type}_damage_value_{stat}", np.nan)
        for curve in curves
        for flood_type in flood_types
        for stat in ["min", "max", "mean"]
    ]
    for col, default in col_tuples:
        disrupted_links[col] = default

    def calculate_for_row(row, flood_type):
        """
        Helper function to calculate damage fractions and costs for a single row.
        """
        # Compute damage fractions
        curve1, damage_fraction1, curve2, damage_fraction2 = compute_damage_fraction(
            row.road_classification,
            row.trunk_road,
            row.road_label,
            row[f"flood_depth_{flood_type}"],
            damage_curves,
        )

        # Compute damage values for both curves
        damage_values_1 = compute_damage_values(
            row.length,
            flood_type,
            damage_fraction1,
            row.road_classification,
            row.form_of_way,
            row.urban,
            row.lanes,
            row.road_label,
            row[f"damage_level_{flood_type}"],
            damage_values,
            row.averageWidth,
        )
        damage_values_2 = compute_damage_values(
            row.length,
            flood_type,
            damage_fraction2,
            row.road_classification,
            row.form_of_way,
            row.urban,
            row.lanes,
            row.road_label,
            row[f"damage_level_{flood_type}"],
            damage_values,
            row.averageWidth,
        )
        if row.road_label == "bridge":
            unit_cost_min = damage_values[f"bridge_{flood_type}"][
                f"{row[f'damage_level_{flood_type}']}"
            ]["min"]
            unit_cost_max = damage_values[f"bridge_{flood_type}"][
                f"{row[f'damage_level_{flood_type}']}"
            ]["max"]
        else:
            unit_cost_min = damage_values[f"{row.road_label}"][
                f"{row[f'damage_level_{flood_type}']}"
            ]["min"]
            unit_cost_max = damage_values[f"{row.road_label}"][
                f"{row[f'damage_level_{flood_type}']}"
            ]["max"]
        # Return a dictionary of results for easier assignment
        return {
            f"{curve1}_{flood_type}_damage_fraction": damage_fraction1,
            f"{curve2}_{flood_type}_damage_fraction": damage_fraction2,
            f"{curve1}_{flood_type}_damage_value_min": damage_values_1[0],
            f"{curve1}_{flood_type}_damage_value_max": damage_values_1[1],
            f"{curve1}_{flood_type}_damage_value_mean": damage_values_1[2],
            f"{curve2}_{flood_type}_damage_value_min": damage_values_2[0],
            f"{curve2}_{flood_type}_damage_value_max": damage_values_2[1],
            f"{curve2}_{flood_type}_damage_value_mean": damage_values_2[2],
            f"{flood_type}_unit_cost_min": unit_cost_min,
            f"{flood_type}_unit_cost_max": unit_cost_max,
        }

    # Apply calculation for each flood type
    for flood_type in flood_types:
        flood_results = disrupted_links.apply(
            lambda row: calculate_for_row(row, flood_type), axis=1
        )
        flood_results_df = pd.DataFrame(
            list(flood_results)
        )  # Convert results to DataFrame
        disrupted_links.update(
            flood_results_df
        )  # Update disrupted_links with the calculated values

    return disrupted_links


def format_intersections(
    intersections: pd.DataFrame,
    road_links: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Format intersection data and enrich it with road link attributes.

    Parameters
    ----------
    intersections: pd.DataFrame
        A DataFrame containing intersection results (module 2), such as
        flood depth and damage level based on road intersections.
    road_links: gpd.GeoDataFrame
        Original road links of the network.

    Returns
    -------
    pd.DataFrame
        A formatted and enriched DataFrame of intersections.
    """

    # Define default values for missing columns
    columns_to_add = {
        "flood_depth_surface": 0.0,
        "flood_depth_river": 0.0,
        "damage_level_surface": "no",
        "damage_level_river": "no",
    }

    # Define mappings for damage levels
    damage_level_dict = {
        "no": 0,
        "minor": 1,
        "moderate": 2,
        "extensive": 3,
        "severe": 4,
    }
    damage_level_dict_reverse = {v: k for k, v in damage_level_dict.items()}

    # Ensure all required columns exist with default values
    for col, default_value in columns_to_add.items():
        if col not in intersections.columns:
            intersections[col] = default_value
        else:
            intersections[col].fillna(default_value, inplace=True)

    # Map damage levels to numeric values
    for col in ["damage_level_surface", "damage_level_river"]:
        intersections[col] = intersections[col].map(damage_level_dict)

    # Group by specific columns and take the max value for each group
    group_columns = ["e_id", "length", "index_i", "index_j"]
    intersections_gp = intersections.groupby(group_columns, as_index=False).max(
        numeric_only=True
    )  # Ensure numeric columns are aggregated

    # Reverse map numeric damage levels back to strings
    for col in ["damage_level_surface", "damage_level_river"]:
        intersections_gp[col] = intersections_gp[col].map(damage_level_dict_reverse)
    intersections_gp = intersections_gp.merge(
        road_links[
            [
                "e_id",
                "road_classification",
                "form_of_way",
                "trunk_road",
                "urban",
                "lanes",
                "averageWidth",
                "road_label",
            ]
        ],
        on="e_id",
        how="left",
    )
    return intersections_gp


# %%
# for recovery and rerouting analysis
def have_common_items(list1: List, list2: List) -> bool:
    common_items = set(list1) & set(list2)
    return common_items


def bridge_recovery(
    day: int,
    damage_level: str,
    designed_capacity: float,
    current_capacity: float,
    bridge_recovery_dict: Dict,
) -> Tuple[float, float]:
    """Compute the daily recovery of bridge capacity and speed based on
    damage level and recovery rate."""

    if day == 0:
        if damage_level in ["extensive", "severe"]:
            current_capacity = 0
    else:
        if damage_level != "no":  # ["minor", "moderate", "extensive", "severe"]
            recovery_rate = bridge_recovery_dict.get(damage_level, [])[day]
            current_capacity = max(designed_capacity * recovery_rate, current_capacity)
    return current_capacity


def ordinary_road_recovery(
    day: int,
    damage_level: str,
    designed_capacity: float,
    current_capacity: float,
    road_recovery_dict: Dict,
) -> Tuple[float, float]:
    """Compute the daily recovery of ordinary road capacity and speed based on
    damage level and recovery rates."""

    if day == 0:  # the occurance of damage
        if damage_level in ["extensive", "severe"]:
            current_capacity = 0
    elif 0 < day <= 2:
        if damage_level != "no":  # ["minor", "moderate", "extensive", "severe"]
            recovery_rate = road_recovery_dict.get(damage_level, [])[day]
            current_capacity = max(designed_capacity * recovery_rate, current_capacity)
    else:
        pass
    return current_capacity


def load_recovery_dicts(base_path: Path) -> Tuple[Dict, Dict]:
    """Load recovery rates for bridges and ordinary roads."""
    bridge_recovery_dict = {}
    for level in ["minor", "moderate", "extensive", "severe"]:
        with open(base_path / "parameters" / f"capt_{level}.json", "r") as f:
            rates = json.load(f)
        rates.insert(0, 0.0)  # day-0
        bridge_recovery_dict[level] = rates

    road_recovery_dict = {
        "minor": [0.0, 1.0, 1.0],
        "moderate": [0.0, 1.0, 1.0],
        "extensive": [0.0, 1.0, 1.0],
        "severe": [0.0, 0.5, 1.0],
    }
    return bridge_recovery_dict, road_recovery_dict
