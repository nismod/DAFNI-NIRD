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


def create_damage_curves(damage_ratio_df):
    """Create a dictionary of piecewise linear damage curves for various road
    classifications and flow conditions based on damage ratio data."""
    list_of_damage_curves = []
    cols = damage_ratio_df.columns[1:]
    for col in cols:
        damage_ratios = damage_ratio_df[["intensity", col]]
        damage_ratios.rename(columns={col: "damage"}, inplace=True)
        damage_curve = damages.PiecewiseLinearDamageCurve(damage_ratios)
        list_of_damage_curves.append(damage_curve)

    damage_curve_dict = defaultdict()

    """
    C1: motorways & trunk roads, sophisiticated accessories, low flow
    C2: motorways & trunk roads, sophisiticated accessories, high flow
    C3: motorways & trunk roads, non sophisticated accessories, low flow
    C4: motorways & trunk roads, non sophisticated accessories, high flow
    C5: other roads, low flow
    C6: other roads, high flow
    """
    keys = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for idx in range(len(keys)):
        key = keys[idx]
        damage_curve = list_of_damage_curves[idx]
        damage_curve_dict[key] = damage_curve

    return damage_curve_dict


def compute_damage_fraction(
    road_classification,
    trunk_road,
    road_label,  # [tunnel, bridge, road]
    flood_depth,  # meter
    damage_curves,
):
    """Compute the damage fraction for a road asset based on its classification,
    label, and flood depth."""

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
    length,  # in meters
    flood_type,
    damage_fraction,
    road_classification,
    form_of_way,
    urban,
    lanes,
    road_label,  # ["tunnel", "bridge", "road"]
    damage_level,
    damage_values,  # million £/unit
    bridge_width=None,
):
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
    tuple[float, float, float]
        A tuple containing:
        - min_cost (float): The minimum estimated damage cost.
        - max_cost (float): The maximum estimated damage cost.
        - mean_cost (float): The mean estimated damage cost.

    Raises:
    -------
    AssertionError
        If `road_label` is "bridge" but `bridge_width` is not provided.
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


def calculate_damage(disrupted_links, damage_curves, damage_values):
    """
    Calculate damage fractions and costs for disrupted road links based on flood depth.

    Parameters:
        disrupted_links: DataFrame of disrupted road links with necessary attributes.
        damage_curves: Dictionary containing damage curves.
        damage_values: Dictionary containing damage cost values.

    Returns:
        pd.DataFrame: Updated DataFrame with calculated damage fractions and costs.
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
            row.geometry.length,
            flood_type,
            damage_fraction1,
            row.road_classification,
            row.form_of_way,
            row.urban,
            row.lanes,
            row.road_label,
            row[f"damage_level_{flood_type}"],
            damage_values,
            row.aveBridgeWidth,
        )
        damage_values_2 = compute_damage_values(
            row.geometry.length,
            flood_type,
            damage_fraction2,
            row.road_classification,
            row.form_of_way,
            row.urban,
            row.lanes,
            row.road_label,
            row[f"damage_level_{flood_type}"],
            damage_values,
            row.aveBridgeWidth,
        )

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


def main():
    # damage curves
    """
    4 damage curvse for M and A roads
    2 damage curvse for B roads
    """
    damages_ratio_df = pd.read_excel(
        base_path / "disruption_analysis_1129" / "damage_ratio_road_flood.xlsx"
    )
    damage_curves = create_damage_curves(damages_ratio_df)

    # damage values (updated with UK values)
    """
    asset types, number of lanes, flood types
    min, max, mean damage values
    """
    road_damage_file = pd.read_excel(
        base_path / "tables" / "damage_values.xlsx", sheet_name="roads"
    )
    tunnel_damage_file = pd.read_excel(
        base_path / "tables" / "damage_values.xlsx", sheet_name="tunnels"
    )
    bridge_surface_damage_file = pd.read_excel(
        base_path / "tables" / "damage_values.xlsx", sheet_name="bridges-surface"
    )
    bridge_river_damage_file = pd.read_excel(
        base_path / "tables" / "damage_values.xlsx", sheet_name="bridges-river"
    )
    dv_road_dict = defaultdict(lambda: defaultdict(float))
    for row in road_damage_file.itertuples():
        dv_road_dict[row.label]["min"] = row.min
        dv_road_dict[row.label]["max"] = row.max
        dv_road_dict[row.label]["mean"] = row.mean

    dv_tunnel_dict = defaultdict(lambda: defaultdict(float))
    for row in tunnel_damage_file.itertuples():
        dv_tunnel_dict[row.label]["min"] = row.min
        dv_tunnel_dict[row.label]["max"] = row.max
        dv_tunnel_dict[row.label]["mean"] = row.mean

    dv_bridge_surface_dict = defaultdict(lambda: defaultdict(float))
    for row in bridge_surface_damage_file.itertuples():
        dv_bridge_surface_dict[row.label]["min"] = row.min
        dv_bridge_surface_dict[row.label]["max"] = row.max
        dv_bridge_surface_dict[row.label]["mean"] = row.mean

    dv_bridge_river_dict = defaultdict(lambda: defaultdict(float))
    for row in bridge_river_damage_file.itertuples():
        dv_bridge_river_dict[row.label]["min"] = row.min
        dv_bridge_river_dict[row.label]["max"] = row.max
        dv_bridge_river_dict[row.label]["mean"] = row.mean

    damage_values = {
        "road": dv_road_dict,
        "tunnel": dv_tunnel_dict,
        "bridge_surface": dv_bridge_surface_dict,
        "bridge_river": dv_bridge_river_dict,
    }

    # Load intersection data and assign attributes for damage calculations
    intersections = gpd.read_parquet(
        base_path.parent
        / "outputs"
        / "disruption_analysis_1129"
        / "UK_2007_May_FLSW_RD_5m_4326_fld_depth.geopq"
    )
    intersections.rename(
        columns={
            "flood_depth": "flood_depth_surface",
            "damage_level": "damage_level_surface",
        },
        inplace=True,
    )
    intersections["flood_depth_river"] = 0
    intersections["damage_level_river"] = "no"
    intersections["road_label"] = "road"
    intersections.loc[intersections.hasTunnel == 1, "road_label"] = "tunnel"
    intersections.loc[intersections.aveBridgeWidth > 0, "road_label"] == "bridge"

    # run damage analysis
    intersections_with_damage = calculate_damage(
        intersections, damage_curves, damage_values
    )
    intersections_with_damage = (
        pd.concat(
            [intersections_with_damage["id"], intersections_with_damage.iloc[:, -48:]],
            axis=1,
        )
        .fillna(0)
        .groupby(by=["id"], as_index=False)
        .sum()  # sum up damage values of all segments -> disrupted links
    )

    # export results
    intersections_with_damage.to_csv(
        base_path.parent
        / "outputs"
        / "disruption_analysis_1129"
        / "disrupted_links_with_damage_values.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
