from pathlib import Path
import warnings
from collections import defaultdict
from typing import Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

from snail import damages
import matplotlib.pyplot as plt
import contextily as ctx

warnings.simplefilter("ignore")

base_path = Path("/data/inputs")


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
    damage_values: float,  # million £/unit
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


def main():
    """Main function

    Model Inputs
    ------------
    - damage_ratio_road_flood.xlsx: damage curves
    - damage_cost_road_flood_uk.xlsx: asset damage values
    - GB_road_links_with_bridges.gpq: network links
    - intersections: module 2 output.

    Outputs
    -------
    - pd.DataFrame:
        intersections_with_damages (min, max, mean)
    """

    # damage curves
    damages_ratio_df = pd.read_excel(
        base_path / "tables" / "damage_ratio_road_flood.xlsx"
    )
    damage_curves = create_damage_curves(damages_ratio_df)

    # damage values (updated with UK values)
    road_links = gpd.read_parquet(
        base_path / "networks" / "GB_road_links_with_bridges.gpq"
    )
    road_damage_file = pd.read_excel(
        base_path / "tables" / "damage_cost_road_flood_uk.xlsx",
        sheet_name="roads",
    )
    tunnel_damage_file = pd.read_excel(
        base_path / "tables" / "damage_cost_road_flood_uk.xlsx",
        sheet_name="tunnels",
    )
    bridge_surface_damage_file = pd.read_excel(
        base_path / "tables" / "damage_cost_road_flood_uk.xlsx",
        sheet_name="bridges-surface",
    )
    bridge_river_damage_file = pd.read_excel(
        base_path / "tables" / "damage_cost_road_flood_uk.xlsx",
        sheet_name="bridges-river",
    )
    dv_road_dict = defaultdict(lambda: defaultdict(float))
    for row in road_damage_file.itertuples():
        dv_road_dict[row.label]["min"] = row.Min
        dv_road_dict[row.label]["max"] = row.Max
        dv_road_dict[row.label]["mean"] = row.Mean

    dv_tunnel_dict = defaultdict(lambda: defaultdict(float))
    for row in tunnel_damage_file.itertuples():
        dv_tunnel_dict[row.label]["min"] = row.Min
        dv_tunnel_dict[row.label]["max"] = row.Max
        dv_tunnel_dict[row.label]["mean"] = row.Mean

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
    print("Damage calculation starts...")
    # format intersections
    intersections = pd.read_parquet(base_path / "intersections" / "intersections_17.pq")
    intersections = format_intersections(intersections, road_links)

    # run damage analysis
    intersections_with_damage = calculate_damage(
        intersections, damage_curves, damage_values
    )

    # aggregate damage values for road links
    # attributes
    intersections_with_damage_gp1 = (
        intersections_with_damage[
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
        ]
        .groupby(by=["e_id"], as_index=False)
        .first()
    )
    # numeric values
    intersections_with_damage_gp2 = (
        pd.concat(
            [
                intersections_with_damage["e_id"],
                intersections_with_damage["length"],
                intersections_with_damage.iloc[:, -48:],
            ],
            axis=1,
        )
        .fillna(0)
        .groupby(by=["e_id"], as_index=False)
        .sum()  # sum up damage values of all segments -> disrupted links
    )

    # merge attributes and numeric values
    common_cols = intersections_with_damage_gp1.columns.intersection(
        intersections_with_damage_gp2.columns
    ).tolist()
    intersections_with_damage_final = intersections_with_damage_gp1.merge(
        intersections_with_damage_gp2, on=common_cols, how="left"
    )

    # modify the value of damage fractions to be no greater than 1.0
    fraction_cols = intersections_with_damage_final.filter(like="fraction").columns
    intersections_with_damage_final[fraction_cols] = intersections_with_damage_final[
        fraction_cols
    ].clip(upper=1.0)

    # export results
    intersections_with_damage_final.to_csv(
        base_path.parent / "outputs" / "intersections17_with_damages_updated.csv",
        index=False,
    )

    # create map for visualisation
    intersections_with_damage_final = intersections_with_damage_final[
        intersections_with_damage_final.C6_river_damage_value_mean > 0
    ]
    intersections_with_damage_final = intersections_with_damage_final[
        ["e_id"]
        + intersections_with_damage_final.filter(like="damage_value").columns.tolist()
    ]
    road_links_merge = road_links.merge(
        intersections_with_damage_final, on="e_id", how="left"
    )
    road_links_merge = road_links_merge[road_links_merge.C6_river_damage_value_mean > 0]
    road_links_merge.reset_index(drop=True, inplace=True)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the road links with damage values
    road_links_merge.plot(
        column="C6_river_damage_value_mean",
        cmap="terrain",
        legend=True,
        ax=ax,
        legend_kwds={
            "label": "Damage Value (£ million)",
            "shrink": 0.5,  # Adjusts the size of the colourbar
        },
    )

    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,
        crs=road_links_merge.crs.to_string(),  # Ensure CRS matches
    )

    ax.set_title(
        "Average Road Damage from River Flooding (C6 Damage Curve)",
        fontsize=11,
        fontweight="bold",
        pad=20,
    )
    ax.set_axis_off()
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        base_path.parent / "outputs" / "road_damage_map.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
