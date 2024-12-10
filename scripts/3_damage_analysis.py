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
    hasTunnel,
    flood_depth,  # meter
    damage_curves,
):
    # print("Computing damage fractions...")
    if hasTunnel == 1 and (
        road_classification == "Motorway"
        or (road_classification == "A Road" and trunk_road)
    ):
        C1_damage_fraction = damage_curves["C1"].damage_fraction(flood_depth)
        C2_damage_fraction = damage_curves["C2"].damage_fraction(flood_depth)
        return ("C1", C1_damage_fraction, "C2", C2_damage_fraction)

    elif hasTunnel == 0 and (
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
    length,  # meter
    flood_type,
    damage_fraction,
    road_classification,
    form_of_way,
    lanes,
    hasTunnel,
    aveBridgeWidth,
    damage_level,
    damage_values,  # million £/unit
):
    # print("Compute damage values...")

    # initialise values
    mins = []
    maxs = []
    # bridges
    if aveBridgeWidth > 0:
        min_bridge = (
            aveBridgeWidth
            * length
            * damage_values[f"bridge_{flood_type}"][damage_level]["min"]
        )
        max_bridge = (
            aveBridgeWidth
            * length
            * damage_values[f"bridge_{flood_type}"][damage_level]["max"]
        )
        mins.append(min_bridge)
        maxs.append(max_bridge)
        # mean_bridge = (
        #     aveBridgeWidth
        #     * length
        #     * damage_values[f"bridge_{flood_type}"][damage_level]["mean"]
        # )

    # tunnels
    if hasTunnel == 1:
        if road_classification == "Motorway":
            if lanes >= 8:
                min_tunnel = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["tunnel"]["m_lanes_ge8"]["min"]
                    * damage_fraction
                )
                max_tunnel = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["tunnel"]["m_lanes_ge8"]["max"]
                    * damage_fraction
                )
                # mean_tunnel = (
                #     length
                #     * 1e-3
                #     * lanes
                #     * damage_values["tunnel"]["m_lanes_ge8"]["mean"]
                #     * damage_fraction
                # )
                mins.append(min_tunnel)
                maxs.append(max_tunnel)

            else:
                min_tunnel = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["tunnel"]["m_lanes_lt8"]["min"]
                    * damage_fraction
                )
                max_tunnel = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["tunnel"]["m_lanes_lt8"]["max"]
                    * damage_fraction
                )
                mins.append(min_tunnel)
                maxs.append(max_tunnel)
                # mean_tunnel = (
                #     length
                #     * 1e-3
                #     * lanes
                #     * damage_values["tunnel"]["m_lanes_lt8"]["mean"]
                #     * damage_fraction
                # )

        else:
            if form_of_way == "Single Carriageway":
                if road_classification == "A Road":
                    min_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["a_single"]["min"]
                        * damage_fraction
                    )
                    max_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["a_single"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_tunnel)
                    maxs.append(max_tunnel)
                    # mean_tunnel = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["tunnel"]["a_single"]["mean"]
                    #     * damage_fraction
                    # )
                else:  # road_classification == "B Road"
                    min_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["b_single"]["min"]
                        * damage_fraction
                    )
                    max_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["b_single"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_tunnel)
                    maxs.append(max_tunnel)
                    # mean_tunnel = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["tunnel"]["b_single"]["mean"]
                    #     * damage_fraction
                    # )

            else:
                if lanes >= 6:
                    min_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["ab_dual_lanes_ge6"]["min"]
                        * damage_fraction
                    )
                    max_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["ab_dual_lanes_ge6"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_tunnel)
                    maxs.append(max_tunnel)
                    # mean_tunnel = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["tunnel"]["ab_dual_lanes_ge6"]["mean"]
                    #     * damage_fraction
                    # )
                else:  # lanes < 6
                    min_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["ab_dual_lanes_lt6"]["min"]
                        * damage_fraction
                    )
                    max_tunnel = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["tunnel"]["ab_dual_lanes_lt6"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_tunnel)
                    maxs.append(max_tunnel)
                    # mean_tunnel = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["tunnel"]["ab_dual_lanes_lt6"]["mean"]
                    #     * damage_fraction
                    # )

    # ordinary roads:
    if hasTunnel == 0 and aveBridgeWidth == 0:
        if road_classification == "Motorway":
            if lanes >= 8:
                min_road = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["road"]["m_lanes_ge8"]["min"]
                    * damage_fraction
                )
                max_road = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["road"]["m_lanes_ge8"]["max"]
                    * damage_fraction
                )
                mins.append(min_road)
                maxs.append(max_road)
                # mean_road = (
                #     length
                #     * 1e-3
                #     * lanes
                #     * damage_values["road"]["m_lanes_ge8"]["mean"]
                #     * damage_fraction
                # )
            else:
                min_road = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["road"]["m_lanes_lt8"]["min"]
                    * damage_fraction
                )
                max_road = (
                    length
                    * 1e-3
                    * lanes
                    * damage_values["road"]["m_lanes_lt8"]["max"]
                    * damage_fraction
                )
                mins.append(min_road)
                maxs.append(max_road)
                # mean_road = (
                #     length
                #     * 1e-3
                #     * lanes
                #     * damage_values["road"]["m_lanes_lt8"]["mean"]
                #     * damage_fraction
                # )

        else:
            if form_of_way == "Single Carriageway":
                if road_classification == "A Road":
                    min_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["a_single"]["min"]
                        * damage_fraction
                    )
                    max_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["a_single"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_road)
                    maxs.append(max_road)
                    # mean_road = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["road"]["a_single"]["mean"]
                    #     * damage_fraction
                    # )
                else:  # road_classification == "B Road"
                    min_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["b_single"]["min"]
                        * damage_fraction
                    )
                    max_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["b_single"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_road)
                    maxs.append(max_road)
                    # mean_road = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["road"]["b_single"]["mean"]
                    #     * damage_fraction
                    # )

            else:
                if lanes >= 6:
                    min_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["ab_dual_lanes_ge6"]["min"]
                        * damage_fraction
                    )
                    max_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["ab_dual_lanes_ge6"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_road)
                    maxs.append(max_road)
                    # mean_road = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["road"]["ab_dual_lanes_ge6"]["mean"]
                    #     * damage_fraction
                    # )
                else:  # lanes < 6
                    min_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["ab_dual_lanes_lt6"]["min"]
                        * damage_fraction
                    )
                    max_road = (
                        length
                        * 1e-3
                        * lanes
                        * damage_values["road"]["ab_dual_lanes_lt6"]["max"]
                        * damage_fraction
                    )
                    mins.append(min_road)
                    maxs.append(max_road)
                    # mean_road = (
                    #     length
                    #     * 1e-3
                    #     * lanes
                    #     * damage_values["road"]["ab_dual_lanes_lt6"]["mean"]
                    #     * damage_fraction
                    # )
    min_cost = min(mins)
    max_cost = max(maxs)
    mean_cost = np.mean([min_cost, max_cost])

    return min_cost, max_cost, mean_cost


def calculate_damage(
    disrupted_links,
    damage_curves,
    damage_values,
    flood_type,
):
    # add default columns
    curves = ["C1", "C2", "C3", "C4", "C5", "C6"]
    for col in curves:
        disrupted_links[f"{col}_damage_fraction"] = np.nan
        disrupted_links[f"{col}_damage_value_min"] = np.nan
        disrupted_links[f"{col}_damage_value_max"] = np.nan
        disrupted_links[f"{col}_damage_value_mean"] = np.nan

    for idx, row in disrupted_links.iterrows():
        # compute damage fractions (2 damage curves for each scenario)
        curve1, damage_fraction1, curve2, damage_fraction2 = compute_damage_fraction(
            row.road_classification,
            row.trunk_road,
            row.hasTunnel,
            row.flood_depth,
            damage_curves,
        )
        disrupted_links.loc[idx, f"{curve1}_damage_fraction"] = damage_fraction1
        disrupted_links.loc[idx, f"{curve2}_damage_fraction"] = damage_fraction2

        # compute damage costs (3 damage values for each scenario)
        min_cost1, max_cost1, mean_cost1 = compute_damage_values(
            row.geometry.length,
            flood_type,
            damage_fraction1,
            row.road_classification,
            row.form_of_way,
            row.lanes,
            row.hasTunnel,
            row.aveBridgeWidth,
            row.damage_level,
            damage_values,
        )
        min_cost2, max_cost2, mean_cost2 = compute_damage_values(
            row.geometry.length,
            flood_type,
            damage_fraction2,
            row.road_classification,
            row.form_of_way,
            row.lanes,
            row.hasTunnel,
            row.aveBridgeWidth,
            row.damage_level,
            damage_values,
        )

        disrupted_links.loc[idx, f"{curve1}_damage_value_min"] = min_cost1
        disrupted_links.loc[idx, f"{curve1}_damage_value_max"] = max_cost1
        disrupted_links.loc[idx, f"{curve1}_damage_value_mean"] = mean_cost1
        disrupted_links.loc[idx, f"{curve2}_damage_value_min"] = min_cost2
        disrupted_links.loc[idx, f"{curve2}_damage_value_max"] = max_cost2
        disrupted_links.loc[idx, f"{curve2}_damage_value_mean"] = mean_cost2

    return disrupted_links


def main(flood_type=None):
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

    # features
    road_links = gpd.read_parquet(
        base_path / "disruption_analysis_1129" / "road_links.pq"
    )
    # intersections
    disrupted_links = road_links[road_links["flood_depth"] > 0].reset_index(drop=True)
    # damage calculation
    disrupted_links_with_damage = calculate_damage(
        disrupted_links, damage_curves, damage_values, flood_type
    )

    # export results
    disrupted_links_with_damage.to_csv(
        base_path
        / "disruption_analysis_1129"
        / "disrupted_links_with_damage_values.csv",
        index=False,
    )


if __name__ == "__main__":
    main(flood_type="surface")
