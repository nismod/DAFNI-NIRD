import json
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

import nird.road_revised as func
from nird.utils import load_config

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["soge_clusters"])
tqdm.pandas()


def have_common_items(list1: List, list2: List) -> bool:
    return bool(set(list1) & set(list2))


def bridge_recovery(
    day: int,
    damage_level: str,
    designed_capacity: float,
    current_capacity: float,
    current_speed: float,
    initial_speed: float,
    max_speed: float,
    bridge_recovery_dict: Dict,
    acc_speed: float,
    acc_capacity: float,
) -> Tuple[float, float]:
    """Compute the daily recovery of brideg capacity and speed based on
    damage level and recovery rates."""

    if day == 0:  # occurrence of damage
        acc_speed = min(current_speed, max_speed)
        if damage_level == ["no", "minor", "moderate"]:
            acc_capacity = current_capacity
        else:  # extensive/severe
            acc_capacity = 0.0
    else:
        acc_speed = initial_speed  # free-flow speed
        if damage_level == "no":
            acc_capacity = current_capacity
        else:
            recover_rate = bridge_recovery_dict[damage_level][day]
            if recover_rate > 0:
                acc_capacity = max(designed_capacity * recover_rate, current_capacity)

    return acc_capacity, acc_speed


def ordinary_road_recovery(
    day: int,
    damage_level: str,
    designed_capacity: float,
    current_capacity: float,
    current_speed: float,
    initial_speed: float,
    max_speed: float,
    road_recovery_dict: Dict,
    acc_speed: float,
    acc_capacity: float,
) -> Tuple[float, float]:
    """Compute the daily recovery of ordinary road capacity and speed based on
    damage level and recovery rates."""

    if day == 0:
        acc_speed = min(current_speed, max_speed)
        if damage_level in ["no", "minor", "moderate"]:
            acc_capacity = current_capacity
        else:
            acc_capacity = 0.0
    elif 0 < day <= 2:
        acc_speed = initial_speed
        if damage_level == "no":
            acc_capacity = current_capacity
        else:
            recovery_rate = road_recovery_dict[damage_level][day]
            if recovery_rate > 0:
                acc_capacity = max(designed_capacity * recovery_rate, current_capacity)
    else:  # day > 2
        acc_speed = initial_speed
        if damage_level == "no":
            acc_capacity = current_capacity
        else:
            acc_capacity = max(designed_capacity, current_capacity)

    return acc_capacity, acc_speed


def main(
    depth_thres,
    number_of_cpu,
):
    """Main function:

    Model Inputs
    ------------
    - Model Parameters
    - odpfc_32p.pq: base scenario output
    - road_links_x.gpq: disruption analysis output
        [disruption analysis] road_label, flood_depth_max, damage_level_max,
        [base scenario analysis] current_capacity, current_speed,
        [config] free_flow_speed, min_flow_speeds, max_speed, initial_flow_speeds

    Outputs
    -------
    - Daily edge flows during the recovery period (D-0 to D-110).
    - Isolated trips after daily recovery (D-0 to D-110).

    """
    # bridge recovery rates
    with open(base_path / "parameters" / "capt_minor.json", "r") as f:
        bridge_minor = json.load(f)
    with open(base_path / "parameters" / "capt_moderate.json", "r") as f:
        bridge_moderate = json.load(f)
    with open(base_path / "parameters" / "capt_extensive.json", "r") as f:
        bridge_extensive = json.load(f)
    with open(base_path / "parameters" / "capt_severe.json", "r") as f:
        bridge_severe = json.load(f)

    # on D-0
    bridge_minor.insert(0, 0.0)
    bridge_moderate.insert(0, 0.0)
    bridge_extensive.insert(0, 0.0)
    bridge_severe.insert(0, 0.0)

    bridge_recovery_dict = {
        "minor": bridge_minor,
        "moderate": bridge_moderate,
        "extensive": bridge_extensive,
        "severe": bridge_severe,
    }

    # road (other than bridge) recovery rates
    road_minor = [0.0, 1.0, 1.0]
    road_moderate = [0.0, 1.0, 1.0]
    road_extensive = [0.0, 1.0, 1.0]
    road_severe = [0.0, 0.5, 1.0]

    road_recovery_dict = {
        "minor": road_minor,
        "moderate": road_moderate,
        "extensive": road_extensive,
        "severe": road_severe,
    }

    # network parameters
    with open(base_path / "parameters" / "flow_cap_plph_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)

    # road links (with 8 added columns)
    road_links = gpd.read_parquet(
        base_path.parent
        / "results"
        / "disruption_analysis"
        / str(depth_thres)
        / "links"
        / "road_links_17.gpq"  # Thames Lloyd's RDS
    )  # input1
    road_links = func.edge_reclassification_func(road_links)
    road_links["designed_capacity"] = (
        road_links.combined_label.map(flow_capacity_dict) * road_links.lanes * 24
    )

    """How to define disrupted links?
    - with extensive/severe damage levels
    - max_speed < current_speed
    """
    disrupted_links = (
        road_links.loc[
            (road_links.max_speed < road_links.current_speed)
            | (road_links.damage_level_max == "extensive")
            | (road_links.damage_level_max == "severe"),
            "e_id",
        ]
        .unique()
        .tolist()
    )

    # extract the disrupted od matrix
    partial_have_common_items = partial(have_common_items, list2=disrupted_links)
    od_path_file = pd.read_parquet(
        base_path.parent / "results" / "base_scenario" / "odpfc_32p.pq",
        engine="fastparquet",
    )  # input2
    od_path_file["disrupted"] = np.vectorize(partial_have_common_items)(
        od_path_file.path
    )
    disrupted_od = od_path_file[od_path_file["disrupted"]].reset_index(drop=True)
    disrupted_od.rename(columns={"flow": "Car21"}, inplace=True)  # (15090, )
    disrupted_od.drop(columns="disrupted", inplace=True)
    # disrupted_od = disrupted_od.head(10)  # debug: 76

    """
    recovery of road capacities (-> for mapping):
                D-0  D-15  D-30  D-60  D-90  D-110
    minor        0    1     1     1     1     1
    moderate     0    0     0.4   1     1     1
    extensive    0    0     0     0.66  1     1
    severe       0    0     0     0.17  0.67  1
    """

    # key variables for rerouting analysis:
    road_links["disrupted"] = 0
    road_links.loc[road_links.e_id.isin(disrupted_links), "disrupted"] = 1
    road_links["acc_flow"] = 0
    road_links["acc_capacity"] = road_links["current_capacity"]
    road_links["acc_speed"] = road_links["current_speed"]

    for day in range(111):  # Day 0-110
        print(f"Rerouting Analysis on D-{day}...")
        (
            road_links["acc_capacity"],
            road_links["acc_speed"],
        ) = zip(
            *road_links.progress_apply(
                lambda row: (
                    bridge_recovery(
                        day,
                        row["damage_level_max"],
                        row["designed_capacity"],
                        row["current_capacity"],
                        row["current_speed"],
                        row["initial_flow_speeds"],
                        row["max_speed"],
                        bridge_recovery_dict,
                        row["acc_speed"],
                        row["acc_capacity"],
                    )
                    if row["disrupted"] == 1 and row["road_label"] == "bridge"
                    else (
                        ordinary_road_recovery(
                            day,
                            row["damage_level_max"],
                            row["designed_capacity"],
                            row["current_capacity"],
                            row["current_speed"],
                            row["initial_flow_speeds"],
                            row["max_speed"],
                            road_recovery_dict,
                            row["acc_speed"],
                            row["acc_capacity"],
                        )
                        if row["disrupted"] == 1 and row["road_label"] != "bridge"
                        else (row["acc_capacity"], row["acc_speed"])
                    )
                ),
                axis=1,
            )
        )

        # drop the disrupted links
        valid_road_links = road_links[
            (road_links.acc_capacity > 0) & (road_links.acc_speed > 0)
        ].reset_index(drop=True)

        # create network
        network = func.create_igraph_network(valid_road_links)

        # run rerouting analysis
        valid_road_links, isolation, _ = func.network_flow_model(
            valid_road_links,
            network,
            disrupted_od,
            flow_breakpoint_dict,
            num_of_cpu=number_of_cpu,  # system input
        )

        # update key variables
        road_links = road_links.set_index("e_id")
        road_links.update(
            valid_road_links.set_index("e_id")[
                ["acc_flow", "acc_capacity", "acc_speed"]
            ]
        )
        road_links = road_links.reset_index()

        # isolation_total.extend(isolation)
        isolation_df = pd.DataFrame(
            isolation,
            columns=[
                "origin_node",
                "destination_node",
                "Car21",
            ],
        )

        # export outputs
        if isolation_df.empty:
            road_links.to_parquet(
                base_path.parent
                / "results"
                / "rerouting_analysis"
                / str(depth_thres)
                / "17"
                / "fixed"
                / f"edge_flows_{day}.gpq"
            )
            isolation_df.to_csv(
                base_path.parent
                / "results"
                / "rerouting_analysis"
                / str(depth_thres)
                / "17"
                / "fixed"
                / f"trip_isolations_{day}.csv",
                index=False,
            )
            print("There is no disrupted flows!")
            break
        else:
            isolation_df.to_csv(
                base_path.parent
                / "results"
                / "rerouting_analysis"
                / str(depth_thres)
                / "17"
                / "fixed"
                / f"trip_isolations_{day}.csv",
                index=False,
            )
            if day in [0, 1, 2, 3, 4, 5, 15, 30, 60, 90, 110]:
                road_links.to_parquet(
                    base_path.parent
                    / "results"
                    / "rerouting_analysis"
                    / str(depth_thres)
                    / "17"
                    / "fixed"
                    / f"edge_flows_{day}.gpq"
                )


if __name__ == "__main__":
    try:
        depth_thres = int(sys.argv[1])
    except (IndexError, ValueError):
        print(
            "Error: Please provide the flood depth for road closure (e.g., 30 or "
            "60 cm) as the first argument!"
        )
        sys.exit(1)

    try:
        number_of_cpu = int(sys.argv[2])
    except (IndexError, ValueError):
        print("Error: Please provide the number of CPUs as the second argument!")
        sys.exit(1)

    main(depth_thres, number_of_cpu)
