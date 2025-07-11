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

import nird.road_recovery as func
from nird.utils import get_flow_on_edges

warnings.simplefilter("ignore")
base_path = Path("/data/inputs")
tqdm.pandas()


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
        if damage_level != "no":
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
        rates.insert(0, 0.0)
        bridge_recovery_dict[level] = rates

    road_recovery_dict = {
        "minor": [0.0, 1.0, 1.0],
        "moderate": [0.0, 1.0, 1.0],
        "extensive": [0.0, 1.0, 1.0],
        "severe": [0.0, 0.5, 1.0],
    }
    return bridge_recovery_dict, road_recovery_dict


def main(number_of_cpu):
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
    # Load recovery dicts
    bridge_recovery_dict, road_recovery_dict = load_recovery_dicts(base_path)

    # Load network parameters
    with open(base_path / "parameters" / "flow_cap_plph_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    partial_speed_flow_func = partial(
        func.speed_flow_func, flow_breakpoint_dict=flow_breakpoint_dict
    )

    # Load road links
    # change to batch process
    road_links = gpd.read_parquet(
        base_path / "links" / "road_links_17.gpq"
    )  # Thames Lloyd's RDS
    road_links["designed_capacity"] = (
        road_links.combined_label.map(flow_capacity_dict) * road_links.lanes * 24
    )

    # Load OD path file
    od_path_file = pd.read_parquet(
        base_path / "od" / "odpfc_32p.pq",  # flag!
        engine="fastparquet",
    )  # input2

    # Randomly sample 1000 rows for testing
    od_path_file = od_path_file.sample(n=1000, random_state=42, ignore_index=True)

    # Identify disrupted links
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
    partial_have_common_items = partial(have_common_items, list2=disrupted_links)
    cDict = {}
    # Recovery analysis loop
    for day in range(111):
        print(f"Rerouting Analysis on D-{day}...")
        if day not in [0, 1, 36]:
            continue
        # update edge capacities
        road_links["current_capacity"] = road_links.progress_apply(
            lambda row: (
                bridge_recovery(
                    day,
                    row["damage_level_max"],
                    row["designed_capacity"],
                    row["current_capacity"],
                    bridge_recovery_dict,
                )
                if row["road_label"] == "bridge"
                else (
                    ordinary_road_recovery(
                        day,
                        row["damage_level_max"],
                        row["designed_capacity"],
                        row["current_capacity"],
                        road_recovery_dict,
                    )
                )
            ),
            axis=1,
        )

        # Update the disrupted OD paths
        od_path_file["disrupted_links"] = np.vectorize(partial_have_common_items)(
            od_path_file.path
        )
        od_path_file["disrupted_links"] = od_path_file["disrupted_links"].apply(list)
        current_capacity_dict = road_links.set_index("e_id")[
            "current_capacity"
        ].to_dict()
        od_path_file["capacities_of_disrupted_links"] = od_path_file[
            "disrupted_links"
        ].apply(lambda x: [current_capacity_dict.get(xi, 0) for xi in x])

        od_path_file["min_capacities_of_disrupted_links"] = od_path_file[
            "capacities_of_disrupted_links"
        ].apply(lambda x: min(x) if len(x) > 0 else np.nan)

        disrupted_od = od_path_file.loc[
            od_path_file["min_capacities_of_disrupted_links"].notnull()
        ]
        disrupted_od["flow"] = np.maximum(
            0,
            disrupted_od["flow"] - disrupted_od["min_capacities_of_disrupted_links"],
        )

        # Adjust road flows
        road_links["disrupted_flow"] = 0.0
        disrupted_edge_flow = get_flow_on_edges(disrupted_od, "e_id", "path", "flow")
        disrupted_edge_flow.rename(columns={"flow": "disrupted_flow"}, inplace=True)
        road_links = road_links.set_index("e_id")
        road_links.update(disrupted_edge_flow.set_index("e_id")["disrupted_flow"])
        road_links = road_links.reset_index()

        road_links["acc_capacity"] = road_links["current_capacity"] + np.where(
            road_links["damage_level_max"].isin(["extensive", "severe"]),
            0,
            road_links["disrupted_flow"],
        )

        # Flow speed recalculation
        road_links["acc_speed"] = np.vectorize(partial_speed_flow_func)(
            road_links["combined_label"],
            road_links["current_flow"],
            road_links["initial_flow_speeds"],
            road_links["min_flow_speeds"],
        )
        if day == 0:
            road_links["acc_speed"] = road_links[["current_speed", "max_speed"]].min(
                axis=1
            )

        # initial key variables
        road_links["acc_flow"] = 0
        disrupted_od.rename(columns={"flow": "Car21"}, inplace=True)

        # create network
        valid_road_links = road_links[
            (road_links["acc_capacity"] > 0) & (road_links["acc_speed"] > 0)
        ].reset_index(drop=True)
        network = func.create_igraph_network(valid_road_links)

        # run flow rerouting analysis
        _, isolation, _, cList = func.network_flow_model(
            valid_road_links,
            network,
            disrupted_od,
            flow_breakpoint_dict,
            num_of_cpu=number_of_cpu,
        )
        cDict[day] = cList
        road_links = road_links.set_index("e_id")
        road_links.update(valid_road_links.set_index("e_id")["acc_flow"])
        road_links = road_links.reset_index()

        isolation_df = pd.DataFrame(
            isolation,
            columns=[
                "origin_node",
                "destination_node",
                "Car21",
            ],
        )
        # export results
        # out_path = base_path.parent / "outputs"
        isolation_df.to_csv(
            base_path.parent / "outputs" / f"trip_isolations_17_{day}.csv",
            index=False,
        )
        # sys.exit(1)
        del isolation_df
        del valid_road_links

    cost_df = pd.DataFrame.from_dict(
        cDict, orient="index", columns=["time", "fuel", "toll", "total"]
    ).reset_index()
    cost_df.rename(columns={"index": "day"}, inplace=True)
    cost_df.to_csv(base_path.parent / "outputs" / "cost_matrix.csv", index=False)


if __name__ == "__main__":
    try:
        number_of_cpu = int(sys.argv[1])
    except (IndexError, ValueError):
        print("Error: Please provide the number of CPUs as the first argument!")
        sys.exit(1)

    main(number_of_cpu)
