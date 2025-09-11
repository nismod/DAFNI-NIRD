# %%
import sys
import json
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import nird.road_revised as func
from nird.utils import load_config, get_flow_on_edges

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["soge_clusters"])
tqdm.pandas()


# %%
def have_common_items(list1: List, list2: List) -> bool:
    common_items = set(list1) & set(list2)
    return common_items


def bridge_recovery(
    day: int,
    damage_level: str,
    pre_event_capacity: float,
    acc_capacity: float,
    bridge_recovery_dict: Dict,
    event_day: bool,
) -> Tuple[float, float]:
    if event_day:
        if damage_level in ["extensive", "severe"]:
            acc_capacity = 0
    else:
        if damage_level != "no":
            recovery_rate = bridge_recovery_dict.get(damage_level, [])[day]
            acc_capacity = max(pre_event_capacity * recovery_rate, acc_capacity)
    return acc_capacity


def ordinary_road_recovery(
    day: int,
    damage_level: str,
    pre_event_capacity: float,
    acc_capacity: float,
    road_recovery_dict: Dict,
    event_day: bool,
) -> Tuple[float, float]:
    if event_day:  # the occurance of damage
        if damage_level in ["extensive", "severe"]:
            acc_capacity = 0
    else:
        if damage_level != "no":
            recovery_rate = road_recovery_dict.get(damage_level, [])[day]
            acc_capacity = max(pre_event_capacity * recovery_rate, acc_capacity)
    return acc_capacity


def load_scenarios(base_path: Path) -> Tuple[Dict, Dict]:
    """Load recovery rates for bridges and ordinary roads."""
    df = pd.read_excel(
        base_path / "tables" / "recovery design_updated.xlsx",
        sheet_name="unique_groups",
    )

    bridge_recovery_dict = defaultdict(list)
    road_recovery_dict = defaultdict(list)
    scenarios = []
    conditions = []
    for _, row in df.iterrows():
        bridge_recovery_dict["minor"].append(row["bridge_minor"])
        bridge_recovery_dict["moderate"].append(row["bridge_moderate"])
        bridge_recovery_dict["extensive"].append(row["bridge_extensive"])
        bridge_recovery_dict["severe"].append(row["bridge_severe"])
        road_recovery_dict["minor"].append(row["road_minor"])
        road_recovery_dict["moderate"].append(row["road_moderate"])
        road_recovery_dict["extensive"].append(row["road_extensive"])
        road_recovery_dict["severe"].append(row["road_severe"])
        scenarios.append(int(row["scenario"]))
        conditions.append(int(row["event_day"]))

    return (bridge_recovery_dict, road_recovery_dict, scenarios, conditions)


def main(
    depth_thres,
    number_of_cpu,
    flood_key,
    scenario_idx,
):
    # Load recovery scenarios
    (
        bridge_recovery_dict,
        road_recovery_dict,
        scenarios,  # list of scenarios
        conditions,  # condition == 1: day 0, otherwise: day > 0
    ) = load_scenarios(base_path)

    # Load network parameters
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)

    # Load road links
    road_links = gpd.read_parquet(
        base_path.parent
        / "results"
        / "disruption_analysis"
        / "revision"
        / str(depth_thres)
        / "links"
        / f"road_links_{flood_key}.gpq"
    )
    road_links["breakpoint_flows"] = road_links["combined_label"].map(
        flow_breakpoint_dict
    )

    # Load OD path file
    od_path_file = pd.read_parquet(
        base_path.parent / "results" / "base_scenario" / "odpfc_validation_0221.pq",
        engine="fastparquet",
    )

    # Identify disrupted links
    disrupted_links = (
        road_links.loc[
            (road_links.max_speed < road_links.current_speed)  # all damage levels
            | (road_links.damage_level_max == "extensive")
            | (road_links.damage_level_max == "severe"),
            "e_id",
        ]
        .unique()
        .tolist()
    )
    partial_have_common_items = partial(have_common_items, list2=disrupted_links)
    od_path_file["disrupted_links"] = np.vectorize(partial_have_common_items)(
        od_path_file.path
    )
    od_path_file["disrupted_links"] = od_path_file["disrupted_links"].apply(list)

    # Recovery analysis loop
    cDict = {}
    logging.info(f"Rerouting Analysis on Scenario-{scenario_idx}...")
    event_day = conditions[scenario_idx] == 1
    # update edge capacities
    logging.info("Updating edge capacities...")
    road_links["acc_capacity"] = road_links["current_capacity"]
    road_links["acc_capacity"] = road_links.progress_apply(
        lambda row: (
            bridge_recovery(
                scenario_idx,
                row["damage_level_max"],
                row["current_capacity"],
                row["acc_capacity"],
                bridge_recovery_dict,
                event_day,
            )
            if row["road_label"] == "bridge"
            else (
                ordinary_road_recovery(
                    scenario_idx,
                    row["damage_level_max"],
                    row["current_capacity"],
                    row["acc_capacity"],
                    road_recovery_dict,
                    event_day,
                )
            )
        ),
        axis=1,
    )

    # Update the disrupted OD paths
    current_capacity_dict = road_links.set_index("e_id")["current_capacity"].to_dict()
    od_path_file["capacities_of_disrupted_links"] = od_path_file[
        "disrupted_links"
    ].apply(lambda x: [current_capacity_dict.get(xi, 0) for xi in x])

    od_path_file["min_capacities_of_disrupted_links"] = od_path_file[
        "capacities_of_disrupted_links"
    ].apply(lambda x: min(x) if len(x) > 0 else np.nan)

    disrupted_od = od_path_file.loc[
        od_path_file["min_capacities_of_disrupted_links"].notnull()
    ]
    # some flows could be still send to the original path
    # with the rest flows disrupted
    disrupted_od["disrupted_flow"] = np.maximum(
        0,
        disrupted_od["flow"] - disrupted_od["min_capacities_of_disrupted_links"],
    )  # disrupted_od_flows

    # Restore capacity for non-disrupted roads
    disrupted_edge_flow = get_flow_on_edges(
        disrupted_od, "e_id", "path", "disrupted_flow"
    )  # redistribute those od back to road links

    # disrupted_edge_flow.rename(columns={"flow": "disrupted_flow"}, inplace=True)
    road_links["disrupted_flow"] = 0.0
    road_links = road_links.set_index("e_id")
    road_links.update(disrupted_edge_flow.set_index("e_id")["disrupted_flow"])
    road_links = road_links.reset_index()

    # initial key variables
    logging.info("Initialising key variables for flow modelling...")
    road_links["acc_capacity"] = (
        road_links["acc_capacity"] + road_links["disrupted_flow"]
    )
    road_links["acc_flow"] = road_links["current_flow"] - road_links["disrupted_flow"]
    logging.info("Updating road speed limits...")
    road_links["acc_speed"] = road_links.progress_apply(
        lambda x: func.update_edge_speed(
            x["combined_label"],  # constant
            x["acc_flow"],  # variable
            x["initial_flow_speeds"],  # constant
            x["min_flow_speeds"],  # constant
            x["breakpoint_flows"],  # constant
        ),
        axis=1,
    )
    if event_day:  # the occurance of damage
        road_links["acc_speed"] = road_links[["acc_speed", "max_speed"]].min(axis=1)

    logging.info("Preparing disrupted OD matrix...")
    disrupted_od.rename(columns={"flow": "Car21"}, inplace=True)

    total_cost = (
        disrupted_od.disrupted_flow
        * (
            disrupted_od.operating_cost_per_flow
            + disrupted_od.time_cost_per_flow
            + disrupted_od.toll_cost_per_flow
        )
    ).sum()
    logging.info(f"The original travel costs for disrupted od: £{total_cost}")

    # create network (time-consuming when updating network edge index)
    logging.info("Creating igraph network...")
    valid_road_links = road_links[
        (road_links["acc_capacity"] > 0) & (road_links["acc_speed"] > 0)
    ].reset_index(drop=True)
    network, valid_road_links = func.create_igraph_network(valid_road_links)

    # run flow rerouting analysis
    logging.info("Running flow simulation...")
    valid_road_links, isolation, _, (_, _, _, total_cost2) = func.network_flow_model(
        valid_road_links,
        network,
        disrupted_od[["origin_node", "destination_node", "Car21"]],
        flow_breakpoint_dict,
        num_of_cpu=number_of_cpu,
    )
    rerouting_cost = total_cost2 - total_cost
    logging.info(f"The total travel costs after disruption: £{total_cost2}")
    logging.info(f"The rerouting cost for scenario {scenario_idx}: £{rerouting_cost}")

    # rerouting costs
    cDict[scenario_idx] = rerouting_cost
    cost_df = pd.DataFrame.from_dict(
        cDict, orient="index", columns=["rerouting_cost"]
    ).reset_index()
    cost_df.rename(columns={"index": "scenario"}, inplace=True)
    # edge flows
    road_links = road_links.set_index("e_id")
    road_links.update(valid_road_links.set_index("e_id")["acc_flow"])
    road_links = road_links.reset_index()
    road_links["change_flow"] = road_links["acc_flow"] - road_links["current_flow"]
    # trip isolations
    isolation_df = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "Car21",
        ],
    )

    logging.info("Saving outputs to disk...")
    out_path = (
        base_path.parent
        / "results"
        / "rerouting_analysis"
        / "revision"
        / str(depth_thres)  # e.g.,30
        / str(flood_key)  # e.g.,17
    )
    out_path.mkdir(parents=True, exist_ok=True)
    cost_df.to_csv(out_path / f"cost_matrix_{scenario_idx}.csv", index=False)
    isolation_df.to_csv(
        out_path / f"trip_isolations_{scenario_idx}.csv",
        index=False,
    )
    road_links.to_parquet(out_path / f"edge_flows_{scenario_idx}.gpq")


if __name__ == "__main__":
    try:
        depth_thres = int(sys.argv[1])
    except (IndexError, ValueError):
        logging.info(
            "Error: Please provide the flood depth for road closure "
            "(e.g., 15, 30 or 60 cm) as the first argument!"
        )
        sys.exit(1)

    try:
        number_of_cpu = int(sys.argv[2])
    except (IndexError, ValueError):
        logging.info("Error: Please provide the number of CPUs as the second argument!")
        sys.exit(1)

    try:
        flood_key = int(sys.argv[3])
    except (IndexError, ValueError):
        logging.info("Error: Please provide the flood key!")
        sys.exit(1)

    try:
        scenario_idx = int(sys.argv[4])  # 0-12
    except (IndexError, ValueError):
        logging.info(
            "Error: Please provide the scenario index (0-12) as the fourth argument!"
        )
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    main(depth_thres, number_of_cpu, flood_key, scenario_idx)
