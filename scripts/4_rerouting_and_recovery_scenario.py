# %%
import sys
import json
import warnings

from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import nird.road_recovery as func

# import nird.road_revised as func
from nird.utils import load_config, get_flow_on_edges

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["soge_clusters"])
tqdm.pandas()


# %%
def have_common_items(list1: List, list2: List) -> Set:
    common_items = set(list1) & set(list2)
    return common_items


def bridge_recovery(
    day: int,
    damage_level: str,
    pre_event_capacity: float,
    acc_capacity: float,
    bridge_recovery_dict: Dict,
) -> Tuple[float, float]:
    if damage_level != "no":
        recovery_rate = bridge_recovery_dict.get(damage_level, [])[day]
        acc_capacity = pre_event_capacity * recovery_rate
    return acc_capacity


def ordinary_road_recovery(
    day: int,
    damage_level: str,
    pre_event_capacity: float,
    acc_capacity: float,
    road_recovery_dict: Dict,
) -> Tuple[float, float]:
    if damage_level != "no":
        recovery_rate = road_recovery_dict.get(damage_level, [])[day]
        acc_capacity = pre_event_capacity * recovery_rate
    return acc_capacity


def load_scenarios(base_path: Path) -> Tuple[Dict, Dict]:
    """Load recovery rates for bridges and ordinary roads."""
    df = pd.read_csv(
        base_path / "tables" / "recovery design_updated.csv",
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
        conditions.append(int(row["event_day"]))  # condition = [1, 0, 0, ..]

    return (bridge_recovery_dict, road_recovery_dict, scenarios, conditions)


def main(
    depth_thres,
    flood_key,
    scenario_idx,
    num_of_chunk,
    num_of_cpu,
):
    # Load recovery scenarios
    (
        bridge_recovery_dict,
        road_recovery_dict,
        scenarios,  # list of scenarios
        conditions,  # indicating event day
    ) = load_scenarios(base_path)

    # Load OD path file
    od_path_file = pd.read_parquet(
        base_path.parent / "results" / "base_scenario" / "revision" / "odpfc.pq",
        engine="pyarrow",
    )

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

    # Identify disrupted links
    disrupted_links = (
        road_links.loc[
            (road_links.max_speed < road_links.current_speed)  # all damage levels
            | (road_links.damage_level_max.isin(["extensive", "severe"])),
            "e_id",
        ]
        .unique()
        .tolist()
    )

    logging.info("Finding disrupted links among OD pairs...")
    disrupted_links_set = set(disrupted_links)
    od_path_file["disrupted_links"] = od_path_file["path"].progress_apply(
        lambda path: have_common_items(path, disrupted_links_set)
    )
    od_path_file["disrupted_links"] = od_path_file["disrupted_links"].apply(list)

    # Recovery analysis loop
    cDict = {}
    logging.info(f"Rerouting Analysis on Scenario-{scenario_idx}...")
    event_day = conditions[scenario_idx]
    # update edge capacities
    logging.info("Updating edge capacities...")
    road_links["acc_capacity"] = road_links["current_capacity"]
    road_links["acc_capacity"] = road_links.apply(
        lambda row: (
            bridge_recovery(
                scenario_idx,
                row["damage_level_max"],
                row["current_capacity"],
                row["acc_capacity"],
                bridge_recovery_dict,
            )
            if row["road_label"] == "bridge"
            else (
                ordinary_road_recovery(
                    scenario_idx,
                    row["damage_level_max"],
                    row["current_capacity"],
                    row["acc_capacity"],
                    road_recovery_dict,
                )
            )
        ),
        axis=1,
    )
    # Update the disrupted OD paths
    logging.info("Preparing the disrupted OD matrix...")
    disrupted_od = od_path_file[
        od_path_file["disrupted_links"].apply(lambda x: len(x)) > 0
    ].reset_index(drop=True)

    # conduct exploded chunks
    max_chunk_size = 100_000
    if max_chunk_size > len(disrupted_od):
        chunk_size = len(disrupted_od)
    else:
        n_chunk = min(100, max(1, len(disrupted_od) // max_chunk_size))
        chunk_size = len(disrupted_od) // n_chunk
    print(f"disrupted_od size: {len(disrupted_od)}")
    print(f"chunk_size: {chunk_size}")

    out_chunks = []
    for start in tqdm(
        range(0, len(disrupted_od), chunk_size),
        desc="Processing chunks",
        unit="chunk",
    ):
        chunk = disrupted_od.iloc[start : start + chunk_size].copy()
        chunk = chunk.explode("disrupted_links")
        chunk = chunk.merge(
            road_links[["e_id", "current_flow", "acc_capacity"]],
            how="left",
            left_on="disrupted_links",
            right_on="e_id",
        )
        chunk["fraction"] = (
            (chunk["acc_capacity"] / chunk["current_flow"].replace(0, np.nan))
            .clip(upper=1.0)
            .fillna(1.0)
        )
        chunk.path = chunk.path.apply(tuple)
        chunk = (
            chunk.groupby(
                [
                    "origin_node",
                    "destination_node",
                    "path",
                    "flow",
                    "time_cost_per_flow",
                    "operating_cost_per_flow",
                    "toll_cost_per_flow",
                ],
                as_index=False,
            )["fraction"]
            .min()
            .reset_index(drop=True)
        )
        out_chunks.append(chunk)

    disrupted_od = pd.concat(out_chunks, ignore_index=True)
    disrupted_od.path = disrupted_od.path.apply(tuple)
    disrupted_od = (
        disrupted_od.groupby(
            [
                "origin_node",
                "destination_node",
                "path",
                "flow",
                "time_cost_per_flow",
                "operating_cost_per_flow",
                "toll_cost_per_flow",
            ],
            as_index=False,
        )["fraction"]
        .min()
        .reset_index(drop=True)
    )
    disrupted_od.path = disrupted_od.path.apply(list)
    disrupted_od["disrupted_flow"] = disrupted_od["flow"] * (
        1 - disrupted_od["fraction"]
    )
    logging.info(f"The total disrupted flows: {disrupted_od.disrupted_flow.sum()}")
    # disrupted_od.to_parquet(
    #     base_path.parent
    #     / "results"
    #     / "rerouting_analysis"
    #     / "test"
    #     / "disrupted_od.pq"
    # )
    # Restore capacity for non-disrupted roads
    disrupted_edge_flow = get_flow_on_edges(
        disrupted_od, "e_id", "path", "disrupted_flow"
    )  # redistribute those od back to road links
    # disrupted_edge_flow.to_parquet(
    #     base_path.parent
    #     / "results"
    #     / "rerouting_analysis"
    #     / "test"
    #     / "disrupted_edge_flow.pq"
    # )
    road_links["disrupted_flow"] = 0.0
    road_links = road_links.set_index("e_id")
    road_links.update(disrupted_edge_flow.set_index("e_id")["disrupted_flow"])
    road_links = road_links.reset_index()

    # estimate the pre-event cost matrix for disrupted flows
    pre_time = (disrupted_od.disrupted_flow * disrupted_od.time_cost_per_flow).sum()
    pre_operate = (
        disrupted_od.disrupted_flow * disrupted_od.operating_cost_per_flow
    ).sum()
    pre_toll = (disrupted_od.disrupted_flow * disrupted_od.toll_cost_per_flow).sum()
    total_pre_cost = pre_time + pre_operate + pre_toll

    # initial key variables
    logging.info("Initialising key variables for flow modelling...")
    disrupted_od.rename(
        columns={"disrupted_flow": "Car21"}, inplace=True
    )  # !!! make sure to pass disrupted flow for rerouting analysis
    road_links["acc_capacity"] = (
        road_links["acc_capacity"] + road_links["disrupted_flow"]
    )
    road_links["acc_flow"] = (
        road_links["current_flow"] - road_links["disrupted_flow"]
    ).clip(lower=0)
    # road_links.to_parquet(
    #     base_path.parent
    #     / "results"
    #     / "rerouting_analysis"
    #     / "test"
    #     / "road_links.gpq"
    # )
    # print("Complete debug!")
    # sys.exit()
    # logging.info("Updating road speed limits...")
    road_links["acc_speed"] = road_links.apply(
        lambda x: func.update_edge_speed(
            x["combined_label"],  # constant
            x["acc_flow"],  # variable
            x["initial_flow_speeds"],  # constant
            x["min_flow_speeds"],  # constant
            x["breakpoint_flows"],  # constant
        ),
        axis=1,
    )
    if event_day == 1:  # apply speed constraint to every road
        road_links["acc_speed"] = road_links[["acc_speed", "max_speed"]].min(axis=1)
    if (
        event_day == 2
    ):  # only apply speed constraint to roads with flooddepth (2-6) meters
        mask = (road_links["flood_depth_max"] >= 2) and (
            road_links["flood_depth_max"] < 6
        )
        road_links.loc[mask, "acc_speed"] = road_links.loc[
            mask, ["acc_speed", "max_speed"]
        ].min(axis=1)
    if event_day == 3:  # only for roads > 6 meters
        mask = road_links["flood_depth_max"] >= 6
        road_links.loc[mask, "acc_speed"] = road_links.loc[
            mask, ["acc_speed", "max_speed"]
        ].min(axis=1)

    # create network (time-consuming when updating network edge index)
    logging.info("Creating igraph network...")
    valid_road_links = road_links[
        (road_links["acc_capacity"] > 0) & (road_links["acc_speed"] > 0)
    ].reset_index(drop=True)
    network, valid_road_links = func.create_igraph_network(valid_road_links)

    # run flow rerouting analysis
    logging.info("Running flow simulation...")
    (
        valid_road_links,
        isolation,
        _,
        (post_time, post_operate, post_toll, total_post_cost),
    ) = func.network_flow_model(
        valid_road_links,
        network,
        disrupted_od[["origin_node", "destination_node", "Car21"]],
        flow_breakpoint_dict,
        num_of_chunk,
        num_of_cpu,
    )

    # estimate rerouting sub costs
    rer_time = post_time - pre_time
    rer_operate = post_operate - pre_operate
    rer_toll = post_toll - pre_toll
    rerouting_cost = rer_time + rer_operate + rer_toll

    logging.info(
        f"The original travel costs for disrupted od: £ million {total_pre_cost/ 1e6}"
    )
    logging.info(
        f"The total travel costs after disruption: £ million {total_post_cost/ 1e6}"
    )
    logging.info(
        f"The rerouting cost for scenario {scenario_idx}: £ million {rerouting_cost / 1e6}"
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

    # rerouting costs
    cDict[scenario_idx] = [rer_time, rer_operate, rer_toll, rerouting_cost]
    cost_df = pd.DataFrame.from_dict(
        cDict,
        orient="index",
        columns=["rer_time", "rer_operate", "rer_toll", "rerouting_cost"],
    ).reset_index()
    cost_df.rename(columns={"index": "scenario"}, inplace=True)
    cost_df.to_csv(out_path / f"rerouting_cost_{scenario_idx}.csv", index=False)

    # trip isolations
    isolation_df = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "Car21",
        ],
    )
    isolation_df = isolation_df[
        (isolation_df.origin_node != isolation_df.destination_node)
        & (isolation_df.Car21 > 0)
    ].reset_index()

    isolation_df.to_csv(
        out_path / f"trip_isolations_{scenario_idx}.csv",
        index=False,
    )

    # edge flows
    road_links = road_links.set_index("e_id")
    road_links.update(valid_road_links.set_index("e_id")["acc_flow"])
    road_links = road_links.reset_index()
    road_links["change_flow"] = road_links["acc_flow"] - road_links["current_flow"]
    road_links.to_parquet(out_path / f"edge_flows_{scenario_idx}.gpq")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        depth_key, event_key, day_key, num_of_chunk, num_of_cpu = sys.argv[1:]
        main(
            int(depth_key),
            int(event_key),
            int(day_key),
            int(num_of_chunk),
            int(num_of_cpu),
        )

    except (IndexError, ValueError):
        logging.info(
            "Please provide inputs: depth_key, event_key, day_key, num_of_chunk, and num_of_cpu!"
        )
        sys.exit(1)
