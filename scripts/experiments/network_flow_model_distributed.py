# %%
import os
import sys
import math
import itertools

from pathlib import Path
import time
import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road as func


import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])
disruption_path = Path(load_config()["paths"]["disruption_path"])
# %%


def network_flow_model(max_flow_speed_dict, event_key):
    start_time = time.time()

    # model parameters
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    with open(base_path / "parameters" / "flow_cap_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "free_flow_speed_dict.json", "r") as f:
        free_flow_speed_dict = json.load(f)
    with open(base_path / "parameters" / "min_speed_cap.json", "r") as f:
        min_speed_dict = json.load(f)
    with open(base_path / "parameters" / "urban_speed_cap.json", "r") as f:
        urban_speed_dict = json.load(f)

    # road networks (urban_filter: mannual correction)
    road_node_file = gpd.read_parquet(
        base_path / "networks" / "road" / "road_node_file.geoparquet"
    )
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "road" / "road_link_file.geoparquet"
    )

    # O-D matrix (2021)
    od_node_2021 = pd.read_csv(
        base_path / "census_datasets" / "od_matrix" / "od_gb_oa_2021_node.csv"
    )
    # od_node_2021 = od_node_2021[od_node_2021.Car21 > 1].reset_index(drop=True)
    od_node_2021["Car21"] = od_node_2021["Car21"] * 2
    # od_node_2021 = od_node_2021.head(10)  #!!! for test
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # generate OD pairs
    list_of_origin_nodes, dict_of_destination_nodes, dict_of_origin_supplies = (
        func.extract_od_pairs(od_node_2021)
    )

    # flow simulation
    (
        road_link_file,  # edge flow results
        od_node_2021,  # edge flow validation & component costs
        isolated_od_dict,  # isolated trips
    ) = func.network_flow_model(
        road_link_file,  # road
        road_node_file,  # road
        list_of_origin_nodes,  # od
        dict_of_origin_supplies,  # od
        dict_of_destination_nodes,  # od
        free_flow_speed_dict,  # net
        flow_breakpoint_dict,  # net
        flow_capacity_dict,  # net
        min_speed_dict,  # net
        urban_speed_dict,  # net
        od_node_2021,  # od_flow_matrix
        max_flow_speed_dict,  # disruption analysis
    )

    # export files
    road_link_file.to_parquet(
        base_path.parent / "outputs" / f"gb_edge_flows_{event_key}.geoparquet"
    )
    od_node_2021.to_csv(
        base_path.parent / "outputs" / f"od_costs_{event_key}.csv", index=False
    )
    isolated_od_df = pd.Series(isolated_od_dict).reset_index()
    if isolated_od_df.shape[0] != 0:  # in case of empty df
        isolated_od_df.columns = ["origin_node", "destination_node", "isolated_flows"]
        isolated_od_df.to_csv(
            base_path.parent / "outputs" / f"isolated_od_flows_{event_key}.csv",
            index=False,
        )
    print(f"The network flow model is completed for {event_key}!")
    print(f"The total simulation time: {time.time() - start_time}. ")


def main(task_id: int, task_count: int):
    flood_event_paths = []
    for root, _, files in os.walk(disruption_path):
        for file in files:
            file_path = Path(root) / file
            flood_event_paths.append(file_path)

    n_events = len(flood_event_paths)
    event_per_task = math.ceil(n_events / task_count)
    event_batches = list(itertools.batched(flood_event_paths, event_per_task))
    try:
        event_paths_to_run = event_batches[task_id]
    except IndexError:
        print(f"No events for {task_id=} {task_count=} {n_events=}")
        sys.exit()

    for event_path in event_paths_to_run:
        event_key = event_path.stem
        flooded_road_max_speed = pd.read_csv(event_path)
        # maximum vehicle speeds (mph) -> flooded roads (id/e_id)
        max_flow_speed_dict = flooded_road_max_speed.set_index("id")[
            "max_speed_mph_adjusted"
        ].to_dict()
        network_flow_model(max_flow_speed_dict, event_key)


if __name__ == "__main__":
    try:
        task_id = int(sys.argv[1])
        task_count = int(sys.argv[2])
    except IndexError:
        print(f"Usage: python {__file__} <task_id> <task_count>")
    main(task_id, task_count)
