# %%
from pathlib import Path

import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road as func

import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])
# %%
if __name__ == "__main__":
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
    # od_node_2021 = od_node_2021.head(10)
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # generate OD pairs
    list_of_origin_nodes, dict_of_destination_nodes, dict_of_origin_supplies = (
        func.extract_od_pairs(od_node_2021)
    )

    # flow simulation
    (
        speed_dict,
        acc_flow_dict,
        acc_capacity_dict,
        od_cost_dict,
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
    )

    # append estimation of: speeds, flows, and remaining capacities
    road_link_file.ave_flow_rate = road_link_file.e_id.map(speed_dict)
    road_link_file.acc_flow = road_link_file.e_id.map(acc_flow_dict)
    road_link_file.acc_capacity = road_link_file.e_id.map(acc_capacity_dict)
    # change field types
    road_link_file.acc_flow = road_link_file.acc_flow.astype(int)
    road_link_file.acc_capacity = road_link_file.acc_capacity.astype(int)
    # calculate od travel costs
    od_node_2021["od_cost"] = od_node_2021.apply(
        lambda row: od_cost_dict[(row["origin_node"], row["destination_node"])], axis=1
    )

    # export files
    road_link_file.to_parquet(base_path.parent / "outputs" / "gb_edge_flows.geoparquet")
    od_node_2021.to_csv(base_path.parent / "outputs" / "od_costs.csv", index=False)
