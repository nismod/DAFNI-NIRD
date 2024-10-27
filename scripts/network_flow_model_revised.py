from pathlib import Path
import time
import pandas as pd
import geopandas as gpd  # type: ignore

from utils import load_config
import road1026 as func

import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])


def main():
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
    # od_node_2021 = od_node_2021.head(10)
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # generate OD pairs
    list_of_origin_nodes, dict_of_destination_nodes, dict_of_origin_supplies = (
        func.extract_od_pairs(od_node_2021)
    )

    # flow simulation
    (
        road_link_file,  # edge flow results
        # od_node_2021,  # edge flow validation & component costs
        isolated_od_dict,  # isolated trips
        edge_odf_dict,
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
        # od_node_2021,  # od_flow_matrix
        max_flow_speed_dict=None,  # disruption analysis
    )

    # compute od matrix: od_flow, od_voc, od_vot, od_toll, od_cost
    data = []
    for _, od_pairs in edge_odf_dict.items():
        for (origin, destination), values in od_pairs.items():
            data.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "flow": values[0],
                    "od_voc": values[1],
                    "od_vot": values[2],
                    "od_toll": values[3],
                }
            )
    edge_odf_df = pd.DataFrame(data)
    edge_odf_df = edge_odf_df.groupby(["origin", "destination"], as_index=False).sum()

    # export files
    road_link_file.to_parquet(
        base_path.parent / "outputs" / "gb_edge_flows_20241026.geoparquet"
    )
    """
    od_node_2021.to_csv(
        base_path.parent / "outputs" / "od_costs_20241026.csv", index=False
    )
    """
    edge_odf_df.to_csv(
        base_path.parent / "outputs" / "od_costs_20241026.csv", index=False
    )
    isolated_od_df = pd.Series(isolated_od_dict).reset_index()
    if isolated_od_df.shape[0] != 0:  # in case of empty df
        isolated_od_df.columns = ["origin_node", "destination_node", "isolated_flows"]
        isolated_od_df.to_csv(
            base_path.parent / "outputs" / "isolated_od_flows_20241026.csv", index=False
        )
    print(f"The total simulation time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
