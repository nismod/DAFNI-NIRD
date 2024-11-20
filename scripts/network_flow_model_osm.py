from pathlib import Path
import time
import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road_osm as func

import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])


def main():
    start_time = time.time()
    # model parameters
    with open(base_path / "parameters" / "osm_flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    with open(base_path / "parameters" / "osm_flow_cap_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "osm_free_flow_speed_dict.json", "r") as f:
        free_flow_speed_dict = json.load(f)
    with open(base_path / "parameters" / "osm_min_speed_cap.json", "r") as f:
        min_speed_dict = json.load(f)
    with open(base_path / "parameters" / "osm_urban_speed_cap.json", "r") as f:
        urban_speed_dict = json.load(f)

    # road networks (urban_filter: mannual correction)
    road_node_file = gpd.read_parquet(
        base_path / "networks" / "road" / "osm_road_node_file.geoparquet"
    )
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "road" / "osm_road_link_file.geoparquet"
    )

    # O-D matrix (2021)
    od_node_2021 = pd.read_csv(
        base_path / "census_datasets" / "od_matrix" / "osm_od_gb_oa_2021_node.csv"
    )
    od_node_2021["Car21"] = od_node_2021["Car21"] * 2
    # od_node_2021 = od_node_2021.head(10)  #!!! for debug
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # generate OD pairs
    list_of_origin_nodes, dict_of_destination_nodes, dict_of_origin_supplies = (
        func.extract_od_pairs(od_node_2021)
    )

    # flow simulation
    (
        road_link_file,  # edge flow results
        isolated_od_dict,  # isolated trips
        odpfc,
    ) = func.network_flow_model(
        road_link_file,  # road
        road_node_file,  # road
        list_of_origin_nodes,  # od
        dict_of_origin_supplies,  # od
        dict_of_destination_nodes,  # od
        free_flow_speed_dict,  # net
        flow_breakpoint_dict,  # net -> [m, a-trunk and bridge, a-primary and secondary]
        flow_capacity_dict,  # net
        min_speed_dict,  # net
        urban_speed_dict,  # net
        max_flow_speed_dict=None,  # disruption analysis
    )

    # export files
    road_link_file.to_parquet(base_path.parent / "outputs" / "osm_edge_flows.pq")
    odpfc.to_parquet(base_path.parent / "outputs" / "osm_odpfc.pq")
    isolated_od_df = pd.Series(isolated_od_dict).reset_index()
    if isolated_od_df.shape[0] != 0:  # in case of empty df
        isolated_od_df.columns = ["origin_node", "destination_node", "isolated_flows"]
        isolated_od_df.to_parquet(
            base_path.parent / "outputs" / "osm_isolations.pq",
            index=False,
        )
    print(f"The total simulation time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
