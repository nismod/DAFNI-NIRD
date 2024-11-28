# %%
from pathlib import Path
import time
import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road_revised as func

import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])


# %%
def main():
    start_time = time.time()
    # model parameters
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    with open(base_path / "parameters" / "flow_cap_plph_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)  # edge capacities per lane per hour
    with open(base_path / "parameters" / "free_flow_speed_dict.json", "r") as f:
        free_flow_speed_dict = json.load(f)
    with open(base_path / "parameters" / "min_speed_cap.json", "r") as f:
        min_speed_dict = json.load(f)
    with open(base_path / "parameters" / "urban_speed_cap.json", "r") as f:
        urban_speed_dict = json.load(f)

    # network links
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "road" / "GB_road_link_file.geoparquet"
    )  # road links with "lanes" info

    # od matrix (2021)
    od_node_2021 = pd.read_csv(
        base_path / "census_datasets" / "od_matrix" / "od_gb_oa_2021_node.csv"
    )
    # od_node_2021 = od_node_2021[od_node_2021.Car21 > 1].reset_index(drop=True)
    od_node_2021["Car21"] = od_node_2021["Car21"] * 2
    od_node_2021 = od_node_2021.head(100000)
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # create igraph network
    road_links = func.edge_init(
        road_link_file,
        flow_capacity_dict,
        free_flow_speed_dict,
        urban_speed_dict,
        min_speed_dict,
        max_flow_speed_dict=None,
    )
    network, road_links = func.create_igraph_network(road_links)

    # run flow simulation
    road_links, isolation, odpfc = func.network_flow_model(
        road_links, network, od_node_2021, flow_breakpoint_dict
    )
    odpfc = pd.DataFrame(
        odpfc,
        columns=[
            "origin_node",
            "destination_node",
            "path",
            "flow",
            "operating_cost",
            "time_cost",
            "toll_cost",
        ],
    )

    isolation = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "path",
            "flow",
            "operating_cost",
            "time_cost",
            "toll_cost",
        ],
    )
    print(f"The total simulation time: {time.time() - start_time}")

    # export files
    road_links.to_parquet(base_path.parent / "outputs" / "edge_flows_1128.pq")
    odpfc.to_parquet(base_path / "odpfc_1128.pq")
    isolation.to_parquet(base_path.parent / "outputs" / "trip_isolation_1128.pq")


if __name__ == "__main__":
    main()
