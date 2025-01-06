# %%
from pathlib import Path
import sys
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
def main(num_of_cpu):
    start_time = time.time()
    # model parameters
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    with open(base_path / "parameters" / "flow_cap_plph_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "free_flow_speed_dict.json", "r") as f:
        free_flow_speed_dict = json.load(f)
    with open(base_path / "parameters" / "min_speed_cap.json", "r") as f:
        min_speed_dict = json.load(f)
    with open(base_path / "parameters" / "urban_speed_cap.json", "r") as f:
        urban_speed_dict = json.load(f)

    # network links -> updated to links with bridges
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "road" / "GB_road_links_with_bridges.gpq"
    )

    # od matrix (2021) -> updated to od with bridges
    od_node_2021 = pd.read_csv(
        base_path
        / "census_datasets"
        / "od_matrix"
        / "od_gb_oa_2021_node_with_bridges_32p.csv"
    )
    od_node_2021["Car21"] = od_node_2021["Car21"] * 2
    # od_node_2021 = od_node_2021.head(10)  # for debug
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # initialise road links
    """ adding columns:
    - edge properties:
        free-flow speeds
        min speeds (urban/rural)
        max speeds (on flooded roads -> for disruption analysis only)
        initial speeds
    - edge variables:
        acc_speed
        acc_flow
        acc_capacity
    """
    road_links = func.edge_init(
        road_link_file,
        flow_capacity_dict,
        free_flow_speed_dict,
        urban_speed_dict,
        min_speed_dict,
        max_flow_speed_dict=None,
    )
    # create igraph network
    network = func.create_igraph_network(road_links)

    # run flow simulation
    road_links, isolation, odpfc = func.network_flow_model(
        road_links,
        network,
        od_node_2021,
        flow_breakpoint_dict,
        num_of_cpu,
    )

    odpfc = pd.DataFrame(
        odpfc,
        columns=[
            "origin_node",
            "destination_node",
            "path",
            "flow",
            "operating_cost_per_flow",
            "time_cost_per_flow",
            "toll_cost_per_flow",
        ],
    )

    odpfc.path = odpfc.path.apply(tuple)
    odpfc = odpfc.groupby(
        by=["origin_node", "destination_node", "path"], as_index=False
    ).agg(
        {
            "flow": "sum",
            "operating_cost_per_flow": "first",
            "time_cost_per_flow": "first",
            "toll_cost_per_flow": "first",
        }
    )

    isolation = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "path",
            "flow",
            "operating_cost_per_flow",
            "time_cost_per_flow",
            "toll_cost_per_flow",
        ],
    )
    print(f"The total simulation time: {time.time() - start_time}")
    # breakpoint()
    # export files
    road_links.to_parquet(base_path.parent / "outputs" / "edge_flows_33p.pq")
    isolation.to_parquet(base_path.parent / "outputs" / "trip_isolation_33p.pq")
    odpfc.to_parquet(base_path.parent / "outputs" / "odpfc_33p.pq")


if __name__ == "__main__":
    try:
        num_of_cpu = int(sys.argv[1])
        main(num_of_cpu)
    except IndexError or NameError:
        print("Please enter the required number of CPUs!")
