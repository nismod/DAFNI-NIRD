# %%
from pathlib import Path
import sys
import time

import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road_validation as func

import json
import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])


def main(num_of_cpu):
    """
    Main function to validate the network flow model.

    Model Inputs:
        - Model parameters:
            - Flow breakpoints, capacity, free-flow speeds, minimum speeds,
            and urban speed limits.
        - GB_road_links_with_bridges.gpq:
            GeoDataFrame containing road network data with attributes.
        - od_gb_oa_2021_node_with_bridges.csv:
            Origin-destination matrix containing traffic flow data.

    Model Outputs:
        - edge_flows_validation.gpq:
            GeoDataFrame of road network data enriched with validation results.
        - trip_isolation_validation.csv:
            CSV file containing data on isolated trips resulting from network
            disruptions.

    Parameters:
        num_of_cpu (int): Number of CPUs to use for parallel processing.

    Returns:
        None: Outputs are saved to files.
    """
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

    # network links -> network links with bridges
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "road" / "GB_road_links_with_bridges.gpq"
    )
    # od matrix (2021) -> updated to od with bridges
    od_node_2021 = pd.read_csv(
        base_path
        / "census_datasets"
        / "od_matrix"
        / "od_gb_oa_2021_node_with_bridges.csv"
    )
    od_node_2021["Car21"] = od_node_2021["Car21"] * 2
    # od_node_2021 = od_node_2021.head(100)  # debug
    print(f"total flows: {od_node_2021.Car21.sum()}")

    # initialise road links
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
    (
        road_links,
        isolation,
        _,
    ) = func.network_flow_model(
        road_links,
        network,
        od_node_2021,
        flow_breakpoint_dict,
        num_of_cpu,
    )
    isolation = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "flow",
        ],
    )
    print(f"The total simulation time: {time.time() - start_time}")

    # export files
    road_links.to_parquet(
        base_path.parent / "outputs" / "base_scenario" / "edge_flows_validation.gpq"
    )
    isolation.to_csv(
        base_path.parent / "outputs" / "base_scenario" / "trip_isolation_validation.csv"
    )


if __name__ == "__main__":
    """
    Entry point of the script. Reads the number of CPUs from command-line arguments
    and calls the main function.

    Command-line Arguments:
        num_of_cpu (int): Number of CPUs to use for parallel processing.

    Returns:
        None: Prints a message if the required argument is missing.
    """
    try:
        num_of_cpu = int(sys.argv[1])
        main(num_of_cpu)
    except IndexError or NameError:
        print("Please enter the required number of CPUs!")
