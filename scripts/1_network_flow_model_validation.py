# %%
from pathlib import Path
import sys
import time

import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road_revised as func

import logging
import json
import warnings

warnings.simplefilter("ignore")
base_path = Path("/data/DAFNI_NIRD/data/processed_data")


def main(num_of_cpu, sample_stride=1):
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
        sample_stride (int): Stride length to use on OD. Defaults to using
            entire matrix.

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
        base_path / "networks" / "GB_road_links_with_bridges.gpq"
    )
    # od matrix (2021) -> updated to od with bridges
    od_node_2021 = pd.read_csv(
        base_path / "census_datasets" / "od_gb_oa_2021_node_with_bridges.csv"
    )
    od_node_2021["Car21"] = od_node_2021["Car21"] * 2
    # od_node_2021 = od_node_2021.head(100)  # debug

    if sample_stride > 1:
        logging.info(f"For testing, sampling every {sample_stride} flows")
        od_node_2021 = od_node_2021.iloc[::sample_stride]

    logging.info(f"\n{od_node_2021}")
    logging.info(f"Total flows: {od_node_2021.Car21.sum()}")

    # initialise road links
    logging.info("Generate road links")
    road_links = func.edge_init(
        road_link_file,
        flow_breakpoint_dict,
        flow_capacity_dict,
        free_flow_speed_dict,
        urban_speed_dict,
        min_speed_dict,
        max_flow_speed_dict=None,
    )
    # create igraph network
    logging.info("Create igraph network")
    network, road_links = func.create_igraph_network(road_links)
    # run flow simulation
    logging.info("Run simulation")
    (
        road_links,
        isolation,
        odpfc,
    ) = func.network_flow_model(
        road_links,
        network,
        od_node_2021,
        flow_breakpoint_dict,
        num_of_cpu,
    )

    isolation_df = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "flow",
        ],
    )
    odpfc_df = pd.DataFrame(
        odpfc,
        columns=[
            "origin",
            "destination",
            "path",
            "flow",
            "fuel_cost_per_flow",
            "time_cost_per_flow",
            "toll_cost_per_flow",
        ],
    )
    logging.info(f"The total simulation time: {time.time() - start_time}")

    # export files
    road_links.to_parquet(
        base_path.parent / "outputs" / "base_scenario" / "edge_flows_validation.gpq"
    )
    isolation_df.to_parquet(
        base_path.parent / "outputs" / "base_scenario" / "trip_isolation_validation.pq"
    )
    odpfc_df.to_parquet(
        base_path.parent / "outputs" / "base_scenario" / "trip_isolation_validation.pq"
    )


if __name__ == "__main__":
    """
    Entry point of the script. Reads the number of CPUs from command-line arguments
    and calls the main function.

    Command-line Arguments:
        num_of_cpu (int): Number of CPUs to use for parallel processing.
        sample_stride (int): Stride length to use on OD.

    Returns:
        None: Prints a message if the required argument is missing.
    """
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        num_of_cpu, sample_stride = sys.argv[1:]
        main(int(num_of_cpu), int(sample_stride))
    except IndexError or NameError:
        logging.info("Please enter the required number of CPUs!")
