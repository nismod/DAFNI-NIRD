import sys
import time
import json
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd

from nird.utils import load_config
import nird.road_revised as func

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["soge_clusters"])


def main(num_of_cpu):
    """
    Main function to simulate traffic flow on a road network.

    Model Inputs:
        - Model parameters:
            - Flow breakpoints, capacity, free-flow speeds, minimum speeds, and urban
                speed limits.
        - GB_road_links_with_bridges.gpq:
            GeoDataFrame containing road network data with attributes:
            - Free-flow speeds: Baseline speeds under normal conditions.
            - Minimum speeds: Defined separately for urban and rural areas.
            - Maximum speeds: Speeds on flooded roads (used for disruption analysis
                only).
            - Initial speeds: Starting speeds for the simulation.
        - od_gb_oa_2021_node_with_bridges.csv or
            od_gb_oa_2021_node_with_bridges_32p.csv:
            Origin-destination matrix containing traffic flow data.

    Model Outputs:
        - edge_flows_32p.gpq:
            GeoDataFrame of road network data enriched with simulation results:
            - acc_flow: Accumulated traffic flow on road links.
            - acc_capacity: Remaining capacity of road links.
            - acc_speed: Current speeds on road links.
        - trip_isolation_32p.pq:
            DataFrame of isolated trips resulting from network disruptions.
        - odpfc_32p.pq:
            Origin-destination-path-flow-cost matrix for evaluating network performance:
            - Path: Sequence of road links for each trip.
            - Flow: Traffic flow along the path.
            - Operating, time, and toll costs per flow.

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

    # network links -> updated to links with bridges
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "GB_road_links_with_bridges.gpq"
    )

    # od matrix (2021) -> updated to od with bridges
    od_node_2021 = pd.read_csv(
        base_path / "census_datasets" / "od_gb_oa_2021_node_with_bridges_32p.csv"
    )
    od_node_2021["Car21"] = od_node_2021["Car21"] * 2
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
            "flow",
        ],
    )
    print(f"The total simulation time: {time.time() - start_time}")

    # export files
    road_links.to_parquet(
        base_path.parent / "results" / "base_scenario" / "edge_flows_32p.gpq"
    )
    isolation.to_parquet(
        base_path.parent / "results" / "base_scenario" / "trip_isolation_32p.pq"
    )
    odpfc.to_parquet(base_path.parent / "results" / "base_scenario" / "odpfc_32p.pq")


if __name__ == "__main__":
    try:
        num_of_cpu = int(sys.argv[2])
    except (IndexError, ValueError):
        print("Error: Please provide the number of CPUs!")
        sys.exit(1)
    main(num_of_cpu)
