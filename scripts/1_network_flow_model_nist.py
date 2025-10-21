# %%
from pathlib import Path
import sys
import time

import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road_capacity as func

import logging
import json
import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["soge_clusters"])
nist_path = Path(load_config()["paths"]["NIST"])


def main(
    year: int,
    num_of_chunk: int,
    num_of_cpu: int,
    sample_stride=1,
):
    start_time = time.time()
    db_path = base_path / "dbs" / f"nist_{year}.duckdb"
    logging.info(f"Database path is: {db_path}")

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

    # network links (England only)
    road_link_file = gpd.read_parquet(
        nist_path / "inputs" / "networks" / "England_road_links_with_bridges.gpq"
    )

    # od matrix (2021, 2030, 2050)
    od_node = pd.read_csv(nist_path / "inputs" / "od" / f"od_node_{year}_england.csv")
    od_node.rename(columns={f"outflow_{year}": "Car21"}, inplace=True)
    od_node["Car21"] = od_node["Car21"] * 2

    if sample_stride > 1:
        logging.info(f"For testing, sampling every {sample_stride} flows")

    logging.info(f"\n{od_node}")
    logging.info(f"total flows: {od_node.Car21.sum()}")

    # initialise road links
    logging.info("Initialising road links...")
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
    logging.info("Creating igraph network...")
    network, road_links = func.create_igraph_network(road_links)

    # run flow simulation
    logging.info("Running flow simulation...")
    (
        road_links,
        isolation,
        _,  # odpfc,
        _,  # cList
    ) = func.network_flow_model(
        road_links,
        network,
        od_node,
        flow_breakpoint_dict,
        num_of_chunk,
        num_of_cpu,
        db_path,
    )

    isolation_df = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "flow",
        ],
    )
    # export files
    isolation_df.to_csv(nist_path / "outputs" / f"trip_isolation_{year}.csv")
    road_links.to_parquet(nist_path / "outputs" / f"edge_flow_{year}.gpq")
    logging.info(f"The total simulation time: {time.time() - start_time}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        year, num_of_chunk, num_of_cpu = sys.argv[1:]
        sample_stride = 1
        main(int(year), int(num_of_chunk), int(num_of_cpu), sample_stride)
    except (IndexError, NameError):
        logging.info(
            "No input year, num_of_chunk, and num_of_cpu, using default values"
        )
