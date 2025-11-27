# %%
from pathlib import Path
import sys
import time

import pandas as pd
import geopandas as gpd  # type: ignore
import logging

from nird.utils import load_config
import nird.road_revised as func

import json
import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["soge_clusters"])
miraca_path = Path(load_config()["paths"]["MIRACA"])
network_path = miraca_path / "processed_data" / "processed_unimodal"
od_path = miraca_path / "processed_data" / "lifelines_OD"
out_path = miraca_path / "flows"
iso_out_path = out_path / "isolation.pq"
odpfc_out_path = out_path / "odpfc.pq"


def main(
    num_of_chunk,
    num_of_cpu,
    sample_stride=1,
):
    start_time = time.time()

    # database
    db_path = base_path / "dbs" / "miraca.duckdb"
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

    # network links
    road_link_file = gpd.read_parquet(
        network_path / "europe_road_edges_TENT.parquet", engine="fastparquet"
    )
    road_link_file.rename(columns={"id": "e_id"}, inplace=True)

    # od matrix
    od_node = pd.read_parquet(od_path / "road_OD_final.parquet", engine="fastparquet")
    od_node.rename(columns={"value": "Car21"}, inplace=True)

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
        _,
    ) = func.network_flow_model(
        road_links,
        network,
        od_node,
        flow_breakpoint_dict,
        num_of_chunk,
        num_of_cpu,
        db_path=db_path,
        iso_out_path=iso_out_path,
        odpfc_out_path=odpfc_out_path,
    )

    # export files
    out_path.mkdir(parents=True, exist_ok=True)
    road_links.to_parquet(out_path / "road_flows.gpq")
    logging.info(f"The total simulation time: {time.time() - start_time}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        num_of_chunk, num_of_cpu = sys.argv[1:]
        sample_stride = 1
        main(
            int(num_of_chunk),
            int(num_of_cpu),
            sample_stride=sample_stride,
        )
    except (IndexError, NameError):
        logging.info("Please enter num_of_chunk and num_of_cpu!")
