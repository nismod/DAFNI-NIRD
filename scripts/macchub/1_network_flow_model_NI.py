# %%
from pathlib import Path
import sys
import time

import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road_capacity as func  # contains latest functions

import logging
import json
import warnings

warnings.simplefilter("ignore")
macc_path = Path(load_config()["paths"]["MACCHUB"])  # ./processed_data


def main(
    num_of_chunk: int,
    num_of_cpu: int,
    sample_stride=1,
):
    start_time = time.time()
    db_path = macc_path / "dbs" / "macc_ni.duckdb"
    logging.info(f"Database path is: {db_path}")

    # outpath
    out_path = macc_path.parent / "outputs"
    out_path.mkdir(parents=True, exist_ok=True)
    iso_out_path = out_path / "isolation_NI.pq"
    odpfc_out_path = out_path / "odpfc_NI.pq"

    # model parameters
    with open(macc_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    with open(macc_path / "parameters" / "flow_cap_plph_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(macc_path / "parameters" / "free_flow_speed_dict.json", "r") as f:
        free_flow_speed_dict = json.load(f)
    with open(macc_path / "parameters" / "min_speed_cap.json", "r") as f:
        min_speed_dict = json.load(f)
    with open(macc_path / "parameters" / "urban_speed_cap.json", "r") as f:
        urban_speed_dict = json.load(f)

    # network links (England only)
    road_link_file = gpd.read_parquet(macc_path / "networks" / "edges_final.gpq")

    # od matrix (2021, 2030, 2050)
    od_node = pd.read_parquet(macc_path / "od" / "od_node_NI.pq")
    od_node["Car21"] = od_node["Car21"] * 2
    logging.info(f"\n{od_node}")
    logging.info(f"total flows: {od_node.Car21.sum()}")

    if sample_stride > 1:
        logging.info(f"For testing, sampling every {sample_stride} flows")

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
        _,  # cList
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
        capacity_mode=True,
    )

    # export files
    road_links.to_parquet(out_path / "edge_flow_NI.gpq")
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
            sample_stride,
        )
    except (IndexError, NameError):
        logging.info("Please check terminal inputs.")
