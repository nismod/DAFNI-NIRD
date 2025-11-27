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
macc_path = Path(load_config()["paths"]["MACCHUB"])


def main(
    year,
    ssp,
    num_of_chunk,
    num_of_cpu,
    sample_stride=1,
):
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

    # network links
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "GB_road_links_with_bridges.gpq"
    )

    # od matrix (2021, 2050-SSP1, SSP2, SSP3, SSP4, SSP5)
    if year == 2021:
        od_path = macc_path / "inputs" / "od" / "od_node_gb_2021.pq"
    else:
        od_path = macc_path / "inputs" / "od" / f"od_node_gb_{year}_SSP{ssp}.pq"
    od_node = pd.read_parquet(od_path, engine="fastparquet")
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
        odpfc,
        _,
    ) = func.network_flow_model(
        road_links,
        network,
        od_node,
        flow_breakpoint_dict,
        num_of_chunk,
        num_of_cpu,
    )

    # isolation
    isolation_df = pd.DataFrame(
        isolation,
        columns=[
            "origin_node",
            "destination_node",
            "flow",
        ],
    )
    isolation_df = isolation_df[
        (isolation_df.origin_node != isolation_df.destination_node)
        & (isolation_df.flow > 0)
    ].reset_index(drop=True)

    # odpfc
    odpfc_df = pd.DataFrame(
        odpfc,
        columns=[
            "origin_node",
            "destination_node",
            "path_idx",
            "path",
            "flow",
            "operating_cost_per_flow",
            "time_cost_per_flow",
            "toll_cost_per_flow",
        ],
    )
    odpfc_df.drop(columns=["path_idx"], inplace=True)
    odpfc_df.path = odpfc_df.path.apply(tuple)
    odpfc_df = odpfc_df.groupby(
        by=["origin_node", "destination_node", "path"], as_index=False
    ).agg(
        {
            "flow": "sum",
            "operating_cost_per_flow": "first",
            "time_cost_per_flow": "first",
            "toll_cost_per_flow": "first",
        }
    )

    # export files
    out_path = macc_path / "outputs"
    out_path.mkdir(parents=True, exist_ok=True)
    road_links.to_parquet(out_path / f"edge_flow_{year}_SSP{ssp}.gpq")
    isolation.to_parquet(out_path / f"trip_isolation_{year}_SSP{ssp}.pq")
    odpfc_df.to_parquet(out_path / f"odpfc_{year}_SSP{ssp}.pq")
    logging.info(f"The total simulation time: {time.time() - start_time}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        year, ssp, num_of_chunk, num_of_cpu = sys.argv[1:]
        sample_stride = 1
        main(
            int(year),
            int(ssp),
            int(num_of_chunk),
            int(num_of_cpu),
            sample_stride=sample_stride,
        )
    except (IndexError, NameError):
        logging.info("Please enter year, ssp, num_of_chunk, and num_of_cpu!")
