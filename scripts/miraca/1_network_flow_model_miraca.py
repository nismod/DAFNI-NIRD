# %%
from pathlib import Path
import time

import pandas as pd
import geopandas as gpd  # type: ignore
import logging

from nird.utils import load_config
import nird.road_revised as func

import json
import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["arc_miraca_path"])
# base_path = Path(load_config()["paths"]["miraca_path"])
out_path = base_path.parent / "results"
iso_out_path = out_path / "isolation.pq"
odpfc_out_path = out_path / "odpfc.pq"


# %%
def edge_init(
    road_links,
    flow_capacity_dict,  # -> capacity
    flow_breakpoint_dict,  # -> speed
    free_flow_speed_dict,
    urban_speed_dict,
    min_speed_dict,
    max_speed_dict=None,
):
    # edge reclassification
    # combined_label: M (motorways), A_dual(trunk), A_single(primary),
    # B_dual(secondary), B_single(tertiary), minor(others)
    assert "combined_label" in road_links.columns, "combined_label column not exists!"
    # breakpoint_flows
    if "breakpoint_flows" not in road_links.columns:
        road_links["breakpoint_flows"] = road_links["combined_label"].map(
            flow_breakpoint_dict
        )
    # free_flow_speeds
    if "free_flow_speeds" not in road_links.columns:
        road_links["free_flow_speeds"] = road_links["combined_label"].map(
            free_flow_speed_dict
        )
    # min_flow_speeds
    if "min_flow_speeds" not in road_links.columns:
        road_links["min_flow_speeds"] = road_links["combined_label"].map(min_speed_dict)
    # initial_flow_speeds
    if "initial_flow_speeds" not in road_links.columns:
        road_links["initial_flow_speeds"] = road_links["free_flow_speeds"]

    # max_flow_speeds (optional, only for disruption analysis)
    if max_speed_dict is not None:
        if "max_speeds" not in road_links.columns:
            road_links["max_speeds"] = road_links["e_id"].map(max_speed_dict)
            # if max < min: close the roads
            road_links.loc[
                road_links.max_speeds < road_links.min_flow_speeds,
                "initial_flow_speeds",
            ] = 0.0
            # if min < max < free
            road_links.loc[
                (road_links.max_speeds >= road_links.min_flow_speeds)
                & (road_links.max_speeds < road_links.free_flow_speeds),
                "initial_flow_speeds",
            ] = road_links.max_speeds
            # if max > free: free flow speeds (by default)
            # remove the closed/damaged roads
            road_links = road_links[road_links["initial_flow_speeds"] > 0]
            road_links.reset_index(drop=True, inplace=True)

    # acc_capacity, acc_flow, acc_speed
    road_links["acc_flow"] = 0.0
    road_links["acc_capacity"] = (
        road_links["combined_label"].map(flow_capacity_dict) * road_links["lanes"] * 24
    )
    road_links["acc_speed"] = road_links["initial_flow_speeds"]

    return road_links


def main(
    num_of_chunk: int,
    num_of_cpu: int,
    od_file: str = "od_psv.pq",
    vehicle_type: str = "psv",
    sample_stride: int = 1,
):
    start_time = time.time()

    # database
    db_path = base_path / "dbs" / "miraca.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)  # create dbs
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
    road_link_file = gpd.read_parquet(base_path / "networks" / "eu_road_links.gpq")
    logging.info(f"road size: {road_link_file.shape[0]}")
    logging.info(f"road links: \n{road_link_file}")

    # od matrix
    od_node = pd.read_parquet(base_path / "census_datasets" / f"{od_file}")
    od_node = od_node[od_node.Car21 > 0].reset_index(drop=True)
    # od_node = od_node.head(100)  # for testing purpose
    logging.info(f"bus od: \n{od_node}")
    logging.info(f"total bus flows: {od_node.Car21.sum()}")

    if sample_stride > 1:
        logging.info(f"For testing, sampling every {sample_stride} flows")
    # initialise road links
    logging.info("Initialising road links...")
    road_links = edge_init(
        road_link_file,
        flow_breakpoint_dict,
        flow_capacity_dict,
        free_flow_speed_dict,
        urban_speed_dict,
        min_speed_dict,
        max_speed_dict=None,
    )

    # create igraph network
    logging.info("Creating igraph network...")
    network, road_links = func.create_igraph_network(
        road_links,
        vehicle_type=vehicle_type,
    )

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
        vehicle_type=vehicle_type,
    )

    # export files
    out_path.mkdir(parents=True, exist_ok=True)
    road_links.to_parquet(out_path / "links.gpq")
    logging.info(f"The total simulation time: {time.time() - start_time}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        num_of_chunk, num_of_cpu, od_file, vehicle_type = sys.argv[1:]
        sample_stride = 1
        main(
            int(num_of_chunk),
            int(num_of_cpu),
            od_file,
            vehicle_type,
            sample_stride=sample_stride,
        )
    except (IndexError, NameError):
        logging.info("Please enter num_of_chunk and num_of_cpu!")
    # main(1, 1)
