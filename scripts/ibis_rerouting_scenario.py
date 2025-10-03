import sys
from pathlib import Path
import pandas as pd
import warnings
from tqdm import tqdm
import logging
import ibis as ib
import nird.road_revised as func
from nird.utils import load_config
import json

ib.options.interactive = True

base_path = Path(load_config()["paths"]["soge_clusters"])
warnings.simplefilter("ignore")
tqdm.pandas()


def main(
    depth_key,
    event_key,
    day,
    number_of_cpu,
):
    # Load network parameters
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)
    cDict = {}

    logging.info("Loading inputs...")
    # road links
    road_links = ib.read_parquet(
        base_path.parent
        / "results"
        / "disruption_analysis"
        / "revision"
        / "ibis_results"
        / f"{depth_key}"
        / f"road_links_depth{depth_key}_event{event_key}_day{day}.gpq"
    )
    road_links = (
        road_links.cast({"geometry": "GEOMETRY"})
        .execute()
        .set_geometry("geometry")
        .set_crs("epsg:27700")
    )
    if day == 0:  # event_day
        road_links["acc_speed"] = road_links[["acc_speed", "max_speed"]].min(axis=1)

    # disrupted od
    disrupted_od = pd.read_parquet(
        base_path.parent
        / "results"
        / "disruption_analysis"
        / "revision"
        / "ibis_results"
        / f"{depth_key}"
        / f"disrupted_od_depth{depth_key}_event{event_key}_day{day}.pq"
    )
    disrupted_od = disrupted_od[
        (disrupted_od.flow.notnull()) & (disrupted_od.min_link_capacity.notnull())
    ].reset_index(drop=True)
    disrupted_od["min_link_capacity"] = disrupted_od["min_link_capacity"].clip(lower=0)
    disrupted_od["Car21"] = (disrupted_od.flow - disrupted_od.min_link_capacity).clip(
        lower=0
    )
    disrupted_od.Car21 = disrupted_od.Car21.round(0).astype(int)

    # pre-event cost matrix
    pre_time = (disrupted_od.Car21 * disrupted_od.time_cost_per_flow).sum()
    pre_fuel = (disrupted_od.Car21 * disrupted_od.operating_cost_per_flow).sum()
    pre_toll = (disrupted_od.Car21 * disrupted_od.toll_cost_per_flow).sum()
    pre_cost = pre_time + pre_fuel + pre_toll

    logging.info("Creating network...")
    valid_road_links = road_links[
        (road_links.acc_capacity > 0) & (road_links.acc_speed > 0)
    ].reset_index(drop=True)
    network, valid_road_links = func.create_igraph_network(valid_road_links)

    logging.info("Running flow simulation...")
    logging.info(f"The total disrupted flow is: {disrupted_od.Car21.sum()}")
    valid_road_links, isolation, _, (time, fuel, toll, total_cost) = (
        func.network_flow_model(
            valid_road_links,
            network,
            disrupted_od[["origin_node", "destination_node", "Car21"]],
            flow_breakpoint_dict,
            num_of_cpu=number_of_cpu,
        )
    )
    logging.info(f"The total pre-event cost: £ million {pre_cost/1e6}")
    logging.info(f"The total after-event cost: £ million {total_cost/1e6}")

    # calculate rerouting cost matrix
    rerouting_time = (time - pre_time) / 1e6
    rerouting_fuel = (fuel - pre_fuel) / 1e6
    rerouting_toll = (toll - pre_toll) / 1e6
    total_rerouting = (total_cost - pre_cost) / 1e6
    logging.info(
        f"The rerouting cost for depth{depth_key}-event{event_key}-scenario{day}: "
        f"£ million {total_rerouting}"
    )
    cDict[day] = [rerouting_time, rerouting_fuel, rerouting_toll, total_rerouting]

    logging.info("Saving results to disk...")
    out_path = (
        base_path.parent
        / "results"
        / "rerouting_analysis"
        / "revision"
        / str(depth_key)  # e.g.,30
        / str(event_key)  # e.g.,17
    )
    out_path.mkdir(parents=True, exist_ok=True)

    # isolations
    isolation_df = pd.DataFrame(
        isolation, columns=["origin_node", "destination_node", "Car21"]
    )
    isolation_df.to_csv(out_path / f"test_trip_isolations_{day}.csv", index=False)

    # edge flows
    road_links = road_links.set_index("e_id")
    road_links.update(valid_road_links.set_index("e_id")["acc_flow"])
    road_links.reset_index(drop=True, inplace=True)
    road_links["change_flow"] = road_links.acc_flow - road_links.current_flow
    road_links.to_parquet(out_path / f"test_edge_flows_{day}.gpq", engine="pyarrow")

    # rerouting costs over all scenarios (days)
    cost_df = pd.DataFrame.from_dict(
        cDict,
        orient="index",
        columns=[
            "rerouting_time",
            "rerouting_fuel",
            "rerouting_toll",
            "total_rerouting_million",
        ],
    ).reset_index()
    cost_df.rename(columns={"index": "scenario"}, inplace=True)
    cost_df.to_csv(out_path / f"test_rerouting_cost_{day}.csv", index=False)


if __name__ == "__main__":
    try:
        depth_key = int(sys.argv[1])
    except (IndexError, ValueError):
        logging.info(
            "Error: Please provide the flood depth for road closure "
            "(e.g., 15, 30 or 60 cm) as the first argument!"
        )
        sys.exit(1)
    try:
        event_key = int(sys.argv[2])
    except (IndexError, ValueError):
        logging.info("Error: which flood event!")
        sys.exit(1)
    try:
        day = int(sys.argv[3])
    except (IndexError, ValueError):
        logging.info("Error: which scenario!")
        sys.exit(1)
    try:
        number_of_cpu = int(sys.argv[4])
    except (IndexError, ValueError):
        logging.info("Error: number of cpus!")
        sys.exit(1)

    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    main(depth_key, event_key, day, number_of_cpu)
