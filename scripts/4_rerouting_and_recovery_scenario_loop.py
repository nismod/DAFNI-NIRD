# %%
import sys
import json
import warnings
import gc

from pathlib import Path
from typing import Tuple, Dict
import logging

import geopandas as gpd
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import nird.road_revised as func
from nird.utils import load_config, get_flow_on_edges
import duckdb

# %%
warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["soge_clusters"])
tqdm.pandas()


# %%
def bridge_recovery(
    day: int,
    damage_level: str,
    pre_event_capacity: float,
    acc_capacity: float,
    bridge_recovery_dict: Dict,
) -> float:
    if damage_level != "no":  # minor, moderate, extensive, severe
        recovery_rate = bridge_recovery_dict.get(damage_level, [])[day]
        acc_capacity = pre_event_capacity * recovery_rate
    return acc_capacity


def ordinary_road_recovery(
    day: int,
    damage_level: str,
    pre_event_capacity: float,
    acc_capacity: float,
    road_recovery_dict: Dict,
) -> float:
    if damage_level != "no":  # minor, moderate, extensive, severe
        recovery_rate = road_recovery_dict.get(damage_level, [])[day]
        acc_capacity = pre_event_capacity * recovery_rate
    return acc_capacity


def load_scenarios(base_path: Path) -> Tuple[Dict, Dict]:
    """Load recovery rates for bridges and ordinary roads."""
    df = pd.read_csv(
        base_path / "tables" / "recovery design_updated.csv",
    )

    bridge_recovery_dict = defaultdict(list)
    road_recovery_dict = defaultdict(list)
    scenarios = []
    conditions = []
    for _, row in df.iterrows():
        bridge_recovery_dict["minor"].append(row["bridge_minor"])
        bridge_recovery_dict["moderate"].append(row["bridge_moderate"])
        bridge_recovery_dict["extensive"].append(row["bridge_extensive"])
        bridge_recovery_dict["severe"].append(row["bridge_severe"])
        road_recovery_dict["minor"].append(row["road_minor"])
        road_recovery_dict["moderate"].append(row["road_moderate"])
        road_recovery_dict["extensive"].append(row["road_extensive"])
        road_recovery_dict["severe"].append(row["road_severe"])
        scenarios.append(int(row["scenario"]))
        conditions.append(int(row["event_day"]))

    return (bridge_recovery_dict, road_recovery_dict, scenarios, conditions)


def main(
    depth_key,
    flood_key,
    num_of_chunk,
    num_of_cpu,
):
    logging.info("Start...")
    db_path = base_path / "dbs" / f"recovery_{depth_key}_{flood_key}.duckdb"
    logging.info(f"Database path is: {db_path}")

    # Load network parameters
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)

    # Load recovery scenarios
    (
        bridge_recovery_dict,
        road_recovery_dict,
        scenarios,  # list of scenarios
        conditions,  # condition == 1: day 0, otherwise: day > 0
    ) = load_scenarios(base_path)

    # Load pre-identified odpfc (containing flooded links)
    disrupted_candidates = pd.read_parquet(
        base_path.parent
        / "results"
        / "disruption_analysis"
        / "revision"
        / "od"
        / f"odpfc_{depth_key}_{flood_key}.pq"
    )
    disrupted_candidates["od_id"] = disrupted_candidates.index  # numbering od pairs
    # Load road links with damage (e.g., flood depth and damage level)
    road_links = gpd.read_parquet(
        base_path.parent
        / "results"
        / "disruption_analysis"
        / "revision"
        / str(depth_key)
        / "links"
        / f"road_links_{flood_key}.gpq"
    )
    road_links["breakpoint_flows"] = road_links["combined_label"].map(
        flow_breakpoint_dict
    )
    initial_road_links_cols = road_links.columns

    # Recovery analysis loop
    cDict = {}
    # Load link recovery scenarios (both capacity and speed)
    for scenario_idx in range(len(scenarios)):
        logging.info(f"Rerouting Analysis on Scenario-{scenario_idx} of recovery...")
        event_day = conditions[scenario_idx]
        logging.info(f"Updating edge capacities on D-{scenario_idx} of recovery...")
        road_links["acc_capacity"] = road_links["current_capacity"]
        road_links["acc_capacity"] = road_links.apply(
            lambda row: (
                bridge_recovery(
                    scenario_idx,
                    row["damage_level_max"],
                    row["current_capacity"],
                    row["acc_capacity"],
                    bridge_recovery_dict,
                )
                if row["road_label"] == "bridge"
                else (
                    ordinary_road_recovery(
                        scenario_idx,
                        row["damage_level_max"],
                        row["current_capacity"],
                        row["acc_capacity"],
                        road_recovery_dict,
                    )
                )
            ),
            axis=1,
        )
        # Extract disrupted od after road recovery
        logging.info("Extracting disrupted OD pairs...")
        disrupted_od = disrupted_candidates.copy()

        # conduct exploded chunks
        total_disrupted = len(disrupted_od)
        if total_disrupted == 0:
            logging.info("No flooded od pairs detected for current scenario.")
            continue

        max_chunk_size = 10_000
        if max_chunk_size > total_disrupted:
            chunk_size = total_disrupted
        else:
            n_chunk = min(100, max(1, total_disrupted // max_chunk_size))
            chunk_size = max(1, total_disrupted // n_chunk)
        logging.info(f"disrupted_od size: {total_disrupted}")
        logging.info(f"chunk_size: {chunk_size}")

        # create Duckdb to store mid-outputs
        conn = duckdb.connect(db_path)
        conn.execute("DROP TABLE IF EXISTS od_results")  # reset table
        conn.execute("DROP TABLE IF EXISTS edge_flows")  # reset table
        first = True
        for start in tqdm(
            range(0, total_disrupted, chunk_size),
            desc="Processing chunks",
            unit="chunk",
        ):
            chunk = disrupted_od.iloc[start : start + chunk_size].copy()
            chunk = chunk.explode("flood_links")  # list of e_id
            if chunk.empty:
                continue
            chunk = chunk.merge(
                road_links[["e_id", "acc_capacity"]],
                how="left",
                left_on="flood_links",
                right_on="e_id",
            )
            od_df = (
                chunk.groupby(by=["od_id"])
                .agg(
                    {
                        "acc_capacity": "min",
                    }
                )
                .reset_index()
            )
            if first:
                conn.register("od_df", od_df)
                conn.execute("CREATE TABLE od_results AS SELECT * FROM od_df")
                first = False
            else:
                conn.append("od_results", od_df)
            del chunk, od_df
            gc.collect()

        logging.info("Aggregating final results...")
        # to retrieve min edge capacity for each od
        min_capacity = conn.execute(
            """
            SELECT od_id, MIN(acc_capacity) AS acc_capacity
            FROM od_results
            GROUP BY od_id
        """
        ).df()
        conn.close()
        logging.info("Completing chunk process...")

        # calculate disrupted flow
        disrupted_od = disrupted_od.merge(min_capacity, how="left", on="od_id")
        disrupted_od["disrupted_flow"] = (
            disrupted_od["flow"] - disrupted_od["acc_capacity"]
        ).clip(lower=0)
        disrupted_od = disrupted_od[disrupted_od["disrupted_flow"] > 0].reset_index(
            drop=True
        )
        logging.info(f"The total disrupted flows: {disrupted_od.disrupted_flow.sum()}")

        # estimate the pre-event cost matrix for disrupted flows
        pre_time = (disrupted_od.disrupted_flow * disrupted_od.time_cost_per_flow).sum()
        pre_operate = (
            disrupted_od.disrupted_flow * disrupted_od.operating_cost_per_flow
        ).sum()
        pre_toll = (disrupted_od.disrupted_flow * disrupted_od.toll_cost_per_flow).sum()
        total_pre_cost = pre_time + pre_operate + pre_toll

        # Restore capacity for non-disrupted roads
        logging.info("Calibrate capacity for non-flooded links...")
        disrupted_edge_flow = get_flow_on_edges(
            disrupted_od, "e_id", "path", "disrupted_flow"
        )

        road_links = road_links.merge(disrupted_edge_flow, on="e_id", how="left")
        road_links["disrupted_flow"] = road_links["disrupted_flow"].fillna(0)

        """ Update road link attributes for rerouting analysis
        """
        road_links.current_capacity = road_links.current_capacity.round(0).astype(int)
        road_links.acc_capacity = road_links.acc_capacity.round(0).astype(int)
        road_links.current_flow = road_links.current_flow.round(0).astype(int)
        road_links.disrupted_flow = road_links.disrupted_flow.round(0).astype(int)

        road_links["acc_capacity"] = (
            road_links["acc_capacity"] + road_links["disrupted_flow"]
        )
        road_links["acc_flow"] = (
            road_links["current_flow"] - road_links["disrupted_flow"]
        )

        logging.info("Updating road speed limits...")
        road_links["acc_speed"] = road_links.apply(
            lambda x: func.update_edge_speed(
                x["combined_label"],  # constant
                x["acc_flow"],  # variable
                x["initial_flow_speeds"],  # constant
                x["min_flow_speeds"],  # constant
                x["breakpoint_flows"],  # constant
            ),
            axis=1,
        )
        if event_day == 1:  # apply speed constraint to every road
            road_links["acc_speed"] = road_links[["acc_speed", "max_speed"]].min(axis=1)
        if (
            event_day == 2
        ):  # only apply speed constraint to roads with flooddepth (2-6) meters
            mask = (road_links["flood_depth_max"] >= 2) & (
                road_links["flood_depth_max"] < 6
            )
            road_links.loc[mask, "acc_speed"] = road_links.loc[
                mask, ["acc_speed", "max_speed"]
            ].min(axis=1)
        if event_day == 3:  # only for roads > 6 meters
            mask = road_links["flood_depth_max"] >= 6
            road_links.loc[mask, "acc_speed"] = road_links.loc[
                mask, ["acc_speed", "max_speed"]
            ].min(axis=1)

        # create network (time-consuming when updating network edge index)
        logging.info("Creating igraph network...")
        valid_road_links = road_links[
            (road_links["acc_capacity"] > 0) & (road_links["acc_speed"] > 0)
        ].reset_index(drop=True)
        network, valid_road_links = func.create_igraph_network(valid_road_links)

        # !!! make sure to pass disrupted flow for rerouting analysis
        disrupted_od.rename(columns={"disrupted_flow": "Car21"}, inplace=True)

        # Run flow model
        logging.info("Running flow simulation...")
        (
            valid_road_links,
            isolation,
            _,
            (post_time, post_operate, post_toll, total_post_cost),
        ) = func.network_flow_model(
            valid_road_links,  # update this one
            network,
            disrupted_od[
                ["origin_node", "destination_node", "Car21"]
            ],  # update this one
            flow_breakpoint_dict,
            num_of_chunk,
            num_of_cpu,
            db_path,
        )

        # estimate rerouting cost matrix
        rer_time = post_time - pre_time
        rer_operate = post_operate - pre_operate
        rer_toll = post_toll - pre_toll
        rerouting_cost = rer_time + rer_operate + rer_toll
        logging.info(
            f"The original travel costs for disrupted od: £ million {total_pre_cost/ 1e6}"
        )
        logging.info(
            f"The total travel costs after disruption: £ million {total_post_cost/ 1e6}"
        )
        logging.info(
            f"The rerouting cost for scenario {scenario_idx}: £ million {rerouting_cost / 1e6}"
        )

        logging.info("Saving results to disk...")
        out_path = (
            base_path.parent
            / "results"
            / "rerouting_analysis"
            / "revision"
            / str(depth_key)  # e.g.,30
            / str(flood_key)  # e.g.,17
        )
        out_path.mkdir(parents=True, exist_ok=True)

        # rerouting costs
        cDict[scenario_idx] = [rer_time, rer_operate, rer_toll, rerouting_cost]
        cost_df = pd.DataFrame.from_dict(
            cDict,
            orient="index",
            columns=["rer_time", "rer_operate", "rer_toll", "rerouting_cost"],
        ).reset_index()
        cost_df.rename(columns={"index": "scenario"}, inplace=True)
        cost_df.to_csv(out_path / f"rerouting_cost_{scenario_idx}.csv", index=False)

        # trip isolations
        isolation_df = pd.DataFrame(
            isolation,
            columns=[
                "origin_node",
                "destination_node",
                "Car21",
            ],
        )
        isolation_df = isolation_df[
            (isolation_df.origin_node != isolation_df.destination_node)
            & (isolation_df.Car21 > 0)
        ].reset_index()
        isolation_df.to_csv(
            out_path / f"trip_isolations_{scenario_idx}.csv",
            index=False,
        )

        # edge flows
        road_links = road_links.set_index("e_id")
        road_links.update(valid_road_links.set_index("e_id")["acc_flow"])
        road_links = road_links.reset_index()
        road_links["change_flow"] = road_links["acc_flow"] - road_links["current_flow"]
        road_links.to_parquet(out_path / f"edge_flows_{scenario_idx}.gpq")

        # reset road_links for next scenario
        road_links = road_links[initial_road_links_cols]

        del disrupted_od
        del disrupted_edge_flow
        del valid_road_links
        gc.collect()

    logging.info("Saving overall rerouting costs to disk...")
    cost_df = pd.DataFrame.from_dict(
        cDict,
        orient="index",
        columns=["rer_time", "rer_operate", "rer_toll", "rerouting_cost"],
    ).reset_index()
    cost_df.rename(columns={"index": "scenario"}, inplace=True)
    cost_df.to_csv(out_path / "cost_matrix_by_scenario.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(process)d %(filename)s %(message)s", level=logging.INFO
    )
    try:
        depth_key, event_key, num_of_chunk, num_of_cpu = sys.argv[1:]
        main(int(depth_key), int(event_key), int(num_of_chunk), int(num_of_cpu))
    except (IndexError, ValueError):
        logging.info(
            "Please provide inputs: depth_key, event_key, num_of_chunk, and num_of_cpu!"
        )
        sys.exit(1)
