# %%
from pathlib import Path
from functools import partial

import geopandas as gpd
import pandas as pd
import numpy as np

from nird.utils import load_config
import nird.road_revised as func

import json
import warnings
from tqdm import tqdm

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])
tqdm.pandas()


# %%
def have_common_items(list1, list2):
    return bool(set(list1) & set(list2))


def bridge_recovery(
    day,
    damage_level,
    designed_capacity,
    current_speed,
    initial_speed,
    max_speed,
    bridge_recovery_dict,
    acc_speed,
    acc_capacity,
):
    if day == 0:  # occurrence of damage
        acc_speed = min(current_speed, max_speed)
        if damage_level == ["no", "minor", "moderate"]:
            acc_capacity = designed_capacity
        else:  # extensive/severe
            acc_capacity = 0.0
    elif day == 1:  # the first day of recovery
        acc_speed = initial_speed
        if damage_level != "no":
            recover_rate = bridge_recovery_dict[damage_level][day]
            if recover_rate > 0:
                acc_capacity = max(designed_capacity * recover_rate, acc_capacity)
    else:
        if damage_level != "no":
            recover_rate = bridge_recovery_dict[damage_level][day]
            if recover_rate > 0:
                acc_capacity = max(designed_capacity * recover_rate, acc_capacity)

    return acc_capacity, acc_speed


def ordinary_road_recovery(
    day,
    damage_level,
    designed_capacity,
    current_speed,
    initial_speed,
    max_speed,
    road_recovery_dict,
    acc_speed,
    acc_capacity,
):
    if day == 0:
        acc_speed = min(current_speed, max_speed)
        if damage_level in ["no", "minor", "moderate"]:
            acc_capacity = designed_capacity
        else:  # extensive/severe
            acc_capacity = 0.0
    elif day == 1:
        acc_speed = initial_speed
        if damage_level != "no":
            recovery_rate = road_recovery_dict[damage_level][day]
            if recovery_rate > 0:
                acc_capacity = max(designed_capacity * recovery_rate, acc_capacity)
    elif day == 2:
        if damage_level != "no":
            recovery_rate = road_recovery_dict[damage_level][day]
            if recovery_rate > 0:
                acc_capacity = max(designed_capacity * recovery_rate, acc_capacity)
    else:
        pass

    return acc_capacity, acc_speed


def main():
    # bridge recovery rates
    with open(base_path / "parameters" / "capt_minor.json", "r") as f:
        bridge_minor = json.load(f)
    with open(base_path / "parameters" / "capt_moderate.json", "r") as f:
        bridge_moderate = json.load(f)
    with open(base_path / "parameters" / "capt_extensive.json", "r") as f:
        bridge_extensive = json.load(f)
    with open(base_path / "parameters" / "capt_severe.json", "r") as f:
        bridge_severe = json.load(f)

    # on D-0
    bridge_minor.insert(0, 0.0)
    bridge_moderate.insert(0, 0.0)
    bridge_extensive.insert(0, 0.0)
    bridge_severe.insert(0, 0.0)

    bridge_recovery_dict = {
        "minor": bridge_minor,
        "moderate": bridge_moderate,
        "extensive": bridge_extensive,
        "severe": bridge_severe,
    }

    # road (other than bridge) recovery rates
    road_minor = [0.0, 1.0, 1.0]
    road_moderate = [0.0, 1.0, 1.0]
    road_extensive = [0.0, 1.0, 1.0]
    road_severe = [0.0, 0.5, 1.0]

    road_recovery_dict = {
        "minor": road_minor,
        "moderate": road_moderate,
        "extensive": road_extensive,
        "severe": road_severe,
    }

    # network parameters
    with open(base_path / "parameters" / "flow_cap_plph_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)

    # road links (with 8 added columns)
    road_links = gpd.read_parquet(
        base_path / "disruption_analysis_1129" / "road_links2.pq"
    )
    road_links = func.edge_reclassification_func(road_links)
    road_links["designed_capacity"] = (
        road_links.combined_label.map(flow_capacity_dict) * road_links.lanes * 24
    )

    """How to define disrupted links?
    - with extensive/severe damage levels
    - max_speed < current_speed
    """
    disrupted_links = (
        road_links.loc[
            (road_links.max_speed < road_links.current_speed)
            | (road_links.damage_level == "extensive")
            | (road_links.damage_level == "severe"),
            "id",
        ]
        .unique()
        .tolist()
    )

    # extract the disrupted od matrix
    partial_have_common_items = partial(have_common_items, list2=disrupted_links)
    od_path_file = pd.read_parquet(
        base_path.parent / "outputs" / "odpfc_partial_1128.pq"
    )
    od_path_file["disrupted"] = np.vectorize(partial_have_common_items)(
        od_path_file.path
    )
    disrupted_od = od_path_file[od_path_file["disrupted"]].reset_index(drop=True)
    disrupted_od.rename(columns={"flow": "Car21"}, inplace=True)
    disrupted_od = disrupted_od.head(10)  # debug: 76

    """
    recovery of road capacities (-> for mapping):
                D-0  D-15  D-30  D-60  D-90  D-110
    minor        0    1     1     1     1     1
    moderate     0    0     0.4   1     1     1
    extensive    0    0     0     0.66  1     1
    severe       0    0     0     0.17  0.67  1
    """

    # key variables for rerouting analysis:
    road_links["disrupted"] = 0
    road_links.loc[road_links.id.isin(disrupted_links), "disrupted"] = 1
    road_links["acc_flow"] = 0
    road_links["acc_capacity"] = road_links["current_capacity"]
    road_links["acc_speed"] = road_links["current_speed"]

    for day in range(111):  # Day 0-110
        (
            road_links["acc_capacity"],
            road_links["acc_speed"],
        ) = zip(
            *road_links.progress_apply(
                lambda row: (
                    bridge_recovery(
                        day,
                        row["damage_level"],
                        row["designed_capacity"],
                        row["current_speed"],
                        row["initial_flow_speeds"],
                        row["max_speed"],
                        bridge_recovery_dict,
                        row["acc_speed"],
                        row["acc_capacity"],
                    )
                    if row["disrupted"] == 1 and row["aveBridgeWidth"] > 0
                    else (
                        ordinary_road_recovery(
                            day,
                            row["damage_level"],
                            row["designed_capacity"],
                            row["current_speed"],
                            row["initial_flow_speeds"],
                            row["max_speed"],
                            road_recovery_dict,
                            row["acc_speed"],
                            row["acc_capacity"],
                        )
                        if row["disrupted"] == 1 and row["aveBridgeWidth"] == 0
                        else (row["acc_capacity"], row["acc_speed"])
                    )
                ),
                axis=1,
            )
        )

        # drop the disrupted links
        valid_road_links = road_links[
            (road_links.acc_capacity > 0) & (road_links.acc_speed > 0)
        ].reset_index(drop=True)

        # create network
        network = func.create_igraph_network(valid_road_links)

        # run rerouting analysis
        valid_road_links, isolation, odpfc = func.network_flow_model(
            valid_road_links, network, disrupted_od, flow_breakpoint_dict
        )

        # update key variables
        road_links.set_index("e_id", inplace=True)
        valid_road_links.set_index("e_id", inplace=True)
        road_links[["acc_flow", "acc_capacity", "acc_speed"]].update(
            valid_road_links[["acc_flow", "acc_capacity", "acc_speed"]]
        )
        road_links.reset_index(inplace=True)

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
                "path",
                "flow",
                "operating_cost_per_flow",
                "time_cost_per_flow",
                "toll_cost_per_flow",
            ],
        )

        # export rerouting results
        if day in [0, 15, 30, 60, 90, 110]:
            road_links.to_parquet(
                base_path.parent
                / "outputs"
                / "disruption_analysis"
                / f"edge_flows_{day}.pq"
            )
            isolation.to_csv(
                base_path.parent
                / "outputs"
                / "disruption_analysis"
                / f"isolated_odf_{day}.csv",
                index=False,
            )
            odpfc.to_csv(
                base_path.parent
                / "outputs"
                / "disruption_analysis"
                / f"odpfc_{day}.csv",
                index=False,
            )


if __name__ == "__main__":
    main()
