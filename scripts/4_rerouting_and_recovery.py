from pathlib import Path
from functools import partial

import geopandas as gpd
import pandas as pd
import numpy as np

from nird.utils import load_config
import nird.road_revised as func

import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])


def have_common_items(list1, list2):
    return bool(set(list1) & set(list2))


def main():
    with open(base_path / "parameters" / "capt_minor.json", "r") as f:
        capacity_per_day_minor = json.load(f)
    with open(base_path / "parameters" / "capt_moderate.json", "r") as f:
        capacity_per_day_moderate = json.load(f)
    with open(base_path / "parameters" / "capt_extensive.json", "r") as f:
        capacity_per_day_extensive = json.load(f)
    with open(base_path / "parameters" / "capt_severe.json", "r") as f:
        capacity_per_day_severe = json.load(f)
    with open(base_path / "parameters" / "flow_cap_plph_dict.json", "r") as f:
        flow_capacity_dict = json.load(f)
    with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
        flow_breakpoint_dict = json.load(f)

    assert (
        len(capacity_per_day_minor)
        == len(capacity_per_day_moderate)
        == len(capacity_per_day_extensive)
        == len(capacity_per_day_severe)
    ), "inconsistent recovery rates!"

    # road links (with 8 add info)
    road_links = gpd.read_parquet(
        base_path / "disruption_analysis_1129" / "road_links.pq"
    )
    disrupted_links = (
        road_links.loc[road_links.damage_level != "no", "id"].unique().tolist()
    )
    partial_have_common_items = partial(have_common_items, list2=disrupted_links)
    od_path_file = pd.read_parquet(
        base_path.parent / "outputs" / "odpfc_partial_1128.pq"
    )
    od_path_file["disrupted"] = np.vectorize(partial_have_common_items)(
        od_path_file.path
    )
    disrupted_od = od_path_file[od_path_file["disrupted"]].reset_index(drop=True)
    disrupted_od.rename(columns={"flow": "Car21"}, inplace=True)
    disrupted_od = disrupted_od.head(10)  # for debug

    """
    recovery process design (-> for mapping):
                D-0  D-15  D-30  D-60  D-90  D-110
    minor        0    1     1     1     1     1
    moderate     0    0     0.4   1     1     1
    extensive    0    0     0     0.66  1     1
    severe       0    0     0     0.17  0.67  1
    """

    # key variables for rerouting analysis:
    # acc_flow, acc_capacity, acc_speed
    road_links["acc_flow"] = 0
    road_links = func.edge_reclassification_func(road_links)
    road_links["designed_capacity"] = (
        road_links.combined_label.map(flow_capacity_dict) * road_links.lanes * 24
    )
    for day in range(len(capacity_per_day_minor) + 1):  # (0 - 110) - 111 days
        if day == 0:  # D-0 (occurrence of damage)
            """edge capacities:
            - no/minor/moderate: remaining capacities
            - extensive/severe: closed (zero capacity)
            """
            road_links["acc_capacity"] = np.where(
                (road_links["damage_level"] == "extensive")
                | (road_links["damage_level"] == "severe"),
                0,
                road_links["current_capacity"],
            )

            """traffic speeds:
            - max_speed, current_speed
            - no/minor/moderate: min(max_speed, current_speed)
            - extensive/severe: closed (zero speed)
            """
            road_links["acc_speed"] = np.where(
                (road_links["damage_level"] == "extensive")
                | (road_links["damage_level"] == "severe"),
                0,
                np.minimum(road_links["current_speed"], road_links["max_speed"]),
            )
        else:  # including idle time & traffic reinstatement time
            """edge capacities: max(acc_capacity, recovered_capacity)"""
            cap_minor = capacity_per_day_minor[day - 1]
            cap_moderate = capacity_per_day_moderate[day - 1]
            cap_extensive = capacity_per_day_extensive[day - 1]
            cap_severe = capacity_per_day_severe[day - 1]

            road_links.loc[road_links["damage_level"] == "no", "acc_capacity"] = (
                road_links["designed_capacity"]
            )
            road_links.loc[road_links["damage_level"] == "minor", "acc_capacity"] = (
                np.maximum(
                    road_links.loc[
                        road_links["damage_level"] == "minor", "designed_capacity"
                    ]
                    * cap_minor,
                    road_links.loc[
                        road_links["damage_level"] == "minor", "acc_capacity"
                    ],
                )
            )
            road_links.loc[road_links["damage_level"] == "moderate", "acc_capacity"] = (
                np.maximum(
                    road_links.loc[
                        road_links["damage_level"] == "moderate", "designed_capacity"
                    ]
                    * cap_moderate,
                    road_links.loc[
                        road_links["damage_level"] == "moderate", "acc_capacity"
                    ],
                )
            )
            road_links.loc[
                road_links["damage_level"] == "extensive", "acc_capacity"
            ] = np.maximum(
                road_links.loc[
                    road_links["damage_level"] == "extensive", "designed_capacity"
                ]
                * cap_extensive,
                road_links.loc[
                    road_links["damage_level"] == "extensive", "acc_capacity"
                ],
            )
            road_links.loc[road_links["damage_level"] == "severe", "acc_capacity"] = (
                np.maximum(
                    road_links.loc[
                        road_links["damage_level"] == "severe", "designed_capacity"
                    ]
                    * cap_severe,
                    road_links.loc[
                        road_links["damage_level"] == "severe", "acc_capacity"
                    ],
                )
            )

            """edge speeds: initial_speeds"""
            road_links["acc_speed"] = road_links["initial_flow_speeds"]

        # run rerouting analysis
        network, road_links = func.create_igraph_network(road_links)
        road_links, isolation, odpfc = func.network_flow_model(
            road_links, network, disrupted_od, flow_breakpoint_dict
        )

        odpfc = pd.DataFrame(
            odpfc,
            columns=[
                "origin_node",
                "destination_node",
                "path",
                "flow",
                "operating_cost",
                "time_cost",
                "toll_cost",
            ],
        )

        isolation = pd.DataFrame(
            isolation,
            columns=[
                "origin_node",
                "destination_node",
                "path",
                "flow",
                "operating_cost",
                "time_cost",
                "toll_cost",
            ],
        )

        # export rerouting results
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
            base_path.parent / "outputs" / "disruption_analysis" / f"odpfc_{day}.csv",
            index=False,
        )


if __name__ == "__main__":
    main()
