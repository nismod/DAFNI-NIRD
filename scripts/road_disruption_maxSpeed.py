"""This script calculate the maximum vehicle speed on the flooded roads
based on the flood intensity (depth)
inputs: flood intensity (m) -> mm
outputs: maximum vehicle speed (mph)
"""

import os
import warnings
from pathlib import Path

import json
import pandas as pd

import geopandas as gpd
from nird.utils import load_config
import nird.constants as cons


warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])
damage_path = Path(load_config()["paths"]["damage_analysis"])
flood_path = damage_path / "outputs"


def identify_flood_scenarios():
    case_paths = []
    for root, _, files in os.walk(flood_path):
        for file in files:
            if "gp" in file and file.endswith(".csv"):
                file_dir = os.path.join(root, file)
                case_paths.append(file_dir)
    return case_paths


def max_edge_speed_under_flood(depth: float) -> float:
    """Calculate the maximum vehicle speed (mph) based on flood depth (mm)"""
    if depth < 300:  # mm
        value = 0.0009 * depth**2 - 0.5529 * depth + 86.9448  # kph
        return value / cons.CONV_MILE_TO_KM
    else:
        return 0.0  # mph


def road_reclassification(road_classification, form_of_way):
    if road_classification == "Motorway":
        return "M"
    elif road_classification == "B Road":
        return "B"
    elif road_classification == "A Road" and form_of_way == "Single Carriageway":
        return "A_single"
    else:
        return "A_dual"


def calculate_road_max_speed(road_link_file, min_speed_dict, output_path):
    road_link_file["label"] = road_link_file.apply(
        lambda x: road_reclassification(x.road_classification, x.form_of_way), axis=1
    )
    road_link_file["min_speed_mph"] = road_link_file["label"].map(min_speed_dict)
    road_min_speed_dict = road_link_file.set_index("id")["min_speed_mph"].to_dict()
    case_paths = identify_flood_scenarios()
    for case_path in case_paths:
        case_key = case_path.split("\\")[-1][:-15]
        temp = pd.read_csv(case_path)
        e_ids = temp.iloc[:, 0]
        intensites = temp.iloc[:, 1]
        case_df = pd.DataFrame({"id": e_ids, "fld_depth_meter": intensites})
        case_df["min_speed_mph"] = case_df.id.map(road_min_speed_dict)
        case_df["max_speed_mph"] = case_df.fld_depth_meter.apply(
            lambda x: max_edge_speed_under_flood(x * 1000)
        )
        case_df["max_speed_mph_adjusted"] = case_df.max_speed_mph
        case_df.loc[
            case_df.max_speed_mph < case_df.min_speed_mph, "max_speed_mph_adjusted"
        ] = 0.0
        case_df["max_speed_mph_adjusted"] = case_df["max_speed_mph_adjusted"].round(1)
        case_df.to_csv(output_path / f"{case_key}.csv", index=False)


def main():
    # attach minimum speed to each raod edge according to road_classification and form_of_ways
    road_link_file = gpd.read_parquet(
        base_path / "networks" / "road" / "road_link_file.geoparquet"
    )
    # min speed limit
    with open(base_path / "parameters" / "min_speed_cap.json", "r") as f:
        min_speed_dict = json.load(f)
    output_path = base_path / "disruption_analysis"

    # calculate the maximum vehicle speed under floods
    calculate_road_max_speed(road_link_file, min_speed_dict, output_path)


if __name__ == "__main__":
    main()
