# %%
from pathlib import Path
import pandas as pd
import numpy as np
from nird.utils import load_config
from collections import defaultdict
import geopandas as gpd

import warnings

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])
res_path = Path(load_config()["paths"]["output_path"])

# %%
dlength_dict = defaultdict(lambda: defaultdict(list))
dcost_dict = defaultdict(lambda: defaultdict(list))
for i in range(1, 20):
    intersections_with_damage = pd.read_csv(
        res_path
        / "damage_analysis"
        / "20241229"
        / "final"
        / f"intersections_{i}_with_damage_values.csv"
    )
    # calculate min and max damaged length for each road segment
    # damage_fraction_min = (
    #     intersections_with_damage.filter(like="_fraction", axis=1)
    #     .replace(0, np.nan)
    #     .min(axis=1, skipna=True)
    # )
    # damage_fraction_min[damage_fraction_min.isnull()] = 0.0
    # damage_fraction_max = intersections_with_damage.filter(
    #     like="_fraction", axis=1
    # ).max(axis=1)

    damaged_length_min = intersections_with_damage.length  # * damage_fraction_min
    damaged_length_max = intersections_with_damage.length  # * damage_fraction_max

    intersections_with_damage["min_dlength"] = damaged_length_min
    intersections_with_damage["max_dlength"] = damaged_length_max

    # calculate min and max damage values for each road segment
    damage_cost_min = (
        intersections_with_damage.filter(regex="_damage_value_(min|max)$", axis=1)
        .replace(0, np.nan)
        .min(axis=1, skipna=True)
    )
    damage_cost_max = intersections_with_damage.filter(
        regex="_damage_value_(min|max)$", axis=1
    ).max(axis=1)

    intersections_with_damage["min_dcost"] = damage_cost_min
    intersections_with_damage["max_dcost"] = damage_cost_max

    for row in intersections_with_damage.itertuples():
        eid = row.e_id
        minl = row.min_dlength
        maxl = row.max_dlength
        minc = row.min_dcost
        maxc = row.max_dcost
        dlength_dict[eid]["min"].append(minl)
        dlength_dict[eid]["max"].append(maxl)
        dcost_dict[eid]["min"].append(minc)
        dcost_dict[eid]["max"].append(maxc)

# %%
# the average disrupted length of road links
min_dlength_dict = {}
max_dlength_dict = {}
min_dcost_dict = {}
max_dcost_dict = {}
for road, v in dlength_dict.items():
    ave_min = np.mean(v["min"])
    ave_max = np.mean(v["max"])
    min_dlength_dict[road] = ave_min
    max_dlength_dict[road] = ave_max

for road, v in dcost_dict.items():
    ave_min = np.mean(v["min"])
    ave_max = np.mean(v["max"])
    min_dcost_dict[road] = ave_min
    max_dcost_dict[road] = ave_max

# %%
intersections_with_damage["min_ave_dlength"] = intersections_with_damage.e_id.map(
    min_dlength_dict
).fillna(0.0)
intersections_with_damage["max_ave_dlength"] = intersections_with_damage.e_id.map(
    max_dlength_dict
).fillna(0.0)
intersections_with_damage["min_ave_dcost"] = intersections_with_damage.e_id.map(
    min_dcost_dict
).fillna(0.0)
intersections_with_damage["max_ave_dcost"] = intersections_with_damage.e_id.map(
    max_dcost_dict
).fillna(0.0)

res = intersections_with_damage[
    ["e_id", "min_ave_dlength", "max_ave_dlength", "min_ave_dcost", "max_ave_dcost"]
]

# %%
road_links = gpd.read_parquet(
    base_path / "networks" / "road" / "GB_road_links_with_bridges.gpq"
)
res2 = res.merge(
    road_links[
        [
            "e_id",
            "road_classification",
            "form_of_way",
            "road_classification_number",
            "trunk_road",
            "road_label",
            "urban",
        ]
    ],
    on="e_id",
    how="right",
)
res2 = res2.fillna(0)

# %%
# interpretations
for i in res2.columns[3:]:
    print(i)
    temp = res2.groupby(by=str(i), as_index=False).agg(
        {
            "min_ave_dlength": "sum",
            # "max_ave_dlength": "sum",
            "min_ave_dcost": "sum",
            "max_ave_dcost": "sum",
        }
    )
    temp.rename(columns={"min_ave_dlength": "mean_dlength"}, inplace=True)
    # temp["mean_ave_dlength_meter"] = temp.filter(like="dlength").mean(axis=1)
    temp["mean_dcost"] = temp.filter(like="dcost").mean(axis=1)
    # temp.rename(
    #     columns={"min_ave_dlength": "min", "max_ave_dlength": "max"},
    #     inplace=True,
    # )
    # temp["mean"] = temp[["min", "max"]].mean(axis=1)
    # breakpoint()
    temp.to_csv(
        res_path / "damage_analysis" / "20241229" / f"exp_vul_{i}.csv",
        index=False,
    )
    # if i == "road_classification_number":
    #     temp_min_dict = temp.set_index(i)["min_ave_dlength"]
    #     temp_max_dict = temp.set_index(i)["max_ave_dlength"]
    #     road_links["min_ave_dlength"] = road_links[str(i)].map(temp_min_dict)
    #     road_links["max_ave_dlength"] = road_links[str(i)].map(temp_max_dict)
    #     road_links.to_parquet(
    #         res_path / "damage_analysis" / "20241229" / "acc_dlength.gpq"
    #     )
