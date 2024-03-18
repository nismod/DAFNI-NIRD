# %%
from pathlib import Path

import pandas as pd
import geopandas as gpd  # type: ignore

from utils import load_config
import functions as func

import json
import warnings

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])


# %%
"""
list of inputs:
 - Parameter dicts.
 - OS open roads.
 - ETISPLUS_urban_roads: to create urban mask.
 - Population-weighed centroids of admin units.
 - O-D matrix (*travel to work by car).

"""
# model parameters
with open(base_path / "parameters" / "flow_breakpoint_dict.json", "r") as f:
    flow_breakpoint_dict = json.load(f)

with open(base_path / "parameters" / "flow_cap_dict.json", "r") as f:
    flow_capacity_dict = json.load(f)

with open(base_path / "parameters" / "free_flow_speed_dict.json", "r") as f:
    free_flow_speed_dict = json.load(f)

with open(base_path / "parameters" / "min_speed_cap.json", "r") as f:
    min_speed_cap = json.load(f)

with open(base_path / "parameters" / "urban_speed_cap.json", "r") as f:
    urban_speed_cap = json.load(f)

# OS open roads
osoprd_link = gpd.read_parquet(
    base_path / "networks" / "road" / "osoprd_road_links.geoparquet"
)
osoprd_node = gpd.read_parquet(
    base_path / "networks" / "road" / "osoprd_road_nodes.geoparquet"
)

# ETISPLUS roads
etisplus_road_links = gpd.read_parquet(
    base_path / "networks" / "road" / "etisplus_road_links.geoparquet"
)
etisplus_urban_roads = etisplus_road_links[["Urban", "geometry"]]
etisplus_urban_roads = etisplus_urban_roads[etisplus_urban_roads["Urban"] == 1]

# population-weighted centroids (combined spatial units)
zone_centroids = gpd.read_parquet(
    base_path / "census_datasets" / "admin_pwc" / "zone_pwc.geoparquet"
)

# O-D matrix (2011)
# original O-D (cross-border trips, by cars, 2011)
# Area of usual residence:
# OA (Scotland, 2011) & OA (ENW, 2011)
# Area of workplace:
# Workplace Zone (ENW, 2011),
# MOSA (ENW, 2011)
# Intermediate Zones (Scotland, 2001)
# OA (Scotland, 2011)
od_df = pd.read_csv(base_path / "census_datasets" / "od_matrix" / "od_gb_2011.csv")
print(f"total flows: {od_df.car.sum()}")  # 14_203_635 trips/day

# %%
# select major roads
road_link_file, road_node_file = func.select_partial_roads(
    road_links=osoprd_link,
    road_nodes=osoprd_node,
    col_name="road_classification",
    list_of_values=["A Road", "B Road", "Motorway"],
)

# classify the selected major road links into urban/suburban
urban_mask = func.create_urban_mask(etisplus_urban_roads)
road_link_file = func.label_urban_roads(road_link_file, urban_mask)

# find the nearest road node for each zone
zone_to_node = func.find_nearest_node(zone_centroids, road_node_file)

# attach od info of each zone to their nearest road network nodes
list_of_origin_node, destination_node_dict, flow_dict = func.od_interpret(
    od_df,
    zone_to_node,
    col_origin="Area of usual residence",
    col_destination="Area of workplace",
    col_count="car",
)
# extract identical origin nodes
list_of_origin_node = list(set(list_of_origin_node))
list_of_origin_node.sort()

# %%
# network creation (igragh)
name_to_index = {name: index for index, name in enumerate(road_node_file.nd_id)}
index_to_name = {value: key for key, value in name_to_index.items()}
test_net_ig = func.create_igraph_network(
    name_to_index, road_link_file, road_node_file, free_flow_speed_dict
)
# edge_idx -> edge_name
edge_index_to_name = {idx: name for idx, name in enumerate(test_net_ig.es["edge_name"])}

# network initailisation
(
    road_link_file,
    road_type_dict,
    form_dict,
    isurban_dict,
    length_dict,
    acc_flow_dict,
    acc_capacity_dict,
    speed_dict,
) = func.initialise_igraph_network(
    road_link_file,
    flow_capacity_dict,
    free_flow_speed_dict,
    "e_id",
    "road_classification",
    "form_of_way",
    "urban",
)

# %%
# flow simulation
speed_dict, acc_flow_dict, acc_capacity_dict = func.network_flow_model(
    test_net_ig,
    list_of_origin_node,
    flow_dict,
    destination_node_dict,
    name_to_index,
    edge_index_to_name,
    road_type_dict,
    form_dict,
    isurban_dict,
    length_dict,
    acc_flow_dict,
    acc_capacity_dict,
    speed_dict,
    "e_id",
)

# %%
# append estimations about: speed, flows, and remaining capacities
road_link_file.ave_flow_rate = road_link_file.e_id.map(speed_dict)
road_link_file.acc_flow = road_link_file.e_id.map(acc_flow_dict)
road_link_file.acc_capacity = road_link_file.e_id.map(acc_capacity_dict)

# change field types
road_link_file.acc_flow = road_link_file.acc_flow.astype(int)
road_link_file.acc_capacity = road_link_file.acc_capacity.astype(int)

# %%
# export file
road_link_file.to_file(
    base_path / "outputs" / "GB_flows_updated.gpkg",
    driver="GPKG",
)
