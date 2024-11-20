# %%
# from typing import Tuple, List
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore
import nird.constants as cons
from nird.utils import load_config
import nird.road_revised as func
import warnings
import json

warnings.simplefilter("ignore")
base_path = Path(load_config()["paths"]["base_path"])

# %%
# OSM roads
osm_links = gpd.read_parquet(base_path / "networks" / "road" / "osm_road_links.pq")
osm_nodes = gpd.read_parquet(base_path / "networks" / "road" / "osm_road_nodes.pq")

# OS open roads
# osoprd_link = gpd.read_parquet(
#     base_path / "networks" / "road" / "osoprd_road_links.geoparquet"
# )
# osoprd_node = gpd.read_parquet(
#    base_path / "networks" / "road" / "osoprd_road_nodes.geoparquet"
# )

# ETISPLUS roads
etisplus_road_links = gpd.read_parquet(
    base_path / "networks" / "road" / "etisplus_road_links.geoparquet"
)
etisplus_urban_roads = etisplus_road_links[["Urban", "geometry"]]
etisplus_urban_roads = etisplus_urban_roads[etisplus_urban_roads["Urban"] == 1]  #

# flooded roads
with open(base_path / "parameters" / "flooded_road_links.json", "r") as f:
    flooded_road_list = json.load(f)


# %%
# select major roads
road_link_file, road_node_file = func.select_partial_roads_osm(
    road_links=osm_links,
    road_nodes=osm_nodes,
    col_name="asset_type",
    list_of_values=[
        "road_motorway",  # 120 km/h -> 70 mph
        "road_trunk",  # 120 km/h -> 70 mph
        "road_primary",  # 100 km/h -> 60 mph
        "road_secondary",  # 80 km/h -> 50 mph
        "road_bridge",  # 80 km/h -> 50 mph
    ],  # ["A Road", "B Road", "Motorway"],
)

# %%
# fillna (using the most frequent value):
# tag_maxspeed:
road_link_file["free_flow_speeds"] = road_link_file.tag_maxspeed * cons.CONV_KM_TO_MILE
# Mapping of asset types to default free flow speeds (mph)
default_speeds = {
    "road_motorway": 70.0,
    "road_trunk": 70.0,
    "road_primary": 60.0,
    "road_secondary": 50.0,
    "road_bridge": 50.0,
}
# Fill NaN values based on the mapping
for asset_type, speed in default_speeds.items():
    road_link_file.loc[road_link_file.asset_type == asset_type, "free_flow_speeds"] = (
        road_link_file.loc[
            road_link_file.asset_type == asset_type, "free_flow_speeds"
        ].fillna(speed)
    )

# tag_lanes:
default_passengers_per_lane = {
    "road_motorway": 16000,  # 15746
    "road_trunk": 10000,  # 7820
    "road_bridge": 10000,  # 8413
    "road_primary": 5000,  # 4427
    "road_secondary": 2000,  # 1772
}

road_link_file["passengers_per_lane"] = road_link_file.asset_type.map(
    default_passengers_per_lane
)

# %%
# classify the selected major road links into urban/suburban
urban_mask = func.create_urban_mask(etisplus_urban_roads)
road_link_file = func.label_urban_roads(road_link_file, urban_mask)

# %%
# drop disrupted roads if necessary (optional)
user_input = input("Please choose a scenario (base/flooded): ")
if user_input != "base" and user_input != "flooded":
    print("Error: please check the scenario input!")
if user_input == "flooded":
    road_link_file = road_link_file[~road_link_file.e_id.isin(flooded_road_list)]
    road_link_file.reset_index(drop=True, inplace=True)

# %%
# attach toll charges to the selected major roads
tolls = pd.read_csv(base_path / "networks" / "road" / "tolls_osm.csv")
road_link_file["average_toll_cost"] = 0
tolls_mapping = (
    tolls.iloc[1:, :].set_index("id")["Average_cost (pounds/passage)"].to_dict()
)
road_link_file["average_toll_cost"] = road_link_file["id"].apply(
    lambda x: tolls_mapping.get(x, 0)
)

# %%
# export road links and nodes
road_node_file.to_parquet(
    base_path / "networks" / "road" / "osm_road_node_file.geoparquet"
)
road_link_file.to_parquet(
    base_path / "networks" / "road" / "osm_road_link_file.geoparquet"
)
