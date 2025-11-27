"""Road network preparation: nodes and links
- Extract Motorways, A roads and B roads
- Classify road links to urban/suburban roads
- Attach toll costs to road links
- Drop the disrupted roads (optional: for disruption analysis)
"""

# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd  # type: ignore

from nird.utils import load_config
import nird.road as func
import warnings
import json

warnings.simplefilter("ignore")

base_path = Path(load_config()["paths"]["base_path"])

# %%
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
etisplus_urban_roads = etisplus_urban_roads[etisplus_urban_roads["Urban"] == 1]  #

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

# attach toll charges to the selected major road links (£/car trip)
tolls = pd.read_csv(base_path / "networks" / "road" / "tolls.csv")
road_link_file["average_toll_cost"] = 0
tolls_mapping = (
    tolls.iloc[1:, :].set_index("e_id")["Average_cost (£/passage)"].to_dict()
)
road_link_file["average_toll_cost"] = road_link_file["e_id"].apply(
    lambda x: tolls_mapping.get(x, 0)
)
road_link_file.loc[
    road_link_file.road_classification_number == "M6 TOLL", "average_toll_cost"
] = 8.0

# %%
# (optional) drop disrupted roads if necessary (for disruption analysis)
with open(base_path / "parameters" / "flooded_road_links.json", "r") as f:
    flooded_road_list = json.load(f)

user_input = input("Please choose a scenario (base/flooded): ")
if user_input != "base" and user_input != "flooded":
    print("Error: please check the scenario input!")
if user_input == "flooded":
    road_link_file = road_link_file[~road_link_file.e_id.isin(flooded_road_list)]
    road_link_file.reset_index(drop=True, inplace=True)

# %%
# export road links and nodes
road_node_file.to_parquet(base_path / "networks" / "road" / "road_node_file.geoparquet")
road_link_file.to_parquet(base_path / "networks" / "road" / "road_link_file.geoparquet")
