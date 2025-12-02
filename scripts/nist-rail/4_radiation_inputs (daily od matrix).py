# %%
from pathlib import Path
import geopandas as gpd
import pandas as pd
import igraph
import json

import warnings

warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")

# %%
# load rail network edges and nodes
# attach node_id to train stations
nodes = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)
stations = nodes[nodes.station_label == "train_station"].reset_index(drop=True)

edges = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_edges_with_speed.gpq"
)

# %%
# create a railway network (igraph.Graph)
edges["weight"] = edges.geometry.length / 1000 / edges.speed_kmh  # hour
graph_df = edges[["from_node", "to_node", "edge_id", "weight"]]
network = igraph.Graph.TupleList(
    graph_df.itertuples(index=False),
    edge_attrs=list(graph_df.columns)[2:],
    directed=False,
)

# %%
# load origin_destinations_dict
selected_date = "2025-05-19"  # update this date as needed
with open(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "timetable"
    / f"origin_to_destinations_{selected_date}.json",
    "rb",
) as f:
    origin_destinations_dict = json.load(f)

origin_nodes_tiploc = list(origin_destinations_dict.keys())
destinations_nodes_tiploc = [
    origin_destinations_dict[origin] for origin in origin_nodes_tiploc
]

# load estimated station outflows and inflows
usage_dates = pd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "timetable"
    / "station_daily_out_in_flows.pq"
)
usage_dates = usage_dates.merge(
    stations[["TIPLOC", "node_id"]], on="TIPLOC", how="left"
)

# %%
# Generate inputs for Radiation Model
network_nodes = network.vs["name"]
station_node_id_map = stations.set_index("TIPLOC")["node_id"].to_dict()
nodeid_idx_map = {}
for idx in range(len(network_nodes)):
    nodeid_idx_map[network_nodes[idx]] = idx

# covert destination list from tiploc to node_id
destinations_node_id = [
    [station_node_id_map[tiploc] for tiploc in sublist if tiploc in station_node_id_map]
    for sublist in destinations_nodes_tiploc
]
# convert destination list from node_id to node_idx (# of node in network)
destinations_idx = [
    [nodeid_idx_map[node_id] for node_id in sublist if node_id in nodeid_idx_map]
    for sublist in destinations_node_id
]
node_id = usage_dates.node_id.tolist()  # all stations (with usage information)

# %%
# node index in network
node_idx = [network_nodes.index(node_name) for node_name in node_id]

# population (inflows)
population_day = usage_dates[selected_date]  # inflow of all stations

# outflows
tot_outflow = [
    row[selected_date] if row["TIPLOC"] in origin_nodes_tiploc else 0
    for idx, row in usage_dates.iterrows()
]  # outflow of origin stations

# searching radius
list_of_destinations = [
    (
        destinations_idx[origin_nodes_tiploc.index(row["TIPLOC"])]
        if row["TIPLOC"] in origin_nodes_tiploc
        else []
    )
    for idx, row in usage_dates.iterrows()
]

temp_df = pd.DataFrame(
    {
        "node_idx": node_idx,
        "population": population_day,
        "outflow": tot_outflow,
        "destinations": list_of_destinations,
    }
)
temp_df = temp_df[temp_df.outflow > 0].reset_index()

# %%
temp_df.to_csv(
    INPUT_PATH_MACC / "data" / "incoming" / "rails" / "radiation_inputs.csv",
    index=False,
)
