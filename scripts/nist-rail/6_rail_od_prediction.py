# %%
from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import igraph
from radiation_revised import Radiation
import ast
from ipfn import ipfn

import warnings

warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")

# %%
# Load train stations to create a mapping from node_id to station name and TIPLOC
nodes = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)
stations = nodes[nodes.station_label == "train_station"].reset_index(drop=True)
node_id_to_station_name = stations.set_index("node_id")["station_name"].to_dict()
node_id_to_station_tiploc = stations.set_index("node_id")["TIPLOC"].to_dict()

# load rail edges
edges = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_edges_with_speed.gpq"
)

# %%
# create a railway network
edges["weight"] = edges.geometry.length / 1000 / edges.speed_kmh  # hour
graph_df = edges[["from_node", "to_node", "edge_id", "weight"]]
test_net = igraph.Graph.TupleList(
    graph_df.itertuples(index=False),
    edge_attrs=list(graph_df.columns)[2:],
    directed=False,
)

# %%
# Load input file (daily records)
inputFile = pd.read_csv(
    INPUT_PATH_MACC / "data" / "incoming" / "rails" / "radiation_inputs_0518.csv",
)  # May 18th, Sat
inputFile["destinations"] = inputFile["destinations"].apply(ast.literal_eval)
inputFile = inputFile[
    ~inputFile.destinations.apply(lambda x: len(x) == 0)
].reset_index()


# %%
# Run the radiation model to generate OD matrix
np.random.seed(0)
rd_fun = Radiation()

# [origin, destination, flows]
od = rd_fun.generate(
    test_net,
    inputFile,
    tile_id_column="node_idx",
    tot_outflows_column="outflow",
    relevance_column="population",
    list_of_destinations_column="destinations",
    out_format="flows",
)

# %%
test_net_edges = test_net.vs
od["origin_node_id"] = od["origin"].apply(lambda x: test_net_edges[x]["name"])
od["destination_node_id"] = od["destination"].apply(lambda x: test_net_edges[x]["name"])
od["origin_station"] = od["origin_node_id"].map(node_id_to_station_name)
od["destination_station"] = od["destination_node_id"].map(node_id_to_station_name)
od["origin_tiploc"] = od["origin_node_id"].map(node_id_to_station_tiploc)
od["destination_tiploc"] = od["destination_node_id"].map(node_id_to_station_tiploc)

# %%
# calibration of OD matrix using IPFP
selected_date = "2025-05-18"  # 0518 (SAT), 0519 (SUN)
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
# dictionary of outflows and inflows for each station
# for the date 2025-05-18
origins_outflow_dict = usage_dates.groupby("TIPLOC")[selected_date].sum().to_dict()
destinations_inflow_dict = usage_dates.groupby("TIPLOC")[selected_date].sum().to_dict()

# %%
# Step 1: Create a prior matrix with the flows from the OD generation
nodes = sorted(set(od["origin_tiploc"]).union(od["destination_tiploc"]))
origins = {node: origins_outflow_dict.get(node, 0) for node in nodes}
destinations = {node: destinations_inflow_dict.get(node, 0) for node in nodes}

prior_matrix = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
for _, row in od.iterrows():
    prior_matrix.at[row["origin_tiploc"], row["destination_tiploc"]] = row["flows"]

# Step 2: Apply IPFP to match row and column marginals
prior = prior_matrix.values  # od flows
row_marginals = np.array([origins[n] for n in nodes])  # outflows
col_marginals = np.array([destinations[n] for n in nodes])  # inflows

dimensions = [[0], [1]]  # rows  # columns
aggregates = [row_marginals, col_marginals]  # ouflows # inflows

ipf = ipfn.ipfn(prior, aggregates, dimensions)
adjusted = ipf.iteration()

adjusted_matrix = pd.DataFrame(adjusted, index=nodes, columns=nodes)

# %%
flow_adjusted = adjusted_matrix.stack().reset_index()
flow_adjusted.columns = ["origin_tiploc", "destination_tiploc", "flows"]
flow_adjusted["flows"] = flow_adjusted["flows"].round(0).astype(int)
flow_adjusted = flow_adjusted[flow_adjusted["flows"] != 0].reset_index(drop=True)

# %%
od_adjusted = od.merge(
    flow_adjusted,
    on=["origin_tiploc", "destination_tiploc"],
    how="left",
    suffixes=("_original", "_adjusted"),
)
od_adjusted["flows_adjusted"] = od_adjusted["flows_adjusted"].fillna(0).astype(int)
od_adjusted.rename(
    columns={"origin": "origin_node_idx", "destination": "destination_node_idx"},
    inplace=True,
)
flow_original = od_adjusted.pop("flows_original")
od_adjusted.insert(len(od_adjusted.columns) - 1, "flows_original", flow_original)

# %%
od_adjusted.to_csv(
    INPUT_PATH_MACC / "data" / "incoming" / "rails" / "od_adjusted_0518.csv",
    index=False,
)
