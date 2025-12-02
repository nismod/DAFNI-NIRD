# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd
import igraph
from utils import get_flow_on_edges
import warnings

warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")

# %%
# load od matrix
od = pd.read_csv(
    INPUT_PATH_MACC / "data" / "incoming" / "rails" / "od_adjusted_updated.csv"
)
od = od[od["flows_adjusted"] > 0].reset_index(drop=True)

# %%
# load the rail edges and nodes
nodes = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_nodes.gpq"
)
stations = nodes[nodes.station_label == "train_station"].reset_index()

edges = gpd.read_parquet(
    INPUT_PATH_MACC
    / "data"
    / "incoming"
    / "rails"
    / "NWR_TrackModel 20250305"
    / "nationrail_edges_with_speed.gpq"
)

# create a railway network (igraph.Graph)
edges["weight"] = edges.geometry.length / 1000 / edges.speed_kmh  # hour
graph_df = edges[["from_node", "to_node", "edge_id", "weight"]]
network = igraph.Graph.TupleList(
    graph_df.itertuples(index=False),
    edge_attrs=list(graph_df.columns)[2:],
    directed=False,
)

# %%
temp = od.groupby("origin_node_idx", as_index=False).agg(
    {"destination_node_idx": list, "flows_adjusted": list}
)
origins = temp["origin_node_idx"].tolist()
destinations = temp["destination_node_idx"].tolist()
flows = temp["flows_adjusted"].tolist()

# %%
list_of_spath = []
for i in range(len(origins)):
    paths = network.get_shortest_paths(
        v=origins[i],
        to=destinations[i],
        weights="weight",
        mode="out",
        output="epath",
    )
    list_of_spath.append((origins[i], destinations[i], paths, flows[i]))

temp_flow_matrix = pd.DataFrame(
    list_of_spath, columns=["origin", "destination", "path", "flow"]
).explode(["destination", "path", "flow"])

# %%
edges["edge_idx"] = edges["edge_id"].apply(
    lambda x: network.es["edge_id"].index(x)
)  # 49s
# %%
temp_edge_flow = get_flow_on_edges(temp_flow_matrix, "edge_idx", "path", "flow")

# to merge flows back to edges
edges = edges.merge(
    temp_edge_flow[["edge_idx", "flow"]], on="edge_idx", how="left"
).fillna(0)

# edges.to_parquet(
#     INPUT_PATH_MACC / "data" / "incoming" / "rails" / "flows_undirected_updated.gpq"
# )
