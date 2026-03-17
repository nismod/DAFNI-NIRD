# %%
from pathlib import Path
import pandas as pd
import geopandas as gpd
import igraph
from nird.utils import get_flow_on_edges, load_config
import warnings

warnings.simplefilter("ignore")

INPUT_PATH_MACC = Path(r"C:\Oxford\Research\MACCHUB\local")
nist_path = Path(load_config()["paths"]["nist_path"])
nist_path = nist_path / "incoming" / "20260216 - inputs to OxfUni models"

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
# use stations and edges
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
edges["edge_idx"] = edges["edge_id"].apply(
    lambda x: network.es["edge_id"].index(x)
)  # 49s

edge_idx_to_id_dict = edges.set_index("edge_idx")["edge_id"].to_dict()
node_idx_to_id_dict = dict(enumerate(network.vs["name"]))

# %%
# load od matrix
# od = pd.read_csv(INPUT_PATH_MACC / "data" / "incoming" / "rails" / "od.csv")
# od = od[od["flows_adjusted"] > 0].reset_index(drop=True)
od_ppp = pd.read_parquet(nist_path / "outputs" / "rails" / "od_oa_ppp_estimates.pq")
od_hhh = pd.read_parquet(nist_path / "outputs" / "rails" / "od_oa_hhh_estimates.pq")

# generate future od
od_ppp_30 = od_ppp[["origin_node_idx", "destination_node_idx", "flows_30"]]
od_ppp_50 = od_ppp[["origin_node_idx", "destination_node_idx", "flows_50"]]
od_hhh_30 = od_hhh[["origin_node_idx", "destination_node_idx", "flows_30"]]
od_hhh_50 = od_hhh[["origin_node_idx", "destination_node_idx", "flows_50"]]


# %%
def flow_simu(od, col):
    temp = od.groupby("origin_node_idx", as_index=False).agg(
        {"destination_node_idx": list, col: list}
    )
    origins = temp["origin_node_idx"].tolist()
    destinations = temp["destination_node_idx"].tolist()
    flows = temp[str(col)].tolist()

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

    return temp_flow_matrix


# %%
# estimate edge flows
od_inputs = {
    "ppp_30": od_ppp_30,
    "ppp_50": od_ppp_50,
    "hhh_30": od_hhh_30,
    "hhh_50": od_hhh_50,
}
temp_flow_matrix_dict = {}
temp_edge_flow_dict = {}

for name, od_df in od_inputs.items():
    temp_flow_matrix = flow_simu(od_df, "flows_" + name.split("_")[-1])
    temp_flow_matrix_dict[name] = temp_flow_matrix
    temp_edge_flow = get_flow_on_edges(temp_flow_matrix, "edge_idx", "path", "flow")
    temp_edge_flow_dict[name] = temp_edge_flow
    col = f"flow_{name}"
    edges = edges.merge(
        temp_edge_flow[["edge_idx", "flow"]].rename(columns={"flow": col}),
        on="edge_idx",
        how="left",
    )
    edges[col] = edges[col].fillna(0.0)
    edges[col] = edges[col] / 2  # half the flows

# %%
# save results
# edges.to_parquet(INPUT_PATH_MACC / "data" / "incoming" / "rails" / "flows.gpq")
clip = gpd.read_parquet(
    nist_path
    / "incoming"
    / "20260216 - inputs to OxfUni models"
    / "outputs"
    / "rails"
    / "base_flows_20251219_monday.gpq"
)

base_flows = gpd.read_parquet(
    nist_path
    / "incoming"
    / "20260216 - inputs to OxfUni models"
    / "outputs"
    / "rails"
    / "rail_edges_with_capacity.gpq"
)  # the baseline flows have been already halved; capacity?

# %%
edges = edges.merge(clip[["edge_id", "within_clip"]], on="edge_id", how="left")
edges = edges.merge(
    base_flows[["edge_id", "baseline_flow", "capacity"]], on="edge_id", how="left"
)
edges["gain_ppp_30"] = edges["flow_ppp_30"] - edges["baseline_flow"]
edges["gain_hhh_30"] = edges["flow_hhh_30"] - edges["baseline_flow"]
edges["gain_ppp_50"] = edges["flow_ppp_50"] - edges["baseline_flow"]
edges["gain_hhh_50"] = edges["flow_hhh_50"] - edges["baseline_flow"]
edges["vc_ppp_30"] = edges["flow_ppp_30"] / edges["capacity"]
edges["vc_hhh_30"] = edges["flow_hhh_30"] / edges["capacity"]
edges["vc_ppp_50"] = edges["flow_ppp_50"] / edges["capacity"]
edges["vc_hhh_50"] = edges["flow_hhh_50"] / edges["capacity"]

# %%
edges.to_parquet(
    nist_path
    / "incoming"
    / "20260216 - inputs to OxfUni models"
    / "outputs"
    / "rails"
    / "future_flows.gpq"
)
